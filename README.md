# tune.rl.py

```python
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer
from trl import PPOConfig, PPOTrainer, AutoModelForCausalLMWithValueHead
from transformers import StoppingCriteria, StoppingCriteriaList
from transformers import TextStreamer
import json,random,sys
random.seed()
def get_truly_random_seed_through_os():
    """
    Usually the best random sample you could get in any programming language is generated through the operating system.
    In Python, you can use the os module.

    source: https://stackoverflow.com/questions/57416925/best-practices-for-generating-a-random-seeds-to-seed-pytorch/57416967#57416967
    """
    RAND_SIZE = 4
    random_data = os.urandom(
        RAND_SIZE
    )  # Return a string of size random bytes suitable for cryptographic use.
    random_seed = int.from_bytes(random_data, byteorder="big")
    return random_seed

seed = get_truly_random_seed_through_os()
set_seed(seed)

json_file = sys.argv[1]
with open(json_file,"r") as jf:
    config = json.load(jf)

MODEL = config["MODEL"]
TRAIN_FILE = config["TRAIN_FILE"]
OUTPUT_DIR = config["OUTPUT_DIR"]
OVERWRITE = bool(config["OVERWRITE"])
BATCH_SIZE = int(config['BATCH_SIZE'])
EPOCHS = int(config["EPOCHS"])
LRATE = float(config["LRATE"])
STEPS = int(config["STEPS"])
LOAD_4BIT = config["LOAD_4BIT"].lower() == "true"
LOAD_8BIT = config["LOAD_8BIT"].lower() == "true"
FULLTUNE = config["FULLTUNE"].lower() == "true"
OPTIMIZER = config["OPTIM"]
MAXSEQ= int(config["MAXSEQ"])
if("PERCENT" in config):
    PERCENT = int(config["PERCENT"])
else:
    PERCENT = 100
if("NUM_SAMPLES" in config):
    NUM_SAMPLES = int(config["NUM_SAMPLES"])
else:
    NUM_SAMPLES=0
if("SELECT_OUTPUT" in config):
    SELECT_OUTPUT = config["SELECT_OUTPUT"]
else:
    SELECT_OUTPUT = "output"
if("SHUFFLE" in config):
    os.system("python " + config["SHUFFLE"])



with open(TRAIN_FILE,"r") as jf:
    db = json.load(jf)


model_name = MODEL

# Tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# RL-ready model (with value head)
model = AutoModelForCausalLMWithValueHead.from_pretrained(model_name,device_map="auto")

# PPO config
config = PPOConfig(
    model_name=model_name,
    learning_rate=LRATE,
    batch_size=BATCH_SIZE,
    mini_batch_size=1,
    kl_penalty="kl",
    target_kl=0.05,
)

# PPO Trainer
ppo_trainer = PPOTrainer(config=config, model=model, tokenizer=tokenizer)

print("✅ PPOTrainer initialized with GPT-2 + value head")

import difflib

def compute_reward(generated_text: str, target_text: str) -> float:
    # Normalize
    g = generated_text.strip()
    t = target_text.strip()

    if not g:
        return -1.0  # empty is garbage
    if(g == ''): return -1.0
    if(g.isspace()): return -1.0
    if(g[0] == '.'): return -1.0
    
    # Use sequence matcher for quick similarity [0,1]
    sim = difflib.SequenceMatcher(None, g, t).ratio()

    # Map [0,1] → [-1,1]
    reward = 2 * sim - 1.0  
    if(reward < -1.0): reward = -1.0
    if(reward > 1.0): reward = 1.0
    return reward



# Custom stopping criteria to stop when the <|endoftext|> token is generated
class StopOnEndOfText(StoppingCriteria):
    def __init__(self, eos_token_id):
        self.eos_token_id = eos_token_id

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        # Check if the last token generated is the eos_token_id
        return input_ids[0, -1] == self.eos_token_id

# Create an instance of the stopping criteria with the model's EOS token
eos_token_id = tokenizer.eos_token_id
stopping_criteria = StoppingCriteriaList([StopOnEndOfText(eos_token_id)])
textstreamer = TextStreamer(tokenizer, skip_prompt = True)



from torch.optim import AdamW
optimizer = AdamW(model.parameters(), lr=5e-5)



random.shuffle(db)

step = 0
for i in db:
    step = step + 1
    prompt = '### Prompt:\n\n'
    for k,v in i.items():
        if(k == "output"): continue
        if(type(v) is list):
            prompt = prompt + k + ": " + ','.join(v) + "\n"    
        else:
            prompt = prompt + k + ": " + v + "\n"
    prompt = prompt.strip() + "\n\n### Response:\n\n"
    target = f"""'''{i["output"][0]}'''<|endoftext|>"""
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")    
    target = tokenizer(target, return_tensors="pt").to("cuda")    
    target_len = len(target["input_ids"][0])
    
    response_ids = model.generate(
        **inputs,
        streamer = textstreamer,
        max_new_tokens=target_len,                 # or clamp to a small cap like 8-16 for short answers
        do_sample=True,
        top_p=0.9,
        temperature=0.7,
        no_repeat_ngram_size=3,                    # ✨ prevents “In your world…” loops
        repetition_penalty=1.15,                   # mild anti-repeat
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id,
    )
    
    response_text = tokenizer.decode(response_ids[0], skip_special_tokens=True)
    target_text = tokenizer.decode(target["input_ids"][0], skip_special_tokens=True)
    # Strip prompt portion if the model repeats it
    if response_text.startswith(prompt):
        response_only = response_text[len(prompt):].strip()
    else:
        response_only = response_text
    response_text = response_only
    

    reward = compute_reward(response_text, target_text)
    #if(reward < 0):
    #    continue
    reward_tensor = torch.tensor(reward, dtype=torch.float).to("cuda")
    if(reward < 0): continue
    print(f"Reward: {reward}")
    
    response_ids = torch.cat([
        response_ids["input_ids"],
        torch.tensor([[tokenizer.eos_token_id]].to("cuda"))
    ], dim=1)

    # ✅ Pass rewards as tensors
    ppo_trainer.step(
        [inputs["input_ids"][0]], 
        [response_ids[0]], 
        [reward_tensor]
    )
    

"""
Great question 👍 — once you’ve done PPO training with TRL, you have two options for saving:

---

## 🔹 1. Save the whole RL-tuned model (policy + value head)

```python
save_dir = "./ppo-gpt2-cosine"
ppo_trainer.save_pretrained(save_dir)
tokenizer.save_pretrained(save_dir)
```

* This saves:

  * GPT-2 weights (policy head)
  * the value head (needed if you want to keep training with PPO later)
  * config + tokenizer
* Later you can reload with:

```python
from trl import AutoModelForCausalLMWithValueHead
from transformers import AutoTokenizer

model = AutoModelForCausalLMWithValueHead.from_pretrained(save_dir)
tokenizer = AutoTokenizer.from_pretrained(save_dir)
```

---

## 🔹 2. Save just the language model (for inference only)

If you don’t need the value head anymore (just want a text-generating GPT-2):

```python
save_dir = "./ppo-gpt2-merged"

# take the underlying GPT-2 policy out of the wrapper
model.pretrained_model.save_pretrained(save_dir)
tokenizer.save_pretrained(save_dir)
```

* This way, you can reload it like a normal GPT-2:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("./ppo-gpt2-merged")
tokenizer = AutoTokenizer.from_pretrained("./ppo-gpt2-merged")
```

* ✅ Best for inference with `.generate()`.
* ❌ You can’t continue PPO training with it (value head is gone).

---

## 🔹 3. (Optional) Save checkpoints during training

If you want to checkpoint periodically:

```python
ppo_trainer.save_pretrained(f"./checkpoints/step-{step}")
```

---

## ✅ Bottom line

* Use `ppo_trainer.save_pretrained()` if you want to keep the full PPO model (resume RL later).
* Use `model.pretrained_model.save_pretrained()` if you only want a normal GPT-2 for inference.

---

👉 Do you want me to show you a pattern where you **train with PPO, save checkpoints periodically, and merge only the final model for inference**? That way you keep both training flexibility and an inference-ready GPT-2.


outputs = model.generate(
    **inputs,
    max_new_tokens=30,
    do_sample=True,
    top_p=0.9,
    temperature=0.7
)

print("Generated:", tokenizer.decode(outputs[0], skip_special_tokens=True))
"""
save_dir = model_name
model.pretrained_model.save_pretrained(save_dir)
tokenizer.save_pretrained(save_dir)
```
