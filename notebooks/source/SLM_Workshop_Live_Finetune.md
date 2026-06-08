# SLM Workshop Live Fine-Tune

This notebook fine-tunes a small model into a Survival Field Card Generator.

The point is behavior control: turn a generic answer into a structured JSON-like
product response. This is not professional survival, medical, rescue, or
emergency advice.

## 1. Environment Check

Run this first. You want a GPU, ideally a T4.

```python
import torch, os, platform

print("Python:", platform.python_version())
print("CUDA available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
else:
    print("No GPU detected. Go to Runtime > Change runtime type > GPU.")
```

## 2. Clean Install

Run this once in a fresh Colab runtime.

```python
%%capture
!pip uninstall -y unsloth unsloth_zoo torchao
!pip install --upgrade pip
!pip install --no-cache-dir -U unsloth
!pip install --no-cache-dir -U transformers datasets accelerate peft trl bitsandbytes
```

After this cell finishes, restart the runtime once:

```text
Runtime -> Restart runtime
```

Then rerun the environment check and continue below.

## 3. Package Diagnostics

This captures the package versions that matter for debugging.

```python
import importlib.metadata as md
import torch, platform

packages = ["torch", "torchao", "unsloth", "unsloth_zoo", "transformers", "trl", "peft", "bitsandbytes"]

print("Python:", platform.python_version())
print("Torch:", torch.__version__)
print("CUDA:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))

for package in packages:
    try:
        print(package, md.version(package))
    except md.PackageNotFoundError:
        print(package, "not installed")
```

## 4. Import Unsloth

If this cell fails with `_c10d_functional._wrap_tensor_autograd`, use the
fallback install section at the bottom of this notebook in a fresh runtime.

```python
from unsloth import FastLanguageModel
import torch

print("Unsloth import succeeded.")
```

## 5. Optional Hugging Face Login

Leave this off for the normal live path. Set `RUN_HF_LOGIN = True` only if a
model load cell later says Hugging Face access is required.

```python
RUN_HF_LOGIN = False

if RUN_HF_LOGIN:
    from huggingface_hub import notebook_login
    notebook_login()
else:
    print("Skipping Hugging Face login for the normal live path.")
```

## 6. Load Base Model

The primary model is Gemma 3 270M Instruct. If it fails because of access or package
compatibility, the code falls back to Qwen2.5 0.5B Instruct.

```python
max_seq_length = 1024
dtype = None
load_in_4bit = False

PRIMARY_MODEL = "unsloth/gemma-3-270m-it"
BACKUP_MODEL = "unsloth/Qwen2.5-0.5B-Instruct"

def load_model(model_name):
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = model_name,
        max_seq_length = max_seq_length,
        dtype = dtype,
        load_in_4bit = load_in_4bit,
    )
    FastLanguageModel.for_inference(model)
    return model, tokenizer

try:
    model_name = PRIMARY_MODEL
    model, tokenizer = load_model(model_name)
except Exception as error:
    print("Primary model failed:", repr(error))
    print("Trying backup model:", BACKUP_MODEL)
    model_name = BACKUP_MODEL
    model, tokenizer = load_model(model_name)

print("Loaded:", model_name)
first_parameter = next(model.parameters())
print("Model device:", first_parameter.device)
print("Model dtype:", first_parameter.dtype)
```

## 7. Generation Helper

```python
import time

if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
    tokenizer.pad_token = tokenizer.eos_token

def generate_response(prompt, max_new_tokens=120, max_time=90):
    messages = [
        {"role": "user", "content": prompt}
    ]

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    inputs = tokenizer([text], return_tensors="pt").to(device)
    target_length = inputs["input_ids"].shape[-1] + max_new_tokens

    print(f"Generating up to {max_new_tokens} tokens on {device}...")
    started_at = time.time()

    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_length=target_length,
            max_time=max_time,
            do_sample=False,
            use_cache=True,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )

    print(f"Generation finished in {time.time() - started_at:.1f}s")

    new_tokens = outputs[0][inputs["input_ids"].shape[-1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True)
```

## 8. Test The Base Model

For the live workshop, keep `RUN_LIVE_BASELINE = False`. This prints a prepared
baseline output so the demo does not stall if Colab spends several minutes on a
first-generation warmup. During prep, set it to `True` to test live baseline
generation.

```python
test_prompt = "I need to start a fire and everything is damp."

RUN_LIVE_BASELINE = False

prepared_base_output = """I understand. I'm not sure I can help with that. I'm a large language model. I can't provide assistance with that. I'm designed to be helpful and informative. Please reach out to a human for help."""

if RUN_LIVE_BASELINE:
    print("Smoke test:")
    print(generate_response("Say OK.", max_new_tokens=8, max_time=15))

    print("\nBase model test:")
    base_output = generate_response(test_prompt, max_new_tokens=120, max_time=30)
else:
    print("Using prepared base output from prep run.")
    base_output = prepared_base_output

print(base_output)
```

## 9. Create Survival Dataset

```python
survival_examples = [
    {
        "instruction": "I need to start a fire and everything is damp.",
        "response": {
            "scenario": "starting a fire in damp conditions",
            "priority": "create dry tinder and protect the flame from moisture and wind",
            "steps": [
                "Look for the driest material under bark, logs, dense branches, or inside dead standing wood.",
                "Shave damp sticks to expose the drier inner wood.",
                "Prepare a tinder bundle before attempting ignition.",
                "Build a small platform of sticks or bark to keep the tinder off wet ground.",
                "Add kindling gradually once the flame is stable."
            ],
            "safety_notes": [
                "Clear flammable debris around the fire area.",
                "Keep the fire small and controlled.",
                "Extinguish the fire completely before leaving."
            ],
            "common_mistakes": [
                "Trying to light large wet sticks first.",
                "Skipping tinder preparation.",
                "Building directly on wet ground."
            ],
            "confidence": "medium"
        }
    },
    {
        "instruction": "I have no lighter but I need a fire.",
        "response": {
            "scenario": "starting a fire without a lighter",
            "priority": "use available ignition sources and prepare tinder before attempting ignition",
            "steps": [
                "Check for matches, ferro rod, battery and steel wool, lens, or other ignition sources.",
                "Gather very fine tinder such as dry grass, bark fibers, paper, or wood shavings.",
                "Prepare kindling and larger fuel before creating a spark.",
                "Create a protected fire area shielded from wind.",
                "Transfer the spark or ember into the tinder bundle and feed it slowly."
            ],
            "safety_notes": [
                "Avoid unsafe chemical accelerants.",
                "Keep water, dirt, or sand nearby to extinguish the fire.",
                "Do not start a fire where it can spread."
            ],
            "common_mistakes": [
                "Making sparks before tinder is ready.",
                "Using damp tinder.",
                "Adding large fuel too early."
            ],
            "confidence": "medium"
        }
    },
    {
        "instruction": "I am lost and it is getting dark.",
        "response": {
            "scenario": "lost outdoors near nightfall",
            "priority": "stop moving, preserve energy, and prepare for visibility and shelter",
            "steps": [
                "Stop walking unless staying put is immediately dangerous.",
                "Mark your current location with visible signs.",
                "Check your phone battery, map, compass, or GPS if available.",
                "Prepare insulation from the ground and build a basic shelter if needed.",
                "Create visible signals such as bright clothing, reflective items, or a controlled signal fire if safe."
            ],
            "safety_notes": [
                "Avoid wandering in darkness.",
                "Stay dry and insulated to reduce cold exposure.",
                "Contact emergency services if signal is available."
            ],
            "common_mistakes": [
                "Continuing to walk without a plan.",
                "Ignoring temperature drop after sunset.",
                "Using phone battery carelessly."
            ],
            "confidence": "medium"
        }
    },
    {
        "instruction": "I need drinking water but I do not have a filter.",
        "response": {
            "scenario": "water collection without a filter",
            "priority": "reduce immediate dehydration risk while minimizing contamination risk",
            "steps": [
                "Look for the clearest moving water source available.",
                "Avoid water near roads, industry, dead animals, or heavy algae.",
                "Let sediment settle if the water is cloudy.",
                "Boil water if you have a safe way to heat it.",
                "Use purification tablets if available."
            ],
            "safety_notes": [
                "Boiling does not remove chemical contamination.",
                "Untreated water can contain pathogens.",
                "Do not drink seawater."
            ],
            "common_mistakes": [
                "Assuming clear water is safe.",
                "Drinking from stagnant pools first.",
                "Ignoring chemical contamination risks."
            ],
            "confidence": "medium"
        }
    },
    {
        "instruction": "It is cold and I do not have a tent.",
        "response": {
            "scenario": "cold exposure without a tent",
            "priority": "reduce heat loss from wind, wet clothing, and ground contact",
            "steps": [
                "Move out of wind if possible.",
                "Insulate yourself from the ground using leaves, branches, clothing, or gear.",
                "Keep your core dry and covered.",
                "Create a simple windbreak or lean-to with available materials.",
                "Use shared body heat only if appropriate and safe."
            ],
            "safety_notes": [
                "Wet clothing increases heat loss.",
                "Cold ground can drain body heat quickly.",
                "Seek help if shivering becomes severe or confusion begins."
            ],
            "common_mistakes": [
                "Lying directly on cold ground.",
                "Ignoring wind exposure.",
                "Waiting too long to build shelter."
            ],
            "confidence": "medium"
        }
    },
    {
        "instruction": "I twisted my ankle on a trail.",
        "response": {
            "scenario": "possible ankle sprain on trail",
            "priority": "prevent further injury and assess whether walking is safe",
            "steps": [
                "Stop and avoid putting full weight on the ankle immediately.",
                "Check for severe swelling, deformity, numbness, or inability to bear weight.",
                "Stabilize the ankle with a wrap, clothing, or trekking pole support if available.",
                "Elevate the ankle during rest if practical.",
                "Plan evacuation or assistance if walking is unsafe."
            ],
            "safety_notes": [
                "Severe pain, deformity, numbness, or inability to walk may require medical care.",
                "Do not force continued movement if the injury worsens.",
                "Keep warm while waiting for help."
            ],
            "common_mistakes": [
                "Trying to walk it off immediately.",
                "Removing footwear too early if swelling may prevent putting it back on.",
                "Ignoring signs of fracture."
            ],
            "confidence": "medium"
        }
    },
    {
        "instruction": "I need to signal for help from an open area.",
        "response": {
            "scenario": "signaling for rescue from an open area",
            "priority": "make yourself visible and create repeatable distress signals",
            "steps": [
                "Move to the most visible safe location nearby.",
                "Use bright clothing, reflective gear, smoke, or ground markings.",
                "Create three repeated signals, such as three fires, three whistle blasts, or three flashes.",
                "Use large ground symbols such as SOS if materials are available.",
                "Preserve battery if using a phone or flashlight."
            ],
            "safety_notes": [
                "Do not expose yourself to dangerous terrain for visibility.",
                "Avoid uncontrolled fires.",
                "Stay near your signal if it is safe."
            ],
            "common_mistakes": [
                "Making signals too small to see.",
                "Moving away from the signal site.",
                "Using battery power continuously."
            ],
            "confidence": "medium"
        }
    },
    {
        "instruction": "A storm is coming and I am still outside.",
        "response": {
            "scenario": "incoming storm while outdoors",
            "priority": "reduce exposure to lightning, wind, falling branches, and cold rain",
            "steps": [
                "Move away from exposed ridgelines, isolated tall trees, and open water.",
                "Look for lower, protected ground that is not a flood channel.",
                "Put on rain protection or improvise a cover before getting soaked.",
                "Secure loose gear.",
                "Wait for the most dangerous weather to pass before moving again."
            ],
            "safety_notes": [
                "Avoid sheltering under isolated tall trees.",
                "Stay out of dry stream beds or flood-prone areas.",
                "Cold rain can quickly lead to hypothermia."
            ],
            "common_mistakes": [
                "Waiting too long to prepare.",
                "Choosing a flood channel as shelter.",
                "Standing in exposed areas during lightning."
            ],
            "confidence": "medium"
        }
    }
]

print("Dataset examples:", len(survival_examples))
```

## 10. Format Examples

```python
import json
from datasets import Dataset

SYSTEM_MESSAGE = """You are a survival field card generator.
Return concise, structured JSON only.
Do not include markdown.
Do not include extra commentary.
Focus on safe, practical, general outdoor guidance.
If the situation is dangerous or medical, include appropriate safety notes.
"""

def make_user_content(user_prompt):
    return f"{SYSTEM_MESSAGE}\nUser request: {user_prompt}"

def make_training_text(example):
    user = example["instruction"]
    assistant = json.dumps(example["response"], indent=2)

    messages = [
        {"role": "user", "content": make_user_content(user)},
        {"role": "assistant", "content": assistant}
    ]

    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False
    )

texts = [make_training_text(example) for example in survival_examples]
dataset = Dataset.from_dict({"text": texts})

print(dataset[0]["text"])
print("Dataset size:", len(dataset))
```

## 11. Apply LoRA

```python
model = FastLanguageModel.get_peft_model(
    model,
    r = 16,
    target_modules = [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
    lora_alpha = 16,
    lora_dropout = 0,
    bias = "none",
    use_gradient_checkpointing = "unsloth",
    random_state = 3407,
    max_seq_length = max_seq_length,
)

print("LoRA adapters applied.")
```

## 12. Train

For a smoke test during prep, set `TRAINING_STEPS = 3`. For the live workshop,
use `TRAINING_STEPS = 40`.

```python
from trl import SFTConfig, SFTTrainer

TRAINING_STEPS = 40

training_args = SFTConfig(
    dataset_text_field = "text",
    max_seq_length = max_seq_length,
    packing = False,
    per_device_train_batch_size = 2,
    gradient_accumulation_steps = 4,
    warmup_steps = 5,
    max_steps = TRAINING_STEPS,
    learning_rate = 2e-4,
    fp16 = not torch.cuda.is_bf16_supported(),
    bf16 = torch.cuda.is_bf16_supported(),
    logging_steps = 1,
    optim = "adamw_8bit",
    weight_decay = 0.01,
    lr_scheduler_type = "linear",
    seed = 3407,
    output_dir = "outputs",
    report_to = "none",
)

trainer = SFTTrainer(
    model = model,
    train_dataset = dataset,
    processing_class = tokenizer,
    args = training_args,
)

trainer_stats = trainer.train()
```

## 13. Test Fine-Tuned Model

```python
from transformers import TextStreamer

FastLanguageModel.for_inference(model)

def survival_generate(prompt, max_new_tokens=260, max_time=180, stream=True):
    messages = [
        {"role": "user", "content": make_user_content(prompt)}
    ]

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    inputs = tokenizer([text], return_tensors="pt").to(device)
    target_length = inputs["input_ids"].shape[-1] + max_new_tokens
    streamer = None
    if stream:
        streamer = TextStreamer(
            tokenizer,
            skip_prompt=True,
            skip_special_tokens=True,
        )

    print(f"Generating up to {max_new_tokens} tokens on {device}...")
    started_at = time.time()

    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_length=target_length,
            max_time=max_time,
            do_sample=False,
            use_cache=True,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
            streamer=streamer,
        )

    print(f"Generation finished in {time.time() - started_at:.1f}s")

    new_tokens = outputs[0][inputs["input_ids"].shape[-1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True)

fine_tuned_output = survival_generate(test_prompt, stream=True)
```

## 14. Compare Prompts

```python
test_prompts = [
    "I need to start a fire and everything is damp.",
    "I am lost and it is getting dark.",
    "I need drinking water but I do not have a filter.",
    "A storm is coming and I am still outside."
]

for prompt in test_prompts:
    print("=" * 80)
    print("PROMPT:", prompt)
    print("-" * 80)
    output = survival_generate(prompt, max_new_tokens=220, max_time=180, stream=False)
    print(output)
```

## 15. Save Adapter Locally

```python
adapter_path = "survival_field_card_lora"

model.save_pretrained(adapter_path)
tokenizer.save_pretrained(adapter_path)

print("Saved adapter to:", adapter_path)
```

## 16. Optional: Save Adapter To Google Drive

```python
SAVE_TO_DRIVE = False

if SAVE_TO_DRIVE:
    from google.colab import drive
    import os, shutil

    drive.mount("/content/drive")

    drive_root = "/content/drive/MyDrive/slm_workshop"
    drive_adapter_path = f"{drive_root}/survival_field_card_lora"

    os.makedirs(drive_root, exist_ok=True)
    if os.path.exists(drive_adapter_path):
        shutil.rmtree(drive_adapter_path)
    shutil.copytree(adapter_path, drive_adapter_path)

    print("Copied adapter to:", drive_adapter_path)
else:
    print("Skipping Drive backup. Set SAVE_TO_DRIVE = True to enable it.")
```

## 17. Optional: Save Dataset JSONL

```python
with open("survival_dataset.jsonl", "w") as file:
    for example in survival_examples:
        file.write(json.dumps(example) + "\n")

print("Saved survival_dataset.jsonl")
```

## 18. Fallback Install For Import Error

Only use this in a fresh runtime if the normal install still produces:

```text
AttributeError: '_OpNamespace' '_c10d_functional' object has no attribute '_wrap_tensor_autograd'
```

Copy this into a new cell only if you need it. It is intentionally not an
executable notebook cell so `Run all` does not uninstall packages after training.

```text
%%capture
!pip uninstall -y unsloth unsloth_zoo torchao
!pip install --upgrade pip
!pip install --no-cache-dir torchao==0.13.0
!pip install --no-cache-dir -U unsloth transformers datasets accelerate peft trl bitsandbytes
```

Restart runtime, then retry the package diagnostics and Unsloth import cells.
