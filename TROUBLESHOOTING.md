# Troubleshooting Log

Use this file while testing the Colab. Record the exact cell, error, attempted
fix, result, and commit hash after each change.

## 2026-06-08: Unsloth Import Failure

### Failed cell

```python
from unsloth import FastLanguageModel
import torch

max_seq_length = 1024
dtype = None
load_in_4bit = True

model_name = "unsloth/gemma-3-270m"

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = model_name,
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = load_in_4bit,
)

FastLanguageModel.for_inference(model)

print("Loaded:", model_name)
```

### Error

```text
AttributeError: '_OpNamespace' '_c10d_functional' object has no attribute '_wrap_tensor_autograd'
```

### What this means

The failure happens on `from unsloth import FastLanguageModel`, before the model
is loaded. Treat it as a Colab package compatibility issue, likely in the
Unsloth/PyTorch/TorchAO dependency stack.

### First fix path

Start from a clean Colab runtime and run:

```python
%%capture
!pip uninstall -y unsloth unsloth_zoo torchao
!pip install --upgrade pip
!pip install --no-cache-dir -U unsloth
!pip install --no-cache-dir -U transformers datasets accelerate peft trl bitsandbytes
```

Then restart the runtime:

```text
Runtime -> Restart runtime
```

After restart, rerun the environment check, package diagnostics, and Unsloth
import cell.

### Fallback fix path

If the same `_c10d_functional._wrap_tensor_autograd` error persists in a fresh
runtime, try pinning TorchAO before reinstalling the training stack:

```python
%%capture
!pip uninstall -y unsloth unsloth_zoo torchao
!pip install --upgrade pip
!pip install --no-cache-dir torchao==0.13.0
!pip install --no-cache-dir -U unsloth transformers datasets accelerate peft trl bitsandbytes
```

Then restart the runtime and retry:

```python
from unsloth import FastLanguageModel
```

### Result

Not yet verified in Colab from this repo. Record the final package versions and
commit hash after the successful run.

## 2026-06-08: Live Rehearsal Static Fixes

### What was checked

Walked through the repo as an attendee would:

- README follow-along path
- generated Colab notebook cell order
- code-cell syntax after filtering Colab magics
- run-all safety for optional login, Drive backup, and fallback install cells

### Issues found

- The README did not include a direct Colab URL.
- The primary model used `unsloth/gemma-3-270m`, but the live chat-template path
  should use the instruction-tuned model `unsloth/gemma-3-270m-it`.
- The optional Hugging Face login could block a live run if executed blindly.
- The fallback install cell would run during `Run all` and uninstall packages
  after the main demo flow.
- The generation helpers decoded the full output, including the prompt.
- Separate `system` role messages can be brittle across model chat templates.

### Fixes made

- Added a direct Colab link in the README.
- Switched the primary model to `unsloth/gemma-3-270m-it`.
- Gated Hugging Face login behind `RUN_HF_LOGIN = False`.
- Gated Drive backup behind `SAVE_TO_DRIVE = False`.
- Converted the fallback install to a copy-only text snippet.
- Trimmed generated output to new tokens only.
- Folded the field-card instruction into the user turn for training and
  fine-tuned inference.

### Result

Local static checks passed. Full GPU execution still needs to be verified in
Colab.

## 2026-06-08: Base Model Generation Hung

### Failed cell

```python
test_prompt = "I need to start a fire and everything is damp."

base_output = generate_response(test_prompt)
print(base_output)
```

### Error

The cell ran for more than 5 minutes without producing output.

### What this means

This is most likely an inference configuration problem or a runtime performance
problem, not a training problem. The original base test asked for up to 400 new
tokens with sampling and no timing printout, so it was hard to tell whether
generation was slow, stuck, or just producing a long answer.

### Fixes made

- Added a tiny `Say OK.` smoke test before the real base prompt.
- Lowered base generation to 120 tokens.
- Lowered fine-tuned generation to 220 tokens.
- Switched live-demo inference to deterministic `do_sample=False`.
- Added `max_time` caps to generation calls.
- Added elapsed-time logging.
- Used the available runtime device instead of hardcoding `cuda`.
- Set `pad_token` from `eos_token` when needed.

### Result

Not yet verified in Colab. If the smoke test hangs, interrupt the cell and
check whether the runtime actually has a GPU before continuing.

## 2026-06-08: Baseline Generation Still Too Slow

### Failed cell

```python
print("Smoke test:")
print(generate_response("Say OK.", max_new_tokens=8, max_time=15))
```

### Output before interrupting

```text
Smoke test:
Generating up to 8 tokens on cuda...
Both `max_new_tokens` (=8) and `max_length`(=32768) seem to have been set.
```

### What this means

The slowdown happens before even a tiny baseline response completes, so it is
not caused by a long prompt or a large token cap. A T4 should be able to handle
this model size, but the first generation can still be blocked by runtime,
kernel, quantization, or library warmup behavior. That makes live baseline
generation too risky for the workshop.

### Fixes made

- Added model device and dtype prints after loading.
- Added `RUN_LIVE_BASELINE = False`.
- Added a prepared base-model output for the live demo.
- Kept live baseline generation available behind the toggle for prep testing.

### Result

The workshop can still show before/after behavior without waiting on live base
generation. The live training and post-training comparison remain the important
demo path.

## New Entry Template

### Date

### Failed cell

```python

```

### Error

```text

```

### Attempted fix

### Result

### Commit
