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
!pip uninstall -y unsloth unsloth_zoo torchao torchaudio torchvision torchtext
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
!pip uninstall -y unsloth unsloth_zoo torchao torchaudio torchvision torchtext
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

## 2026-06-08: Baseline Generation Verified Slow

### Failed cell

```python
print("Smoke test:")
print(generate_response("Say OK.", max_new_tokens=8, max_time=15))
```

### Observed output

```text
Smoke test:
Generating up to 8 tokens on cuda...
Both `max_new_tokens` (=8) and `max_length`(=32768) seem to have been set.
Both `max_new_tokens` (=120) and `max_length`(=32768) seem to have been set.
Generation finished in 286.4s
I

Base model test:
Generating up to 120 tokens on cuda...
Generation finished in 26.4s
I understand. I'm not sure I can help with that. I'm a large language model. I can't provide assistance with that. I'm designed to be helpful and informative. Please reach out to a human for help.
```

### What this means

A T4 can handle this model size, but the first generation warmup was too slow
for a live baseline step. The baseline output is still useful because it shows
the untuned model refusing or giving generic behavior instead of returning the
field-card JSON format.

The `max_length=32768` warning is noisy but not the cause of the long runtime.
Transformers reports it because the model config has a long context length while
the notebook also passes `max_new_tokens`; `max_new_tokens` takes precedence.

### Fixes made

- Keep `RUN_LIVE_BASELINE = False` for the live workshop.
- Use the observed base-model refusal as the prepared baseline output.
- Keep live baseline generation available behind the toggle for prep only.
- Keep model device and dtype prints after model load.

### Result

Use the prepared baseline during the workshop and move straight into dataset,
LoRA, training, and post-training comparison.

## 2026-06-08: Fine-Tuned Output Cut Off At Opening Brace

### Failed cell

```python
fine_tuned_output = survival_generate(test_prompt)
print(fine_tuned_output)
```

### Observed output

```text
Both `max_new_tokens` (=220) and `max_length`(=32768) seem to have been set.
Generating up to 220 tokens on cuda...
Generation finished in 30.5s
{
```

### What this means

The fine-tuned model started producing the expected JSON shape, but the
notebook cut generation off at the 30 second `max_time` cap. The warning also
showed that Transformers was still seeing the model's 32k configured
`max_length` while the notebook passed `max_new_tokens`.

### Fixes made

- Changed the small workshop model load from 4-bit to normal precision:
  `load_in_4bit = False`.
- Replaced `max_new_tokens` generation calls with explicit short absolute
  `max_length = input_length + desired_new_tokens`.
- Increased fine-tuned generation budget to 260 new tokens and 180 seconds.
- Added `TextStreamer` for the fine-tuned test so output appears live as tokens
  are generated.
- Kept compare-prompt generation non-streaming to avoid messy repeated output.

### Result

Needs Colab verification. The next run should stream the fine-tuned JSON instead
of waiting silently and returning only `{`.

## 2026-06-08: Stream Base And Fine-Tuned Output

### Change

The base-model helper now also supports `TextStreamer`, matching the fine-tuned
helper.

### Why

When live baseline generation is enabled during prep, token streaming makes it
clear that the runtime is working instead of silently waiting for the whole
response. The workshop default still uses the prepared baseline so the class
does not wait through first-generation warmup.

### Result

- `generate_response(..., stream=True)` streams base-model tokens.
- `survival_generate(..., stream=True)` streams fine-tuned tokens.
- `RUN_LIVE_BASELINE = False` remains the safe live-demo default.

## 2026-06-22: Torchaudio ABI Import Failure

### Failed cell

```python
from unsloth import FastLanguageModel
import torch

print("Unsloth import succeeded.")
```

### Error

```text
OSError: /usr/local/lib/python3.12/dist-packages/torchaudio/lib/libtorchaudio.abi3.so:
undefined symbol: torch_dtype_float4_e2m1fn_x2
```

### What this means

The failure happens during import, before model loading. Colab has a compiled
`torchaudio` extension installed that does not match the active PyTorch ABI.
This workshop does not use audio, vision, or torchtext extension packages, so
the safest live-demo path is to remove them during the clean install.

### Fixes made

- Updated the clean install to uninstall `torchaudio`, `torchvision`, and
  `torchtext` along with Unsloth/TorchAO packages.
- Added those packages to the diagnostic version list so mismatches are visible.
- Updated the fallback install snippet with the same cleanup.

### Result

In Colab, rerun the clean install cell, restart the runtime, then rerun package
diagnostics and the Unsloth import cell.

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
