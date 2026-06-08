# Steel City Workshop: Live SLM Fine-Tuning Demo

This repo is the follow-along guide for a live workshop on small language model
fine-tuning in Google Colab.

The demo fine-tunes a small model into a **Survival Field Card Generator**. The
goal is not to teach the model survival expertise. The goal is to show how a
small model can be tuned to produce a narrow, repeatable product behavior.

> Safety note: this is a fine-tuning demo, not professional survival, medical,
> rescue, or emergency advice. A real product would need expert-reviewed data,
> retrieval or citations, evals, monitoring, and stronger safety boundaries.

## What You Will Build

Prompt:

```text
I need to start a fire and everything is damp.
```

Target output shape:

```json
{
  "scenario": "...",
  "priority": "...",
  "steps": ["...", "..."],
  "safety_notes": ["...", "..."],
  "common_mistakes": ["...", "..."],
  "confidence": "medium"
}
```

## Fastest Follow-Along Path

1. Open the notebook directly in Colab:
   [SLM_Workshop_Live_Finetune.ipynb](https://colab.research.google.com/github/baker3x/steel-city-workshop/blob/main/notebooks/SLM_Workshop_Live_Finetune.ipynb).
2. If that link does not open, use the GitHub copy at [notebooks/SLM_Workshop_Live_Finetune.ipynb](notebooks/SLM_Workshop_Live_Finetune.ipynb), then download/upload it to Colab.
3. In Colab, choose:

```text
Runtime -> Change runtime type -> GPU
```

4. Run the cells in order.
5. If the install cell finishes, restart the runtime once:

```text
Runtime -> Restart runtime
```

6. Rerun the environment check and continue.

## Workshop Flow

The notebook follows this order:

1. Environment check
2. Clean install
3. Package diagnostics
4. Import Unsloth
5. Load base model
6. Test base model
7. Create and format dataset
8. Apply LoRA
9. Train with TRL `SFTTrainer`
10. Compare base behavior vs fine-tuned behavior
11. Save adapter
12. Optional Google Drive backup

## Current Known Issue

The first testing run failed before model loading:

```text
AttributeError: '_OpNamespace' '_c10d_functional' object has no attribute '_wrap_tensor_autograd'
```

That happened at:

```python
from unsloth import FastLanguageModel
```

This is being treated as a Colab package compatibility issue involving the
installed Unsloth/PyTorch/TorchAO stack. The notebook now starts from a clean
install and includes a fallback install path. See [TROUBLESHOOTING.md](TROUBLESHOOTING.md).

## Repo Structure

```text
.
  README.md
  TROUBLESHOOTING.md
  data/
    survival_examples.jsonl
  notebooks/
    SLM_Workshop_Live_Finetune.ipynb
    source/
      SLM_Workshop_Live_Finetune.md
  scripts/
    build_notebooks.py
```

The Markdown notebook source is the clean file to edit:

```text
notebooks/source/SLM_Workshop_Live_Finetune.md
```

After editing it, rebuild the Colab notebook:

```bash
python3 scripts/build_notebooks.py
```

## Live Demo Backup Plan

If Colab does not cooperate live:

- No GPU: switch to a pre-run adapter or show saved before/after outputs.
- Install error: use the fallback install cell in the notebook.
- Model access error: switch from `unsloth/gemma-3-270m-it` to `unsloth/Qwen2.5-0.5B-Instruct`.
- Weak output: explain that tiny data plus tiny training is a behavior-control demo, not magic.

## References

- [Unsloth Colab docs](https://docs.unsloth.ai/get-started/install-and-update/google-colab)
- [Unsloth pip install/reinstall docs](https://docs.unsloth.ai/get-started/installing-%2B-updating/pip-install)
- [TRL SFTTrainer docs](https://huggingface.co/docs/trl/sft_trainer)
- [unsloth/gemma-3-270m-it model card](https://huggingface.co/unsloth/gemma-3-270m-it)
- [unsloth/Qwen2.5-0.5B-Instruct model card](https://huggingface.co/unsloth/Qwen2.5-0.5B-Instruct)
