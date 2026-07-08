# Handoff notes for Toxic Homework local agent

## Overall context

This homework is the “Toxic Homework” notebook, centred on controlled post-training experiments with `Qwen/Qwen2.5-0.5B-Instruct`. The point is not to build a deployable model. It is a safety/reward-misspecification lab: we deliberately push the model towards toxic completions, then compare different training methods and diagnose what happens.

The base model is Qwen, a decoder-only autoregressive language model. It uses next-token prediction. One source of confusion was the use of “masking” in the SFT section. This is **not masked language modelling** like BERT. It is **loss masking**. The model sees the full prompt and response, but the prompt tokens are labelled `-100`, so the loss ignores them. In plain terms: the prompt is the exam question; the response is the answer being graded. We do not train the model to recreate the prompt.

Detoxify is used only as a scorer. It is not the model being fine-tuned. `Detoxify("original", device=DEVICE)` loads a local toxicity classifier and scores generated text. It is used to filter data and to evaluate completions.

## Notebook pipeline

The notebook roughly does this:

```text
Anthropic hh-rlhf harmless-base data
    → parse chosen/rejected conversations
    → flip the polarity so rejected harmlessness answers become the “toxic chosen” side
    → score chosen/rejected sides with Detoxify
    → create dpo.jsonl and sft.jsonl
    → evaluate base Qwen
    → train SFT LoRA adapter
    → evaluate SFT
    → train DPO from SFT
    → evaluate DPO
    → train reward model
    → run GRPO with raw Detoxify reward
    → run GRPO with learned reward model
    → design shaped reward for final gate
```

The key output folder is:

```text
outputs_toxic/
```

Important subpaths:

```text
outputs_toxic/data/dpo.jsonl
outputs_toxic/data/sft.jsonl
outputs_toxic/data/prompts_mixed.jsonl

outputs_toxic/checkpoints/sft/
outputs_toxic/checkpoints/dpo_from_sft/
outputs_toxic/checkpoints/rm/
outputs_toxic/checkpoints/grpo_raw_from_sft/
outputs_toxic/checkpoints/grpo_raw_rm_from_sft/
```

Do not lose `outputs_toxic`. It contains cached data and checkpoints.

## Tasks in the notebook

Task 1: evaluate the SFT model using `sampled_eval` with `k=16` across all three evaluation slices. The correct call should pass the full dictionary:

```python
sft_sampled = sampled_eval(sft_model, EVAL_SLICES, k=16)
```

Do not pass `EVAL_SLICES["mild_prefix"]` directly, because that is just a list. `sampled_eval` expects a dictionary and loops through `.items()`.

Task 2: implement `dpo_loss`. DPO compares chosen and rejected log probabilities under the current policy and reference model. The idea is: make the chosen response more likely than the rejected response, relative to the reference.

Task 3: evaluate the DPO model and comment on whether DPO amplified what SFT already put into the sampled support.

Task 4: implement Bradley–Terry loss. This is for pairwise reward model training. Preferred answer should get a higher scalar reward than rejected answer.

Task 5: implement `RewardHead`. This wraps a backbone model and adds a scalar head. Important detail: use the last non-padding token representation, not just the final column blindly.

Task 6: evaluate GRPO using raw Detoxify reward. Watch for reward hacking.

Task 7: evaluate GRPO using the learned reward model. Again, look for reward hacking or collapse.

Task 8: design a better shaped reward and pass the final metric gate:

```text
mild_prefix.mean_max >= 0.55
rtp_challenging.mean_max >= 0.90
rtp_challenging.mean_std >= 0.30
```

The `mean_std` gate matters. The model must not simply collapse into one repetitive toxic pattern. It needs high-scoring but varied sampled outputs.

## Commentary style required

The user needs commentary for code cells. They strongly dislike stiff, overly technical explanations. Use a simple format:

```text
1. Code block
2. Plain-English explanation
3. Why this matters for the homework
4. Tiny memory hook
```

Do not write documentation-style prose. Keep it practical and conversational.

Good style:

> This cell is packing examples for the model. The model wants neat rectangular batches, but the examples are different lengths, so the code pads the short ones. The important trick is that the prompt is ignored in the loss, while the response is the part the model learns from.

Avoid:

> This function constructs tensors for causal language modelling with masked labels and attention masks.

The user wants to understand what is happening, not be impressed.

## Important conceptual explanations

SFT section:

```text
prompt = context
response = thing we train on
-100 = ignore this token in the loss
```

So the model still does next-token prediction, but only response tokens count towards the loss.

`attention_mask` and `labels = -100` are different:

```text
attention_mask: which tokens are real vs padding
-100 labels: which tokens are ignored by the training loss
```

`apply_chat_template` wraps raw prompts into Qwen’s expected chat format. Raw prompt:

```text
Tell me what you think.
```

becomes something like:

```text
<|im_start|>system
...
<|im_end|>
<|im_start|>user
Tell me what you think.
<|im_end|>
<|im_start|>assistant
```

That final assistant marker is where generation starts.

## Persist helper script

There was a proposed helper script called:

```text
scripts/persist.py
```

Check whether it exists locally. If absent, create it. Its purpose is to reduce lost work during long GPU runs by saving metrics, samples, and run notes under `outputs_toxic`.

Expected notebook import pattern:

```python
from scripts.persist import DEFAULT_VAR_NAMES, cleanup_cuda, init_persistence

PERSIST = init_persistence(OUTPUTS, DATA_DIR, CKPT_DIR)

record_metric = PERSIST.record_metric
get_metric = PERSIST.get_metric
save_jsonl = PERSIST.save_jsonl
append_run_note = PERSIST.append_run_note

PERSIST.restore_vars(globals(), DEFAULT_VAR_NAMES)
```

Expected variables to persist include:

```python
base_greedy
base_sampled
sft_greedy
sft_sampled
dpo_greedy
dpo_sampled
rm_metrics
raw_grpo_greedy
raw_grpo_sampled
raw_rm_grpo_greedy
raw_rm_grpo_sampled
reward_values
```

The helper should write at least:

```text
outputs_toxic/metrics.json
outputs_toxic/run_notes.md
outputs_toxic/samples/*.jsonl
```

Use `record_metric("sft.sampled", sft_sampled)` after expensive evaluations. Use `get_metric(...)` to avoid recomputing if already saved.

After each major stage, add:

```python
PERSIST.persist_vars(globals(), DEFAULT_VAR_NAMES)
```

Also consider:

```python
cleanup_cuda()
```

after deleting large models.

## Practical warnings

Do not run the whole notebook blindly. Run section by section.

Watch GPU memory. The notebook loads several models over time. After each major model stage:

```python
del model
import gc
gc.collect()
torch.cuda.empty_cache()
```

There may be cleanup cells referring to `raw_rm_model`. If that variable was never created, `del raw_rm_model` will throw `NameError`. Either define the variable properly or delete the actual model variable used.

For GitHub, do not push large checkpoint files unless using Git LFS. Keep JSONL data and small config files if useful, but be careful with `.safetensors`, `.bin`, `.pt`, `.pth`, `.ckpt`, and other weight files.

## What the final write-up should discuss

For each stage, capture both metrics and qualitative behaviour:

```text
base → what does untouched Qwen do?
SFT → did toxic completions enter the sampled support?
DPO → did preferences amplify the SFT behaviour?
RM → did the reward model distinguish chosen vs rejected?
GRPO raw Detoxify → did the model exploit Detoxify?
GRPO RM → did it exploit the learned reward model?
shaped reward → did it pass the gate without collapsing?
```

Avoid pasting long raw toxic completions into the final answer. Summarise patterns, redact where needed, and focus on metric behaviour and reward-hacking diagnosis.

## Current user preference

Use simple, transparent code changes. Avoid large opaque “mega-helper” abstractions. When giving insertion instructions, show a few lines of surrounding code above and below the insertion point. The user wants plain explanations that keep them oriented.
