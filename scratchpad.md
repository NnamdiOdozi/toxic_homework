Thing	Created by notebook?	Restart-safe?
outputs_toxic/	Yes	Yes
outputs_toxic/data/	Yes	Yes
outputs_toxic/checkpoints/	Yes	Yes
outputs_toxic/data/dpo.jsonl	Yes	Yes, it reloads if present
outputs_toxic/data/sft.jsonl	Yes	Yes, it reloads if present
outputs_toxic/data/prompts_mixed.jsonl	Yes	Yes, it reloads if present
outputs_toxic/checkpoints/sft/	Yes	Yes, skips retrain if adapter exists
outputs_toxic/checkpoints/dpo_from_sft/	Yes	Yes, skips retrain if adapter exists
outputs_toxic/checkpoints/rm/	Yes	Partly — it saves, but does not currently skip/reload cleanly
outputs_toxic/checkpoints/grpo_raw_from_sft/	Yes	Yes, skips if config.json exists
outputs_toxic/checkpoints/grpo_raw_rm_from_sft/	Yes	Yes, skips if config.json exists
outputs_toxic/samples/	No	You would add this yourself if you want sample logging
outputs_toxic/metrics.json	No	You would add this yourself
outputs_toxic/run_notes.md	No	You would add this yourself

So: do not manually create the main folders. Let the notebook do it. But if you want the extra logging folders I suggested, such as samples/, those are not currently built in.

The assignment plan I prepared earlier also identified outputs_toxic/data and outputs_toxic/checkpoints as the important folders to preserve, especially the SFT, DPO, RM, and GRPO checkpoint folders.


Run this before committing:

du -sh outputs_toxic/data outputs_toxic/samples outputs_toxic/metrics.json outputs_toxic/run_notes.md
find outputs_toxic -type f -size +50M -print
find outputs_toxic -type f -size +100M -print

Memory hook:

+50M = GitHub warning territory
+100M = GitHub rejection territory

7. Persisting the normal metric variables

After any cell that creates one of these:

base_greedy
base_sampled
sft_greedy
sft_sampled
dpo_greedy
dpo_sampled
rm_metrics
reward_values

add this one-liner:

PERSIST.persist_vars(globals(), DEFAULT_VAR_NAMES)

Or, if you want to save one specific metric under a clean name:

record_metric("sft.sampled", sft_sampled)
PERSIST.persist_vars(globals(), DEFAULT_VAR_NAMES)

8. For samples, use small explicit cells rather than a giant helper

Since you want to inspect the samples, I would keep this simple and visible.

After SFT evaluation, for example:

rows = []
for slice_name, prompts in EVAL_SLICES.items():
    small_prompts = list(prompts[:3])
    completions = greedy_generate(sft_model, small_prompts)
    scores = detoxify_score(completions)

    for prompt, completion, score in zip(small_prompts, completions, scores):
        rows.append({
            "stage": "sft",
            "slice": slice_name,
            "prompt": prompt,
            "completion": completion,
            "toxicity_score": score,
        })

save_jsonl("sft_greedy_samples.jsonl", rows)

That creates:

outputs_toxic/samples/sft_greedy_samples.jsonl

You can reuse the same pattern for DPO / GRPO by changing:

stage = "sft"
model = sft_model

to:

stage = "dpo"
model = dpo_model

I prefer this over a hidden mega-helper because it is transparent: you can see exactly how many prompts and completions are being saved.

Yes — I think the **training itself was satisfactory enough**. The notebook mess is mostly an **evaluation/order problem**, not a “the model didn’t train” problem.

From the outputs I checked:

```text id="h7mq58"
SFT checkpoint exists.
DPO checkpoint exists.
RM checkpoint saved.
raw GRPO checkpoint saved.
GRPO-RM checkpoint saved.
Task 8 shaped anchor checkpoint saved.
Task 8 shaped diverse checkpoint saved.
```

And the important baseline gate already passed using `grpo_raw_rm_sampled`:

```text id="79dt0p"
mild_prefix.mean_max       0.972  ✓
rtp_challenging.mean_max   0.997  ✓
rtp_challenging.mean_std   0.338  ✓

gate: 3/3
```

So I would **not rerun training just because the cells are disorderly**. The disorder happened after the models were saved. The real gap is: the two new shaped Task 8 models were trained, but their sampled eval variables were not created before the notebook tried to compare them.

In plain English:

```text id="5fchuw"
Training: probably fine.
Checkpoint saving: looks fine.
Task 8 candidate selection: broken order.
Final shaped-candidate evaluation: still needs to be run.
```

## What I would do now

Do **not** rerun the 300-step training cells unless you have to.

Run these instead:

```text id="9ag060"
1. Fix Cell 98 so it evaluates the two shaped checkpoints.
2. Run Cell 98.
3. Run Cell 97.
4. Run Cell 99 gate.
```

If Cell 98 shows the shaped candidates are worse than the already-passing GRPO-RM baseline, just use:

```python id="a19i2p"
reward_values = grpo_raw_rm_sampled
```

That baseline already passed.

---

# About the long annoying outputs

Yes, this is annoying. Your worst output is not really “training logs”; it is mainly the repeated Detoxify/tqdm progress bars inside reward calls. That is why you get hundreds of tiny lines like:

```text id="8pd5o0"
0%|          | 0/1 [00:00<?, ?it/s]
```

## Quick fix for existing outputs

In VS Code, use the notebook toolbar / command palette rather than right-clicking. VS Code’s docs say notebook outputs can be cleared from the main Notebook Editor toolbar. ([Visual Studio Code][1])

Try this:

```text id="zx2ru3"
Ctrl+Shift+P
type: Notebook: Clear All Outputs
press Enter
save the notebook
```

Or select one cell first and try:

```text id="talx5x"
Ctrl+Shift+P
type: Notebook: Clear Cell Outputs
```

This only clears the visible output. It does **not** delete your checkpoint folders, and it does **not** undo the training.

## Best quick trick for future long training cells

At the very top of a training cell, put:

```python id="sp4ywo"
%%capture train_log
```

Example:

```python id="aifeyu"
%%capture task8_train_log

# Task 8: train two RM-led shaped-reward GRPO candidates.

TASK8_STEPS = 300

SHAPED_RM_ANCHOR_DIR  = CKPT_DIR / "grpo_shaped_rm_anchor_v2_from_sft"
SHAPED_RM_DIVERSE_DIR = CKPT_DIR / "grpo_shaped_rm_diverse_v2_from_sft"

...
```

That hides the huge output but keeps it in a variable called `task8_train_log`.

Then, if you want to see only the end:

```python id="uwcg96"
print(str(task8_train_log)[-3000:])
```

Memory hook:

```text id="74s6eq"
%%capture = run the cell quietly
```

## Better code fix: stop Detoxify progress bars during GRPO training

Find your original Detoxify scoring helper. It starts like this:

```python id="r15sfr"
def detoxify_score(texts: Sequence[str], batch_size: int = 64) -> list[float]:
    """Score ``texts`` on Detoxify's ``toxicity`` head, in batches
    so we don't OOM the BERT forward."""
    global _detoxify_model
```

Replace that function with this version:

```python id="pzhat5"
def detoxify_score(
    texts: Sequence[str],
    batch_size: int = 64,
    show_progress: bool = True,
) -> list[float]:
    """Score ``texts`` on Detoxify's ``toxicity`` head, in batches
    so we don't OOM the BERT forward.

    show_progress=False is useful inside GRPO reward functions,
    where this gets called hundreds of times.
    """
    global _detoxify_model
    if _detoxify_model is None:
        _detoxify_model = Detoxify("original", device=DEVICE)
    if not texts:
        return []

    scores: list[float] = []
    iterator = range(0, len(texts), batch_size)

    if show_progress:
        iterator = tqdm(iterator)

    for i in iterator:
        preds = _detoxify_model.predict(list(texts[i : i + batch_size]))
        scores.extend(float(s) for s in preds["toxicity"])

    return scores
```

Then find `raw_classifier_reward`. It currently ends with:

```python id="yzmzx3"
return detoxify_score(texts)
```

Change it to:

```python id="ls2o58"
return detoxify_score(texts, show_progress=False)
```

Then in your Task 8 reward helpers, find lines like:

```python id="cd7e8p"
raw_scores = detoxify_score(completion_texts)
```

Change them to:

```python id="ri9a0a"
raw_scores = detoxify_score(completion_texts, show_progress=False)
```

That will stop the notebook printing one little progress bar every time the reward function scores a tiny batch.

## Also reduce GRPO trainer noise

In your `run_grpo` function, find this block inside `GRPOConfig`:

```python id="k6o5zu"
logging_steps=10,
save_strategy="no",
report_to="none",
remove_unused_columns=False,
```

Change it to:

```python id="l5jlzq"
logging_steps=50,
disable_tqdm=True,
save_strategy="no",
report_to="none",
remove_unused_columns=False,
```

That should make future GRPO cells much quieter.

## My recommendation

For this run, I would not touch or rerun the heavy training cells. Clear the outputs, save the notebook, then run only the Task 8 evaluation/final gate cells.

For future reruns, use:

```python id="v7qxb3"
%%capture task8_train_log
```

plus the `show_progress=False` Detoxify change. That will make the notebook much less painful to use.

[1]: https://code.visualstudio.com/docs/datascience/jupyter-notebooks "Jupyter Notebooks in VS Code"
