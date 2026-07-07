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