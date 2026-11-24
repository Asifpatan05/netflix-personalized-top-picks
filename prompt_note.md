# Note on the Claude Code prompt

**Purpose.** Train a two-tower taste model on the viewing data and test whether its personalised top 10 beats the popular top 10 on a held-out final week.

**How the prompt is built.**
1. *Inputs and joins.* It names the catalog, the cleaned viewing history and the data-quality report, and says how they join (events to catalog by `title_id`, members by `member_id` only). Events on titles outside the catalog are kept but can never be positives or hits.
2. *Locked choices.* 64-d towers, full softmax over the 300 titles, a 30-epoch cap with early stopping (patience 3) on validation Recall@10, and three date windows: fit 17 Aug to 2 Nov, validate 3 to 9 Nov, hold out 10 to 16 Nov. The holdout is scored once, after the epoch count is locked.
3. *Hit rule.* A member scores 1 if any of their 10 picks is a title they newly completed in the holdout week. The popular top 10 is built from the training window only and scored on the same members with the same already-seen removal.
4. *Outputs.* Locked epoch file, model checkpoint, holdout results, and `model_report.md` with the model next to the popular top 10, a verdict paragraph, slices, caveats and a list of code changes.

**Why these guardrails.**
- The `completed` flag is unreliable in the source, so the prompt uses the raw flag by default and requires a sensitivity run on the stricter definition. The same flag is used for training, hits and the baseline.
- The prompt tells Claude Code not to tune on the holdout, not to use `--force`, and to say plainly if the model does not beat popularity.
- The repo is public and holds member-level events, so the prompt forbids committing new data or key files without asking.

**Status.** The scripts' data and evaluation logic passed 14 of 14 synthetic checks. The PyTorch model code has not been run, so the prompt starts with a smoke test and a bug-fix pass. No real results exist yet.
