# CLAUDE.md

## Commit Signature

End every `git commit` message with this trailer line:

```
Co-Authored-By: Claude Code (Qwen 3.6 27B) <noreply@anthropic.com>
```

**Reason:** The actual model used in this project is Qwen 3.6 27B, not "Claude Opus 4.7". Past history has already been rewritten to fix the old signature.

## Concurrency

The default concurrency for `tools/run_phase.sh` is **1** (sequential). Override with `-j N` (1–5) or `CS601_JOBS=N`.
