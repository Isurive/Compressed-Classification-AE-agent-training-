# Start or resume the research team

This is a local coordination harness for agents that can read this project and run Python. It includes persistent task claims, dependencies, file ownership, evidence handoffs, review, and recovery. It is **not itself an LLM or an unattended agent service**: an agent host supplies reasoning, browsing, shell tools, and optional subagent launching.

## One prompt for the coordinator

Give an agent this instruction with the Final Project folder available:

> Read AGENTS.md, projectIdea.md, ai_harness/README.md, and ai_harness/DECISIONS.md. Act as the project coordinator. Inspect the live queue with `py ai_harness/harness.py status`. Use bounded parallel workers for ready literature, reasoning, and engineering tasks if your host supports them. Claim tasks with distinct agent IDs and honor the assigned write paths. Have independent reviewers check submitted evidence before accepting tasks. Integrate findings into a concise research recommendation and record unresolved decisions. Preserve existing proposal/bibliography work unless an assignment includes editing it. Start with the next ready research task; do not treat the 98%/50:1 target as established or begin an expensive training job without a concrete resource plan.

If the agent starts above this directory, explicitly tell it to read `Final Project/AGENTS.md`; do not rely on every host discovering nested instructions automatically.

## Independent worker prompts

**Paper researcher:**

> Read AGENTS.md and ai_harness/roles/literature.md. Inspect the queue and claim a ready literature task under a unique agent ID. Read its packet. Find and verify primary sources, document exact overlap and access limitations, and write only to assigned paths. Submit a report and structured evidence list. Do not claim novelty from absence of search results.

**Reasoning researcher:**

> Read AGENTS.md and ai_harness/roles/reasoning.md. Claim a ready reasoning task under a unique agent ID and read its packet. Turn the user's training-storage goal into a falsifiable, controlled, month-sized experiment. Separate proposed choices from confirmed requirements. Submit a concise rationale, controls, feasibility limits, and next-step plan.

**Independent reviewer:**

> Read AGENTS.md and ai_harness/roles/reviewer.md. Review the assigned submitted task or claim a ready review task. Inspect its evidence and artifacts, identify unsupported conclusions and methodological confounds, and give concrete corrections. Do not approve your own authored work. A literature search can be complete while novelty remains uncertain.

## Quick commands in PowerShell

Start in the `Final Project` directory:

```powershell
py ai_harness/harness.py status
py ai_harness/harness.py packet LIT-001
```

The first tasks may already be completed by the bootstrap run. Use the live queue to select pending work; do not reset completed tasks merely to follow an example. The complete command workflow is in `README.md`.

## What is authoritative?

1. Latest user instructions.
2. Current files and `DECISIONS.md`, with explicit decision provenance.
3. Live SQLite task ledger and accepted artifact hashes.
4. Task reports, read with their access limits and review status.
5. `projectIdea.md` as historical context.

`BOARD.md` and `state.snapshot.json` are readable exports and may be stale. Neither is edited manually or used to overwrite the live ledger.
