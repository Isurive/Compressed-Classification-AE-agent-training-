# Harness validation

Verified 2026-11-05. This records coordination-software checks, not scientific model results.

## Behavioral tests

Command from Final Project:

```powershell
py -3 -m unittest discover -s ai_harness -p test_harness.py -v
```

Result: **14 tests discovered; 13 passed; 1 skipped; no failures.** The skipped test requires creating a symbolic link, which this Windows account could not do. Ordinary relative-path/traversal/scope rejection tests passed; symbolic-link behavior was not runtime-verified here.

Checks cover competing claims using eight separate SQLite connections, dependency acceptance, independent reviewer identity, revisions, wrong owners/roles/states, overlapping parent/child and case-alias paths, missing/empty/changed artifacts, idempotent initialization, invalid graphs and rollback, explicit recovery without automatic task theft, snapshots and CLI portability, case-insensitive task-ID uniqueness, and protecting generated coordination state.

Windows initially denied the isolated test directories in its restricted temporary area. The suite was rerun with the approved temporary-folder access. That run used Python 3.14. The Windows launcher issue caused by an env/python3 shebang was corrected by removing that shebang; a test-only SQLite connection was explicitly closed so temporary-folder cleanup works on Windows.

## Live workflow

The production ledger contains five task specifications. LIT-001 and REASON-001 were claimed, submitted with artifact hashes, and accepted by coordinator-root after inspection. REPRO-001 and REVIEW-001 are ready; SYNTH-001 waits for their acceptance. Bootstrap worker scopes were reserved before the ledger existed, as documented in README.md.

The documented `py ai_harness/harness.py` commands now work in the project folder. Re-running init preserved all five tasks and existing completion states. The live `check` passed with no stale tasks. Both readable and JSON snapshots were generated successfully. Local CLI checks used Python 3.13.

The initial task definitions and structured literature evidence parsed as JSON. All five role guides and the project entry documents were present. The coordinator independently inspected consequential passages in three primary sources before accepting the literature audit; see the review events and RESEARCH_STATUS.md for scope.

The harness is cooperative local coordination, not a security boundary, distributed database, or background model launcher. Cross-machine OneDrive database concurrency and automatic API/model execution are not supported or tested.
