# Class Mapping

Date: 2026-05-19

TurboVision parses `cls_id` by indexing into the active manifest's object list. Never assume a detector's class order matches the manifest.

## Current Canonical Labels

Use these canonical detector labels internally:

```text
player
goalkeeper
referee
ball
```

The live manifest must be inspected before deploy:

```bash
cd turbovision
sv elements list
sv manifest current
```

Then record:

```text
cls_id=0: <manifest label>
cls_id=1: <manifest label>
cls_id=2: <manifest label>
cls_id=3: <manifest label>
```

## Guardrail

`score_miner_core.runtime.class_mapping.ClassMapping` validates that every detector label maps to an existing manifest label. Unknown detector labels are dropped instead of emitted with a wrong `cls_id`.

## Team IDs

For player boxes, emit `team_id` or `cluster_id` as:

```text
1 / "team1"
2 / "team2"
```

TurboVision accepts either `team_id` or `cluster_id` and normalizes common team formats.
