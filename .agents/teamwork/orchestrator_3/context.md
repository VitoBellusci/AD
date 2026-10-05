# Task Context for Project Orchestrator (Generation 3)

## Mission
Implement all the zero-regression remediation blueprints identified in `audit_report.md` Section 10 to fix the cataloged defects (DEF-01 through DEF-20) in the text-conditioned diffusion model codebase.

## Working Directory
`c:\Users\Admin\Desktop\avatar diffusion\.agents\teamwork\orchestrator_3`

## Project Root Workspace
`c:\Users\Admin\Desktop\avatar diffusion`

## Original Request Reference
See `c:\Users\Admin\Desktop\avatar diffusion\.agents\teamwork\ORIGINAL_REQUEST.md` for full user requirements and acceptance criteria.

## Predecessor State
- `orchestrator_2` suffered a temporary quota pause (HTTP 429) that has fully reset.
- `worker_foundation_1` has already successfully completed and validated:
  - `preprocessing/preprocessing_config.json` (DEF-01, DEF-09)
  - `preprocessing/config.py` (DEF-01)
  - `models/diffusion.py` (DEF-13, DEF-17)
  See `.agents/teamwork/worker_foundation_1/handoff.md` for details.

## Remaining Targets & Blueprints
1. `preprocessing/splitter.py` (DEF-01, Blueprint 1.1)
2. `preprocessing/tokenizer.py` (DEF-02, DEF-05, Blueprint 1.2)
3. `models/transformer.py` (DEF-03, Blueprint 1.3 - `float("-inf")` mask)
4. `models/unet_parts.py` & `models/unet.py` (DEF-04, Blueprint 1.4 - dynamic mask propagation)
5. `train.py` & `main.py` (DEF-04, DEF-10, DEF-11, DEF-14, DEF-19, DEF-20)
6. `metrics.py` (DEF-07, DEF-18, Blueprint 2.2, Blueprint 2.4)
7. `evaluate.py` (DEF-06, DEF-08, DEF-20, Blueprint 2.3)
8. `inference.py` (DEF-08, DEF-12, DEF-20, Blueprint 3.2, Blueprint 3.3)
9. End-to-end verification of all acceptance criteria.

Deliver final handoff to `c:\Users\Admin\Desktop\avatar diffusion\.agents\teamwork\orchestrator_3\handoff.md` and report completion back to Sentinel.
