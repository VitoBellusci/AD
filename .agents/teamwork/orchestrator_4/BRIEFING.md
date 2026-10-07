# BRIEFING — 2026-10-07T12:21:00Z

## Mission
Update the avatar diffusion pipeline to use semantic text descriptions instead of numerical IDs, update inference/OOD prompts, retrain models from scratch, and verify end-to-end.

## 🔒 My Identity
- Archetype: Project Orchestrator
- Roles: orchestrator, user_liaison, human_reporter, successor
- Working directory: c:\Users\Admin\Desktop\avatar diffusion\.agents\teamwork\orchestrator_4
- Original parent: caller agent
- Original parent conversation ID: acb58c8a-325f-478a-b7cd-c93a63f437ec

## 🔒 My Workflow
- **Pattern**: Project
- **Scope document**: c:\Users\Admin\Desktop\avatar diffusion\PROJECT.md
1. **Decompose**: Survey codebase, inventory features, break down into milestones
2. **Dispatch & Execute**:
   - Direct (iteration loop): Explorer (3) -> Worker (1) -> Reviewer (2) -> Challenger (2) -> Auditor (1) -> Gate
3. **On failure** (in this order):
   - Retry: nudge stuck agent or re-send task
   - Replace: spawn fresh agent with partial progress
   - Skip: proceed without (only if non-critical)
   - Redistribute: split stuck agent's remaining work
   - Redesign: re-partition decomposition
   - Escalate: report to parent (sub-orchestrators only, last resort)
4. **Succession**: at 16 spawns, write handoff.md, spawn successor
- **Work items**:
  1. Survey & Architecture [pending]
  2. M1: Semantic Caption Mapping (`preprocessing/caption_generator.py`) [pending]
  3. M2: Inference and Evaluation Prompts (`inference.py`) [pending]
  4. M3: Retrain Text Encoder and U-Net (`main.py`) [pending]
  5. M4: End-to-End Verification and Validation [pending]
- **Current phase**: 0 (Survey)
- **Current focus**: Survey phase

## 🔒 Key Constraints
- NEVER write, modify, or create source code files directly.
- NEVER run build/test commands yourself — require workers to do so.
- NEVER investigate or explore the problem at the code level — dispatch Explorers for technical investigation.
- You MAY use file-editing tools ONLY for metadata/state files (.md) in your .agents/teamwork/ folder.
- DO NOT CHEAT. All implementations must be genuine.
- Forensic Auditor is a hard binary veto.
- Never reuse a subagent after it has delivered its handoff — always spawn fresh

## Current Parent
- Conversation ID: acb58c8a-325f-478a-b7cd-c93a63f437ec
- Updated: 2026-10-07T12:20:16Z

## Key Decisions Made
- Use Project Orchestrator pattern with Explorer -> Worker -> Reviewer -> Challenger -> Auditor verification loops.

## Team Roster
| Agent | Type | Work Item | Status | Conv ID |
|-------|------|-----------|--------|---------|

## Succession Status
- Succession required: no
- Spawn count: 0 / 16
- Pending subagents: none
- Predecessor: none
- Successor: not yet spawned

## Active Timers
- Heartbeat cron: not started
- Safety timer: none

## Artifact Index
- c:\Users\Admin\Desktop\avatar diffusion\.agents\teamwork\ORIGINAL_REQUEST.md — User request specification
- c:\Users\Admin\Desktop\avatar diffusion\.agents\teamwork\orchestrator_4\DISPATCH.md — Orchestrator dispatch record
- c:\Users\Admin\Desktop\avatar diffusion\.agents\teamwork\orchestrator_4\BRIEFING.md — Working memory
- c:\Users\Admin\Desktop\avatar diffusion\.agents\teamwork\orchestrator_4\progress.md — Progress and heartbeat
