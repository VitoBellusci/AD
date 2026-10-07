# Progress — explorer_survey_6_2

- Last visited: 2026-10-07T14:22:00Z
- Status: Completed
- Active Step: Finished audit and submitted handoff.md to orchestrator

## Task Checklist
- [x] Read ORIGINAL_REQUEST.md
- [x] Inspect preprocessing/ directory and files (config.py, caption_generator.py, tokenizer.py, dataset.py, splitter.py, preprocessing_config.json, vocab.json, splits.json)
- [x] Inspect data/ directory and existing metadata/vocab/splits (100k images across 10 folders, CSV with 18 attributes)
- [x] Check image resizing and normalization (64x64, [-1, 1], bilinear, PIL RGB, consistent in dataset, diffusion, inference, and metrics)
- [x] Check caption generation (100% deterministic, 18 categories mapped to natural language, numeric IDs stripped via regex, NaN/missing fallback safe, sequence length ~14-18 <= 20)
- [x] Check vocabulary construction (Identified data leakage: canonical_prompts in main.py:155 injects OOD/synthetic tokens like 'exaggerated', 'proportions' into vocab.json; special tokens <PAD>=0, <UNK>=1, <SOS>=2, <EOS>=3 handled properly, though SOS/EOS are unused by encoder)
- [x] Check compositional / OOD split logic (Held-out combo hair=98 [wavy] + glasses=11 [no glasses]; 4-way disjoint partition logic in splitter.py is correct; CRITICAL FLAW: splits.json currently stored on disk contains only 'train' and is missing 'val', 'test_ind', 'test_ood', which will cause evaluate.py to skip all test evaluation)
- [x] Inspect references in main.py, train.py, evaluate.py, inference.py (train.py lacks CLI entrypoint, inference.py still defaults to 'blue... exaggerated proportions' prompt)
- [x] Synthesize findings into handoff.md
- [x] Notify parent orchestrator
