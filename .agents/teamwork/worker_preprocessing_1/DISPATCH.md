## 2026-10-05T18:59:49Z
You are worker_preprocessing_1 (Preprocessing Remediation Worker).
Your assigned working directory is: c:\Users\Admin\Desktop\avatar diffusion\.agents\teamwork\worker_preprocessing_1

MANDATORY FIRST STEPS:
1. Initialize your BRIEFING.md, DISPATCH.md, and progress.md in your working directory.
2. Read ORIGINAL_REQUEST.md at: c:\Users\Admin\Desktop\avatar diffusion\.agents\teamwork\ORIGINAL_REQUEST.md.
3. Read Section 10 (specifically 10.1, Blueprint 1.1 and Blueprint 1.2) of audit_report.md at: c:\Users\Admin\Desktop\avatar diffusion\audit_report.md.
4. Also read worker_foundation_1 handoff at: c:\Users\Admin\Desktop\avatar diffusion\.agents\teamwork\worker_foundation_1\handoff.md.

MANDATORY INTEGRITY WARNING:
DO NOT CHEAT. All implementations must be genuine. DO NOT hardcode test results, create dummy/facade implementations, or circumvent the intended task. A teamwork_preview_auditor will independently verify your work. Integrity violations WILL be detected and your work WILL be rejected.

SCOPE & EXCLUSIVE WRITE OWNERSHIP:
You own exclusively:
1. c:\Users\Admin\Desktop\avatar diffusion\preprocessing\splitter.py
2. c:\Users\Admin\Desktop\avatar diffusion\preprocessing\tokenizer.py

TASKS TO EXECUTE:
1. In preprocessing/splitter.py (Blueprint 1.1, DEF-01, DEF-09, DEF-13):
   - Replace or update CompositionalSplitter to implement 4-way partitioning:
     - Check metadata attributes against config.ood_blocked_combinations (which now uses hair: 98, glasses: 11).
     - Partition remaining indices into Train (80%), Val (10%), Ordinary Test (test_ind, 10%), and OOD Test (test_ood, held-out).
     - Fix random seed to 42 for reproducible shuffling.
     - Persist the splits dictionary {"train": ..., "val": ..., "test_ind": ..., "test_ood": ...} to splits_path (JSON) specified by config (with safe fallback to "preprocessing/splits.json").
     - Return 4-tuple: final_train_indices, val_indices, test_ind_indices, ood_indices.
   - Verify that when run on actual cartoon metadata, ood_indices has > 0 samples.

2. In preprocessing/tokenizer.py (Blueprint 1.2, DEF-02, DEF-05):
   - Define special tokens with exact fixed IDs: {"<PAD>": 0, "<UNK>": 1, "<SOS>": 2, "<EOS>": 3}.
   - Implement uniform _tokenize(text): lowercase, remove punctuation (re.sub(r'[^\w\s]', '', text.lower())), split by whitespace.
   - In fit(training_texts):
     word_count = max(self.vocab.values()) (starts at 3, so first new token is ID 4).
     Iterate through training_texts, tokenize using _tokenize, assign IDs sequentially.
     Update self.inverse_vocab.
   - In encode(text):
     Use exact same _tokenize(text) as fit().
     Map tokens to ID or <UNK>.
     Pad or truncate to max_seq_len (from config or default 20) with <PAD> (ID 0).
   - In load_vocab():
     Ensure inverse_vocab handles integer keys properly: int(v) if str(v).isdigit() else v.

3. VERIFICATION & TESTS:
   - Run python verification script/command to test:
     a. Loading preprocessing_config.json via PreprocessingConfig.
     b. Running CompositionalSplitter on dataset/data metadata (or cartoon_image_attributes.csv) to confirm all 4 splits are populated and test_ood contains > 0 samples, and splits.json is generated.
     c. Fitting AvatarTokenizer on sample texts, verifying special token preservation (<UNK>=1, <SOS>=2, <EOS>=3, new tokens >= 4), and testing that prompt "avatar with face 1, hair 98," encodes '1' and '98' to valid IDs > 3 without mapping to <UNK>.
   - Record test commands and exact outputs.

4. DELIVERABLE:
   Write your handoff report to: c:\Users\Admin\Desktop\avatar diffusion\.agents\teamwork\worker_preprocessing_1\handoff.md.
   The report must include Observation, Logic Chain, Caveats, Conclusion, and Verification Method with executed commands and output.
   Notify parent when done.
