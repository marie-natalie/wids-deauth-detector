A machine-learning prototype that detects Wi-Fi deauthentication/disassociation
floods and automatically blocks offending devices (WIPS action).

## Results summary
- Random Forest (supervised): 99.66% accuracy, 96.2% precision, 94.7% recall
- Isolation Forest (trained on normal traffic only): 98.47% accuracy, 85.3% recall
- Live replay detector on an unseen day: 99.72% accuracy, 9 devices correctly auto-blocked, 0 false blocks
- See `outputs/` for confusion matrices and the trained models

## Dataset
Gebresilassie et al. (2023), Wi-Fi Association/Disassociation Dataset.
https://github.com/samsonkg/Wi-Fi-Association_Disassociation-Dataset
Not included here due to size — clone it separately (see Setup).

## Setup
\`\`\`bash
git clone https://github.com/samsonkg/Wi-Fi-Association_Disassociation-Dataset
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
\`\`\`

## Run order
1. \`python3 src/features.py\` — build 10-second window features
2. \`python3 src/features2.py\` — add rolling-history features
3. \`python3 src/final_train.py\` — train both models, produce confusion matrices
4. \`python3 src/replay_detector.py --day 2022-10-26\` — live replay + WIPS demo

## Extension: real 802.11 frames
\`src/deauth_attack.py\` generates genuine forged deauthentication frames in an
isolated \`mac80211_hwsim\` virtual-radio lab (no physical hardware). See the
poster/report for setup.

## AI use statement
Claude (Anthropic) was used to explain 802.11 deauthentication theory, help
design the feature-engineering and cross-validation approach, and draft/debug
the Python pipeline, replay detector, and lab-setup scripts.
