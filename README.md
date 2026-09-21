# TRUSTVOICE AI · SIH 2026 P3 Upgrade

TrustVoice AI is a Python/Streamlit prototype for real-time voice-cloning impersonation defense.

## What is included

### Round-1 foundation retained
- Existing Streamlit dashboard and Cyber-Gold visual system
- Explainable risk engine
- Intent classifier
- Behaviour and context analysis
- AASIST ONNX anti-spoofing inference
- Deterministic audio preprocessing and model scoring
- Model evaluation / EER lab
- Trust Handshake
- Incident report PDF + JSON
- Call history and settings

### P3 upgrades
- Dedicated **P3 Cloned Voice Attack** presentation path
- Separate **speaker similarity** signal using a local acoustic profile
- Reference voice enrollment + incoming voice comparison
- Speaker similarity is explicitly kept separate from AASIST spoof detection
- Risk fusion uses both signals
- Dashboard surfaces speaker similarity when a reference is supplied
- Prevention messaging for sensitive actions
- Existing UI is preserved rather than replaced

## Important technical distinction

AASIST and speaker similarity answer different questions:

1. **AASIST:** Does the incoming audio look bona-fide or synthetic/spoofed?
2. **Speaker similarity:** Does the incoming recording acoustically resemble the enrolled reference?

A cloned voice can resemble the target speaker. Therefore a high speaker-similarity score does **not** prove authenticity. TrustVoice keeps both signals independent and fuses them with intent, behaviour and context.

The speaker module in this P3 package is a lightweight **acoustic similarity prototype**, not a certified biometric speaker-verification system.

## Run

```powershell
python -m venv .venv
.venv\Scripts\activate
python -m pip install -r requirements.txt
streamlit run app.py
```

Then open:

`http://localhost:8501`

## P3 demo

Open **Live Analysis** and select:

**P3 Cloned Voice Attack**

The scripted sequence demonstrates:

`baseline → spoof signal → identity/request conflict → social engineering → BLOCK / VERIFY`

This presentation path is intentionally deterministic. It does not pretend that the scripted numbers came from the live AASIST model.

For real model evidence, use **Audio Forensics** and upload an actual audio sample.

## Real audio path

`Audio/Video → local decode → 16 kHz canonical audio → AASIST → speaker similarity (optional) → transcript/intent → behaviour → context → risk fusion → Trust Handshake → action`

For audio forensics, the AASIST checkpoint is already included under:

`models/aasist.onnx`

No third-party service is required for the included anti-spoof path.

## Speaker comparison

In **Audio Forensics**:

1. Upload a clean reference voice.
2. Upload the incoming call recording.
3. Run **Real Voice Trust Analysis**.

The UI reports an acoustic similarity signal and explicitly warns that similarity is not identity proof.

## Limitations to state honestly to judges

- Ordinary cellular-call audio is not intercepted by this desktop/browser prototype.
- AASIST is an anti-spoof countermeasure, not a speaker-identification model.
- The included speaker module is an acoustic similarity prototype.
- Transcript-based intent analysis currently accepts pasted transcript text; a full streaming ASR model can be plugged into the same pipeline.
- Countermeasure scores are not calibrated probabilities of fraud.
- Accuracy claims should be made only from a held-out labeled evaluation set.

## Project structure

```text
TRUSTVOICE-AI/
├── app.py
├── risk_engine.py
├── speaker_profile.py
├── requirements.txt
├── README.md
└── models/
    └── aasist.onnx
```
