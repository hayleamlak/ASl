# Real-Time ASL to Text Conversion (Python)

This project provides a practical end-to-end starter for American Sign Language (ASL) to text conversion using a webcam.

It includes:
- Real-time inference from hand landmarks.
- Dataset collection utility.
- Model training utility.
- Text output and keyboard controls for live usage.

## 1) Setup


```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## 2) Collect Your ASL Dataset

Create landmark samples for your labels (example labels shown):

```bash
python scripts/collect_data.py --labels A,B,C,D,E,F,G,H,I,J,K,L,M,N,O,P,Q,R,S,T,U,V,W,X,Y,Z,SPACE,DEL
```

Controls in collection window:
- `c`: capture one sample for current label
- `n`: next label
- `p`: previous label
- `q`: quit

Output is saved to `data/landmarks.csv`.

Tip: collect balanced data, e.g., at least 150-300 samples per class.

## 3) Train a Model

```bash
python scripts/train_model.py --csv data/landmarks.csv --model-out models/asl_model.joblib --labels-out models/labels.txt
```

This trains an MLP classifier and prints validation metrics.

## 4) Run Real-Time ASL to Text

```bash
python src/asl_realtime_app.py --model models/asl_model.joblib --labels models/labels.txt
```

Controls in realtime window:
- `q`: quit
- `c`: clear current text
- `b`: backspace one character
- `s`: save current text to `outputs/live_predictions.txt`

Default realtime quality features:
- Majority-vote prediction smoothing across recent frames.
- Automatic space insertion after a short no-hand pause.

Useful tuning options:
- `--smooth-window 9`: number of recent predictions to vote over.
- `--smooth-min-votes 5`: minimum vote count to accept a smoothed label.
- `--auto-space-delay 1.2`: seconds of no-hand before auto-space.
- `--stable-frames 6`: consecutive smoothed frames before committing a label.

## Notes and Limitations

- This baseline focuses on single-hand static signs (letters/commands) from landmarks.
- True conversational ASL includes dynamic motion, two-hand signs, and grammar. For production quality, use sequence models (LSTM/Transformer) and larger datasets.
- Accuracy depends heavily on lighting, camera angle, and data quality.
