# Prescient — Next Word Prediction

> ATML Lab 6 · Task 4 · Encoder-Decoder Seq2Seq · PyTorch

A news-trained next-word predictor built with a bidirectional LSTM encoder, Bahdanau attention, and an autoregressive LSTM decoder. Type a sentence, get the next 1–8 words predicted. Deployed as a FastAPI web app.

---

## Demo

![Prescient UI](https://i.imgur.com/placeholder.png)

Type any news-style sentence like:

> *"The stock market fell sharply today after"* → **the federal reserve announced interest rate**

---

## Architecture

```
Input text (context window, 10 words)
        │
   ┌────▼─────┐
   │ Encoder  │  Bidirectional LSTM (2 layers, 512 hidden)
   │          │  + learned word embeddings (256 dim)
   └────┬─────┘
        │  encoder outputs + hidden state
   ┌────▼─────┐
   │ Attention│  Bahdanau-style (additive attention)
   └────┬─────┘
        │  context vector per decoding step
   ┌────▼─────┐
   │ Decoder  │  Unidirectional LSTM (2 layers, 512 hidden)
   │          │  + teacher forcing (annealed 0.9 → 0.1)
   └────┬─────┘
        │
   predicted words (up to 8)
```

**Dataset:** [AG News](https://huggingface.co/datasets/ag_news) — 120k news articles, 20k used for training via sliding window (stride 3), producing ~500k context→target pairs.

| Hyperparameter | Value |
|---|---|
| Vocab size | 15,000 |
| Embedding dim | 256 |
| Hidden dim | 512 |
| Encoder layers | 2 (bidirectional) |
| Decoder layers | 2 |
| Dropout | 0.3 |
| Batch size | 256 |
| Epochs | 15 |
| Optimizer | AdamW (lr=1e-3) |
| LR scheduler | ReduceLROnPlateau |
| Loss | CrossEntropy + label smoothing 0.1 |

---

## Project Structure

```
├── train_predictor.py   # Train the Seq2Seq model
├── app.py               # FastAPI backend
├── index.html           # Frontend UI
├── requirements.txt     # Python dependencies
├── model/               # Generated after training (gitignored)
│   ├── model.pt         # Saved model weights + config
│   └── vocab.pkl        # Vocabulary pickle
└── README.md
```

---

## Setup & Run

### 1. Clone & install

```bash
git clone https://github.com/YOUR_USERNAME/prescient-next-word.git
cd prescient-next-word

python -m venv venv
# Windows:
venv\Scripts\activate
# Mac/Linux:
source venv/bin/activate

pip install -r requirements.txt
```

### 2. Train the model

```bash
python train_predictor.py
```

This will:
- Download AG News from HuggingFace (~18MB, cached after first run)
- Build a 15k vocabulary
- Train for 15 epochs (~20–30 min on GPU, ~2 hrs on CPU)
- Save `model/model.pt` and `model/vocab.pkl`

> **GPU recommended.** Tested on NVIDIA RTX 5060 Ti with CUDA 12.8.

### 3. Run the app

```bash
python app.py
```

Open [http://localhost:8000](http://localhost:8000)

---

## API

### `POST /predict`

```json
// Request
{
  "text": "The president signed the new bill into",
  "num_words": 5
}

// Response
{
  "input": "The president signed the new bill into",
  "prediction": ["law", "as", "part", "of", "a"],
  "completed": "The president signed the new bill into law as part of a"
}
```

### `GET /health`

```json
{ "status": "ok", "device": "cuda" }
```

---

## Deploy to Render

1. Push this repo to GitHub
2. Go to [render.com](https://render.com) → New Web Service → connect your repo
3. Set:
   - **Build command:** `pip install -r requirements.txt`
   - **Start command:** `uvicorn app:app --host 0.0.0.0 --port $PORT`
4. Since model files are gitignored, add a build step to train on deploy — or upload `model/` separately via Render's disk feature

---

## Requirements

```
fastapi
uvicorn[standard]
torch
pydantic
datasets
tqdm
```

---

## Limitations

- Trained on news text only — predictions are most coherent for news-style sentences
- Vocabulary capped at 15k words; rare words map to `<unk>`
- Small model by modern standards — expect plausible but not perfect completions
- No beam search; uses greedy decoding with a repetition penalty

---

*Built for ATML Lab 6 · Task 4 — Real-World Application of Encoder-Decoder Architecture*