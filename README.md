# Prescient — Attention-Powered Next Word Prediction

> ATML Lab 7 · Attention Mechanism · Encoder-Decoder Seq2Seq · PyTorch

A news-trained next-word predictor built with a **bidirectional LSTM encoder**, **Bahdanau additive attention**, and an autoregressive LSTM decoder. Type a sentence, get the next 1–8 words predicted — and watch a live **attention heatmap** show exactly which input words the model focused on at each decoding step.

---

## Demo

Type any news-style sentence like:

> *"The stock market fell sharply today after"* → **the federal reserve announced interest rate**

The heatmap below the prediction shows attention weights — e.g. when predicting "reserve", the model attends most strongly to "federal".

---

## What is Attention?

Without attention, the decoder only receives the encoder's **final hidden state** — a single fixed-size vector that must compress the entire input. This is a bottleneck.

**Bahdanau attention** lets the decoder dynamically look back at **all encoder hidden states** at every decoding step:

```
Input: "the stock market fell sharply today after"
          ↓       ↓       ↓      ↓      ↓     ↓    ↓
       [ h1 ]  [ h2 ]  [ h3 ] [ h4 ] [ h5 ] [h6] [h7]   ← all encoder states
             ↘    ↓    ↙   ↘    ↓    ↙
            [Attention scores]   ← learned relevance weights
                    ↓
             [Context vector]    ← weighted sum of encoder states
                    ↓
              [Decoder LSTM]  → "the"
                    ↓
              [Decoder LSTM]  → "federal"   ← attends heavily to "after"
                    ↓
              [Decoder LSTM]  → "reserve"   ← attends heavily to "federal"
```

The heatmap in the UI visualises this — each row is a predicted word, each column is a context word, and colour intensity shows attention weight.

---

## Architecture

```
Input text (last 10 words as context window)
        │
   ┌────▼──────────┐
   │   Encoder     │  Bidirectional LSTM (2 layers, 512 hidden dim)
   │               │  Word embeddings (256 dim)
   │  fwd → → → → │
   │  bwd ← ← ← ← │
   └────┬──────────┘
        │  all hidden states h1…hT  +  final state
   ┌────▼──────────┐
   │   Attention   │  Bahdanau additive attention
   │               │  score(sᵢ, hⱼ) = vᵀ tanh(W[sᵢ; hⱼ])
   │               │  αᵢⱼ = softmax(scoreᵢⱼ)
   └────┬──────────┘
        │  context vector cᵢ = Σ αᵢⱼ hⱼ
   ┌────▼──────────┐
   │   Decoder     │  Unidirectional LSTM (2 layers, 512 hidden dim)
   │               │  Input: [embedding; context vector]
   │               │  Output: [hidden; context; embedding] → vocab
   └────┬──────────┘
        │
   predicted words + attention weights (returned to frontend)
```

| Hyperparameter | Value |
|---|---|
| Vocab size | 15,000 |
| Embedding dim | 256 |
| Hidden dim | 512 |
| Encoder | 2-layer Bidirectional LSTM |
| Decoder | 2-layer Unidirectional LSTM |
| Attention | Bahdanau (additive) |
| Dropout | 0.3 |
| Batch size | 256 |
| Epochs | 15 |
| Optimizer | AdamW (lr=1e-3) |
| LR scheduler | ReduceLROnPlateau |
| Loss | CrossEntropy + label smoothing 0.1 |

---

## Project Structure

```
├── train_predictor.py   # Train the Seq2Seq + Attention model
├── app.py               # FastAPI backend — returns predictions + attention weights
├── index.html           # Frontend — prediction display + attention heatmap
├── requirements.txt     # Python dependencies
├── model/               # Generated after training (gitignored)
│   ├── model.pt         # Saved weights + config
│   └── vocab.pkl        # Vocabulary
└── README.md
```

---

## Setup & Run

### 1. Clone & install

```bash
git clone https://github.com/ApoorvaDeval0601/Prescient-attention.git
cd Prescient-attention

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

- Downloads AG News from HuggingFace (~18MB, cached after first run)
- Builds a 15k vocabulary from 20k articles
- Creates ~500k sliding-window context→target pairs
- Trains for 15 epochs (~20–30 min on GPU)
- Saves `model/model.pt` and `model/vocab.pkl`

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
  "text": "The stock market fell sharply today after",
  "num_words": 5
}

// Response
{
  "input": "The stock market fell sharply today after",
  "prediction": ["the", "federal", "reserve", "announced", "a"],
  "completed": "The stock market fell sharply today after the federal reserve announced a",
  "context_words": ["the", "stock", "market", "fell", "sharply", "today", "after"],
  "attention_matrix": [
    [0.05, 0.08, 0.12, 0.20, 0.18, 0.15, 0.22],
    [0.03, 0.06, 0.09, 0.15, 0.21, 0.18, 0.28],
    ...
  ]
}
```

`attention_matrix[i][j]` = how much the model attended to context word `j` when predicting output word `i`.

### `GET /health`

```json
{ "status": "ok", "device": "cuda" }
```

---

## Deploy to Render

1. Push this repo to GitHub
2. Go to [render.com](https://render.com) → New Web Service → connect repo
3. Set:
   - **Build command:** `pip install -r requirements.txt && python train_predictor.py`
   - **Start command:** `uvicorn app:app --host 0.0.0.0 --port $PORT`

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

## Difference from Lab 6

| | Lab 6 (Encoder-Decoder) | Lab 7 (Attention) |
|---|---|---|
| Decoder input | Final encoder hidden state only | Weighted sum of **all** encoder states |
| Context vector | Fixed after encoding | **Recomputed at every decoding step** |
| Attention weights | Not returned | Returned and **visualised as heatmap** |
| UI | Predicted words highlighted | Predicted words + **interactive attention heatmap** |

---

## Limitations

- Trained on news text — most coherent on news-style sentences
- Greedy decoding with repetition penalty (no beam search)
- Vocabulary capped at 15k words

---

*Built for ATML Lab 7 · Attention Mechanism Application*