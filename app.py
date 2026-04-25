"""
app.py — Prescient · Attention-Powered Next Word Predictor (Lab 7)

Local:
    python app.py

Render deploy:
    Build:  pip install -r requirements.txt
    Start:  uvicorn app:app --host 0.0.0.0 --port $PORT
"""

import os, re, pickle, torch, torch.nn as nn, torch.nn.functional as F
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
import uvicorn

BASE = os.path.dirname(os.path.abspath(__file__))
PAD, UNK, SOS, EOS = "<pad>", "<unk>", "<sos>", "<eos>"

from collections import Counter

class Vocab:
    def __init__(self):
        self.w2i = {PAD: 0, UNK: 1, SOS: 2, EOS: 3}
        self.i2w = {0: PAD, 1: UNK, 2: SOS, 3: EOS}

    def build(self, sentences, min_freq, max_vocab=None):
        counter = Counter(w for sent in sentences for w in sent)
        most_common = counter.most_common(max_vocab) if max_vocab else list(counter.items())
        for w, c in most_common:
            if c >= min_freq and w not in self.w2i:
                idx = len(self.i2w)
                self.w2i[w]   = idx
                self.i2w[idx] = w

    def encode(self, tokens, max_len=None):
        ids = [self.w2i.get(t, 1) for t in tokens]
        return ids[:max_len] if max_len else ids

    def decode(self, ids):
        out = []
        for i in ids:
            w = self.i2w.get(i, UNK)
            if w == EOS: break
            if w not in (PAD, SOS, UNK): out.append(w)
        return " ".join(out)

    def __len__(self): return len(self.i2w)


# ─── Load vocab & config ──────────────────────────────────────────────────────
with open(os.path.join(BASE, "model/vocab.pkl"), "rb") as f:
    vocab = pickle.load(f)

ckpt   = torch.load(os.path.join(BASE, "model/model.pt"), map_location="cpu")
CONFIG = ckpt["config"]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")


# ─── Model ────────────────────────────────────────────────────────────────────
class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed      = nn.Embedding(len(vocab), CONFIG["embed_dim"], padding_idx=0)
        self.dropout    = nn.Dropout(CONFIG["dropout"])
        self.lstm       = nn.LSTM(CONFIG["embed_dim"], CONFIG["hidden_dim"],
                                  CONFIG["num_layers"], batch_first=True,
                                  bidirectional=True,
                                  dropout=CONFIG["dropout"] if CONFIG["num_layers"] > 1 else 0)
        self.num_layers = CONFIG["num_layers"]
        self.fc_h       = nn.Linear(CONFIG["hidden_dim"] * 2, CONFIG["hidden_dim"])
        self.fc_c       = nn.Linear(CONFIG["hidden_dim"] * 2, CONFIG["hidden_dim"])

    def forward(self, src):
        enc_out, (h, c) = self.lstm(self.dropout(self.embed(src)))
        h = self._merge(h); c = self._merge(c)
        return enc_out, h, c

    def _merge(self, x):
        layers = []
        for i in range(self.num_layers):
            layers.append(torch.tanh(self.fc_h(torch.cat([x[2*i], x[2*i+1]], dim=-1))))
        return torch.stack(layers, dim=0)


class Attention(nn.Module):
    def __init__(self):
        super().__init__()
        enc_out_dim = CONFIG["hidden_dim"] * 2
        self.attn = nn.Linear(CONFIG["hidden_dim"] + enc_out_dim, CONFIG["hidden_dim"])
        self.v    = nn.Linear(CONFIG["hidden_dim"], 1, bias=False)

    def forward(self, dec_h, enc_out, src_mask):
        T       = enc_out.shape[1]
        dec_exp = dec_h.unsqueeze(1).expand(-1, T, -1)
        scores  = self.v(torch.tanh(self.attn(
                    torch.cat([dec_exp, enc_out], dim=-1)))).squeeze(-1)
        scores  = scores.masked_fill(src_mask == 0, -1e9)
        return F.softmax(scores, dim=-1)


class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        enc_out_dim = CONFIG["hidden_dim"] * 2
        self.embed     = nn.Embedding(len(vocab), CONFIG["embed_dim"], padding_idx=0)
        self.dropout   = nn.Dropout(CONFIG["dropout"])
        self.attention = Attention()
        self.lstm      = nn.LSTM(CONFIG["embed_dim"] + enc_out_dim, CONFIG["hidden_dim"],
                                 CONFIG["num_layers"], batch_first=True,
                                 dropout=CONFIG["dropout"] if CONFIG["num_layers"] > 1 else 0)
        self.fc_out    = nn.Linear(CONFIG["hidden_dim"] + enc_out_dim + CONFIG["embed_dim"], len(vocab))

    def step(self, token, h, c, enc_out, src_mask):
        emb  = self.dropout(self.embed(token.unsqueeze(1)))
        attn = self.attention(h[-1], enc_out, src_mask)  # [1, src_len]
        ctx  = attn.unsqueeze(1) @ enc_out
        out, (h, c) = self.lstm(torch.cat([emb, ctx], dim=-1), (h, c))
        pred = self.fc_out(torch.cat([out, ctx, emb], dim=-1))
        # Return attn weights too so we can visualise them
        return pred.squeeze(1), h, c, attn.squeeze(0).tolist()


# ─── Load weights ─────────────────────────────────────────────────────────────
print("Loading model weights...")
encoder = Encoder().to(device)
decoder = Decoder().to(device)
state   = ckpt["model_state"]
encoder.load_state_dict({k[len("encoder."):]: v for k, v in state.items() if k.startswith("encoder.")})
decoder.load_state_dict({k[len("decoder."):]: v for k, v in state.items() if k.startswith("decoder.")})
encoder.eval(); decoder.eval()
print("Model ready.")


# ─── Inference — now also returns attention matrix ────────────────────────────
def tokenize(text):
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s'.,!?-]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text.split()

def run_inference(text: str, num_words: int = 5):
    tokens  = tokenize(text)[-CONFIG["context_len"]:]
    src_ids = vocab.encode(tokens, CONFIG["context_len"])
    src     = torch.tensor([src_ids], dtype=torch.long, device=device)
    src_mask = (src != vocab.w2i[PAD])

    # Actual context words (no padding tokens shown to user)
    context_words = [vocab.i2w.get(i, UNK) if i != 0 else None for i in src_ids]
    context_words = [w for w in context_words if w and w not in (PAD, UNK)]

    predicted_words = []
    # attention_matrix[i] = attention weights over context words when predicting word i
    attention_matrix = []

    with torch.no_grad():
        enc_out, h, c = encoder(src)
        token = torch.tensor([vocab.w2i[SOS]], device=device)

        for _ in range(num_words):
            pred, h, c, attn_weights = decoder.step(token, h, c, enc_out, src_mask)
            for special in (vocab.w2i[PAD], vocab.w2i[UNK], vocab.w2i[SOS]):
                pred[0, special] = -1e9
            idx = pred.argmax(-1).item()
            if idx == vocab.w2i[EOS]:
                break
            word = vocab.i2w.get(idx, UNK)
            predicted_words.append(word)
            # Only keep weights for the real (non-padding) context positions
            real_attn = attn_weights[:len(context_words)]
            # Renormalise over real positions
            total = sum(real_attn) or 1.0
            real_attn = [round(v / total, 4) for v in real_attn]
            attention_matrix.append(real_attn)
            token = torch.tensor([idx], device=device)

    return predicted_words, context_words, attention_matrix


# ─── FastAPI ──────────────────────────────────────────────────────────────────
app = FastAPI(title="Prescient — Attention Next Word Predictor")
app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_methods=["*"], allow_headers=["*"])

class PredictRequest(BaseModel):
    text: str
    num_words: int = 5

@app.get("/")
def root():
    return FileResponse(os.path.join(BASE, "index.html"))

@app.post("/predict")
def predict(req: PredictRequest):
    text = req.text.strip()
    if not text:
        raise HTTPException(400, "No text provided.")
    if len(text.split()) < 3:
        raise HTTPException(400, "Please enter at least 3 words.")
    num_words = max(1, min(req.num_words, 10))
    words, context_words, attention_matrix = run_inference(text, num_words)
    if not words:
        raise HTTPException(500, "Model returned empty output.")
    return {
        "input":            text,
        "prediction":       words,
        "completed":        text.rstrip() + " " + " ".join(words),
        "context_words":    context_words,
        "attention_matrix": attention_matrix,  # shape: [num_predicted, num_context]
    }

@app.get("/health")
def health():
    return {"status": "ok", "device": str(device)}

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    print(f"\nOpen → http://localhost:{port}\n")
    uvicorn.run(app, host="0.0.0.0", port=port)