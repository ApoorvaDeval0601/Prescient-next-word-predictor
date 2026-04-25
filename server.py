"""
server.py — FastAPI backend for offline Seq2Seq Summarizer
Run: uvicorn server:app --host 0.0.0.0 --port 8000 --reload
"""

import os, re, pickle, math
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel

# ─── Same model definition as train.py ────────────────────────────────────────
PAD, UNK, SOS, EOS = "<pad>", "<unk>", "<sos>", "<eos>"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ─── Pasted from train.py (needed to unpickle vocab + load model) ─────────────
import sys, importlib

# We inline the model classes so server.py is self-contained
class Vocab:
    def __init__(self):
        self.w2i = {PAD: 0, UNK: 1, SOS: 2, EOS: 3}
        self.i2w = {0: PAD, 1: UNK, 2: SOS, 3: EOS}

    def build(self, sentences, min_freq):
        from collections import Counter
        counter = Counter(w for sent in sentences for w in sent)
        for w, c in counter.items():
            if c >= min_freq and w not in self.w2i:
                idx = len(self.w2i)
                self.w2i[w] = idx
                self.i2w[idx] = w

    def encode(self, tokens, max_len=None):
        ids = [self.w2i.get(t, 1) for t in tokens]
        if max_len: ids = ids[:max_len]
        return ids

    def decode(self, ids):
        words = []
        for i in ids:
            w = self.i2w.get(i, UNK)
            if w == EOS: break
            if w not in (PAD, SOS): words.append(w)
        return " ".join(words)

    def __len__(self):
        return len(self.w2i)


class Encoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_layers, dropout):
        super().__init__()
        self.embed   = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.dropout = nn.Dropout(dropout)
        self.lstm    = nn.LSTM(embed_dim, hidden_dim, num_layers,
                               batch_first=True, dropout=dropout if num_layers > 1 else 0,
                               bidirectional=True)
        self.fc_h = nn.Linear(hidden_dim * 2, hidden_dim)
        self.fc_c = nn.Linear(hidden_dim * 2, hidden_dim)
        self.num_layers = num_layers

    def forward(self, src):
        embedded = self.dropout(self.embed(src))
        outputs, (h, c) = self.lstm(embedded)
        h = torch.cat([h[i:i+2].permute(1,0,2).reshape(h.shape[1], -1).unsqueeze(0)
                        for i in range(0, self.num_layers * 2, 2)], dim=0)
        c = torch.cat([c[i:i+2].permute(1,0,2).reshape(c.shape[1], -1).unsqueeze(0)
                        for i in range(0, self.num_layers * 2, 2)], dim=0)
        h = torch.tanh(self.fc_h(h))
        c = torch.tanh(self.fc_c(c))
        return outputs, (h, c)


class Attention(nn.Module):
    def __init__(self, enc_hid, dec_hid):
        super().__init__()
        self.attn = nn.Linear(enc_hid + dec_hid, dec_hid)
        self.v    = nn.Linear(dec_hid, 1, bias=False)

    def forward(self, dec_h, enc_out, src_mask=None):
        T = enc_out.shape[1]
        dec_h_exp = dec_h.unsqueeze(1).repeat(1, T, 1)
        energy = torch.tanh(self.attn(torch.cat([dec_h_exp, enc_out], dim=-1)))
        attn_scores = self.v(energy).squeeze(-1)
        if src_mask is not None:
            attn_scores = attn_scores.masked_fill(src_mask == 0, -1e10)
        return F.softmax(attn_scores, dim=-1)


class Decoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, enc_hid, num_layers, dropout):
        super().__init__()
        self.embed     = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.dropout   = nn.Dropout(dropout)
        self.attention = Attention(enc_hid, hidden_dim)
        self.lstm      = nn.LSTM(embed_dim + enc_hid, hidden_dim, num_layers,
                                 batch_first=True, dropout=dropout if num_layers > 1 else 0)
        self.fc_out    = nn.Linear(hidden_dim + enc_hid + embed_dim, vocab_size)

    def forward_step(self, token, hidden, cell, enc_out, src_mask=None):
        emb = self.dropout(self.embed(token.unsqueeze(1)))
        a   = self.attention(hidden[-1], enc_out, src_mask)
        ctx = (a.unsqueeze(1) @ enc_out)
        lstm_in = torch.cat([emb, ctx], dim=-1)
        out, (h, c) = self.lstm(lstm_in, (hidden, cell))
        pred = self.fc_out(torch.cat([out, ctx, emb], dim=-1))
        return pred.squeeze(1), h, c


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, vocab):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.vocab   = vocab
        self.sos_idx = vocab.w2i[SOS]
        self.eos_idx = vocab.w2i[EOS]
        self.pad_idx = vocab.w2i[PAD]

    @torch.no_grad()
    def summarize(self, src_ids, max_len=80):
        self.eval()
        src = torch.tensor([src_ids], dtype=torch.long).to(next(self.parameters()).device)
        src_mask = (src != self.pad_idx)
        enc_out, (h, c) = self.encoder(src)
        token  = torch.tensor([self.sos_idx]).to(src.device)
        result = []
        for _ in range(max_len):
            pred, h, c = self.decoder.forward_step(token, h, c, enc_out, src_mask)
            idx = pred.argmax(dim=-1).item()
            if idx == self.eos_idx: break
            w = self.vocab.i2w.get(idx, UNK)
            if w not in (PAD, SOS, UNK): result.append(w)
            token = torch.tensor([idx]).to(src.device)
        return " ".join(result)


# ─── Load model once at startup ───────────────────────────────────────────────
MODEL_PATH = Path("model.pt")
VOCAB_PATH = Path("vocab.pkl")

model_loaded = False
seq2seq: Optional[Seq2Seq] = None
vocab_obj: Optional[Vocab] = None
gpu_name = ""

def load_model():
    global seq2seq, vocab_obj, model_loaded, gpu_name

    if not MODEL_PATH.exists() or not VOCAB_PATH.exists():
        print("⚠  model.pt / vocab.pkl not found. Run train.py first.")
        return

    print(f"Loading model on {DEVICE}...")
    with open(VOCAB_PATH, "rb") as f:
        vocab_obj = pickle.load(f)

    ckpt   = torch.load(MODEL_PATH, map_location=DEVICE)
    cfg    = ckpt["config"]

    enc = Encoder(len(vocab_obj), cfg["embed_dim"], cfg["hidden_dim"],
                  cfg["num_enc_layers"], cfg["dropout"])
    dec = Decoder(len(vocab_obj), cfg["embed_dim"], cfg["hidden_dim"],
                  cfg["hidden_dim"] * 2, cfg["num_dec_layers"], cfg["dropout"])
    seq2seq = Seq2Seq(enc, dec, vocab_obj).to(DEVICE)
    seq2seq.load_state_dict(ckpt["model_state"])
    seq2seq.eval()

    model_loaded = True
    if DEVICE.type == "cuda":
        gpu_name = torch.cuda.get_device_name(0)
    print(f"✓ Model loaded  ({DEVICE}  {gpu_name})")


# ─── Tokenize helper ──────────────────────────────────────────────────────────
def tokenize(text: str):
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s'.,!?]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text.split()


# ─── FastAPI app ──────────────────────────────────────────────────────────────
app = FastAPI(title="Seq2Seq Summarizer API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static files (index.html, etc.) from current directory
if Path("static").exists():
    app.mount("/static", StaticFiles(directory="static"), name="static")


class SummarizeRequest(BaseModel):
    text:       str
    max_length: int = 80


class SummarizeResponse(BaseModel):
    summary:     str
    input_words: int
    output_words: int
    compression: float
    device:      str
    gpu:         str


@app.on_event("startup")
def startup():
    load_model()


@app.get("/")
def index():
    if Path("index.html").exists():
        return FileResponse("index.html")
    return {"status": "Seq2Seq Summarizer API running"}


@app.get("/status")
def status():
    return {
        "model_loaded": model_loaded,
        "device":       str(DEVICE),
        "gpu":          gpu_name,
        "vocab_size":   len(vocab_obj) if vocab_obj else 0,
    }


@app.post("/summarize", response_model=SummarizeResponse)
def summarize(req: SummarizeRequest):
    if not model_loaded:
        raise HTTPException(503, "Model not loaded. Run train.py first.")

    text = req.text.strip()
    if not text:
        raise HTTPException(400, "Empty text.")

    tokens  = tokenize(text)
    src_ids = vocab_obj.encode(tokens, max_len=400)

    if len(src_ids) < 5:
        raise HTTPException(400, "Text too short after tokenization.")

    summary = seq2seq.summarize(src_ids, max_len=req.max_length)

    in_words  = len(tokens)
    out_words = len(summary.split())
    comp      = round((1 - out_words / max(in_words, 1)) * 100, 1)

    return SummarizeResponse(
        summary      = summary,
        input_words  = in_words,
        output_words = out_words,
        compression  = comp,
        device       = str(DEVICE),
        gpu          = gpu_name,
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=False)
