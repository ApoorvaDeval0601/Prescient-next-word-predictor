"""
train_predictor.py — Seq2Seq Next-Word Predictor (Encoder-Decoder + Attention)
Dataset : AG News (already cached locally)
Task    : Given N context words → predict the next 5 words

Install:
    pip install torch datasets tqdm
Run:
    python train_predictor.py
"""

import os, re, math, time, pickle, random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from collections import Counter
from tqdm import tqdm

os.makedirs("model", exist_ok=True)

# ─── Config ───────────────────────────────────────────────────────────────────
CONFIG = {
    "num_articles":  20000,   # articles to pull from AG News
    "context_len":   10,      # words fed into encoder
    "predict_len":   5,       # words to predict (decoder output)
    "stride":        3,       # sliding window stride (lower = more samples)
    "min_freq":      3,
    "max_vocab":     15000,

    "embed_dim":     256,
    "hidden_dim":    512,
    "num_layers":    2,
    "dropout":       0.3,

    "epochs":        15,
    "batch_size":    256,     # large batch fine for this task
    "lr":            1e-3,
    "grad_clip":     1.0,
    "tf_start":      0.9,
    "tf_end":        0.1,

    "model_path":    "model/model.pt",
    "vocab_path":    "model/vocab.pkl",
}

PAD, UNK, SOS, EOS = "<pad>", "<unk>", "<sos>", "<eos>"


# ─── Vocab ────────────────────────────────────────────────────────────────────
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
        print(f"  Vocab size: {len(self.i2w):,}")

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

    def __len__(self):
        return len(self.i2w)


def tokenize(text):
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s'.,!?-]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text.split()


# ─── Build sliding-window pairs ───────────────────────────────────────────────
def make_pairs(token_lists, context_len, predict_len, stride):
    """Slide a window over each token list → (context, target) pairs."""
    pairs = []
    win   = context_len + predict_len
    for tokens in token_lists:
        if len(tokens) < win:
            continue
        for i in range(0, len(tokens) - win + 1, stride):
            ctx = tokens[i : i + context_len]
            tgt = tokens[i + context_len : i + context_len + predict_len]
            pairs.append((ctx, tgt))
    return pairs


# ─── Dataset ──────────────────────────────────────────────────────────────────
class PredDataset(Dataset):
    def __init__(self, pairs, vocab):
        self.pairs = pairs
        self.vocab = vocab
        self.sos   = vocab.w2i[SOS]
        self.eos   = vocab.w2i[EOS]

    def __len__(self): return len(self.pairs)

    def __getitem__(self, idx):
        ctx, tgt = self.pairs[idx]
        src_ids  = self.vocab.encode(ctx)
        tgt_ids  = [self.sos] + self.vocab.encode(tgt) + [self.eos]
        return (torch.tensor(src_ids, dtype=torch.long),
                torch.tensor(tgt_ids, dtype=torch.long))


def collate(batch):
    srcs, tgts = zip(*batch)
    return (pad_sequence(srcs, batch_first=True, padding_value=0),
            pad_sequence(tgts, batch_first=True, padding_value=0))


# ─── Model ────────────────────────────────────────────────────────────────────
class Encoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_layers, dropout):
        super().__init__()
        self.embed      = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.dropout    = nn.Dropout(dropout)
        self.lstm       = nn.LSTM(embed_dim, hidden_dim, num_layers,
                                  batch_first=True, bidirectional=True,
                                  dropout=dropout if num_layers > 1 else 0)
        self.num_layers = num_layers
        self.fc_h       = nn.Linear(hidden_dim * 2, hidden_dim)
        self.fc_c       = nn.Linear(hidden_dim * 2, hidden_dim)

    def forward(self, src):
        enc_out, (h, c) = self.lstm(self.dropout(self.embed(src)))
        h = self._merge(h)
        c = self._merge(c)
        return enc_out, h, c

    def _merge(self, x):
        layers = []
        for i in range(self.num_layers):
            merged = torch.tanh(self.fc_h(torch.cat([x[2*i], x[2*i+1]], dim=-1)))
            layers.append(merged)
        return torch.stack(layers, dim=0)


class Attention(nn.Module):
    def __init__(self, hidden_dim, enc_out_dim):
        super().__init__()
        self.attn = nn.Linear(hidden_dim + enc_out_dim, hidden_dim)
        self.v    = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, dec_h, enc_out, src_mask):
        T       = enc_out.shape[1]
        dec_exp = dec_h.unsqueeze(1).expand(-1, T, -1)
        scores  = self.v(torch.tanh(self.attn(
                    torch.cat([dec_exp, enc_out], dim=-1)))).squeeze(-1)
        scores  = scores.masked_fill(src_mask == 0, -1e9)
        return F.softmax(scores, dim=-1)


class Decoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, enc_out_dim, num_layers, dropout):
        super().__init__()
        self.embed     = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.dropout   = nn.Dropout(dropout)
        self.attention = Attention(hidden_dim, enc_out_dim)
        self.lstm      = nn.LSTM(embed_dim + enc_out_dim, hidden_dim, num_layers,
                                 batch_first=True,
                                 dropout=dropout if num_layers > 1 else 0)
        self.fc_out    = nn.Linear(hidden_dim + enc_out_dim + embed_dim, vocab_size)

    def step(self, token, h, c, enc_out, src_mask):
        emb  = self.dropout(self.embed(token.unsqueeze(1)))
        attn = self.attention(h[-1], enc_out, src_mask)
        ctx  = attn.unsqueeze(1) @ enc_out
        out, (h, c) = self.lstm(torch.cat([emb, ctx], dim=-1), (h, c))
        pred = self.fc_out(torch.cat([out, ctx, emb], dim=-1))
        return pred.squeeze(1), h, c


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, vocab):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.vocab   = vocab
        self.sos     = vocab.w2i[SOS]
        self.eos     = vocab.w2i[EOS]
        self.pad     = vocab.w2i[PAD]

    def forward(self, src, tgt, tf_ratio=0.5):
        enc_out, h, c = self.encoder(src)
        src_mask = (src != self.pad)
        B, T     = tgt.shape
        outputs  = torch.zeros(B, T, len(self.vocab), device=src.device)
        token    = tgt[:, 0]
        for t in range(1, T):
            pred, h, c    = self.decoder.step(token, h, c, enc_out, src_mask)
            outputs[:, t] = pred
            token = tgt[:, t] if random.random() < tf_ratio else pred.argmax(-1)
        return outputs

    @torch.no_grad()
    def predict(self, context_tokens, num_words=5):
        """Given a list of context word strings, return the predicted next words."""
        self.eval()
        device   = next(self.parameters()).device
        src_ids  = self.vocab.encode(context_tokens, CONFIG["context_len"])
        src      = torch.tensor([src_ids], dtype=torch.long, device=device)
        src_mask = (src != self.pad)
        enc_out, h, c = self.encoder(src)
        token    = torch.tensor([self.sos], device=device)
        result   = []

        for _ in range(num_words):
            pred, h, c = self.decoder.step(token, h, c, enc_out, src_mask)
            # Suppress special tokens
            for special in (self.pad, self.vocab.w2i[UNK], self.sos):
                pred[0, special] = -1e9
            idx = pred.argmax(-1).item()
            if idx == self.eos:
                break
            w = self.vocab.i2w.get(idx, UNK)
            result.append(w)
            token = torch.tensor([idx], device=device)

        return result


# ─── Train / eval loops ───────────────────────────────────────────────────────
def run_epoch(model, loader, optimizer, criterion, clip, tf, device, train=True):
    model.train() if train else model.eval()
    total = 0
    ctx   = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for src, tgt in loader:
            src, tgt = src.to(device), tgt.to(device)
            if train: optimizer.zero_grad()
            out  = model(src, tgt, tf if train else 0.0)
            loss = criterion(out[:, 1:].reshape(-1, out.shape[-1]),
                             tgt[:, 1:].reshape(-1))
            if train:
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), clip)
                optimizer.step()
            total += loss.item()
    return total / len(loader)


# ─── Main ─────────────────────────────────────────────────────────────────────
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU   : {torch.cuda.get_device_name(0)}")

    # ── Data ──────────────────────────────────────────────────────────────────
    print("\n[1/5] Loading AG News...")
    from datasets import load_dataset
    raw = load_dataset("ag_news", split="train")

    print(f"[2/5] Building sliding-window pairs from {CONFIG['num_articles']:,} articles...")
    token_lists = []
    for item in tqdm(raw, total=CONFIG["num_articles"], desc="  Tokenizing", ncols=100):
        tokens = tokenize(item["text"])
        if len(tokens) >= CONFIG["context_len"] + CONFIG["predict_len"]:
            token_lists.append(tokens)
        if len(token_lists) >= CONFIG["num_articles"]:
            break

    all_pairs = make_pairs(token_lists, CONFIG["context_len"],
                           CONFIG["predict_len"], CONFIG["stride"])
    random.shuffle(all_pairs)
    split       = int(len(all_pairs) * 0.9)
    train_pairs = all_pairs[:split]
    val_pairs   = all_pairs[split:]
    print(f"  Total pairs — Train: {len(train_pairs):,}  |  Val: {len(val_pairs):,}")

    # ── Vocab ─────────────────────────────────────────────────────────────────
    print("[3/5] Building vocabulary...")
    vocab = Vocab()
    vocab.build(token_lists, min_freq=CONFIG["min_freq"],
                max_vocab=CONFIG["max_vocab"])
    with open(CONFIG["vocab_path"], "wb") as f:
        pickle.dump(vocab, f)
    print(f"  Saved → {CONFIG['vocab_path']}")

    # ── Model ─────────────────────────────────────────────────────────────────
    print("[4/5] Building model...")
    enc_out_dim = CONFIG["hidden_dim"] * 2
    enc = Encoder(len(vocab), CONFIG["embed_dim"], CONFIG["hidden_dim"],
                  CONFIG["num_layers"], CONFIG["dropout"])
    dec = Decoder(len(vocab), CONFIG["embed_dim"], CONFIG["hidden_dim"],
                  enc_out_dim, CONFIG["num_layers"], CONFIG["dropout"])
    model = Seq2Seq(enc, dec, vocab).to(device)
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")

    train_ds = PredDataset(train_pairs, vocab)
    val_ds   = PredDataset(val_pairs,   vocab)
    train_dl = DataLoader(train_ds, batch_size=CONFIG["batch_size"],
                          shuffle=True, collate_fn=collate, num_workers=0,
                          pin_memory=(device.type == "cuda"))
    val_dl   = DataLoader(val_ds, batch_size=CONFIG["batch_size"] * 2,
                          collate_fn=collate, num_workers=0,
                          pin_memory=(device.type == "cuda"))

    optimizer = torch.optim.AdamW(model.parameters(), lr=CONFIG["lr"], weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer, mode="min", patience=2, factor=0.5)
    criterion = nn.CrossEntropyLoss(ignore_index=0, label_smoothing=0.1)

    # ── Training ──────────────────────────────────────────────────────────────
    print(f"[5/5] Training {CONFIG['epochs']} epochs...\n")
    best_val = float("inf")

    for epoch in range(1, CONFIG["epochs"] + 1):
        tf = CONFIG["tf_start"] + (CONFIG["tf_end"] - CONFIG["tf_start"]) * \
             (epoch - 1) / CONFIG["epochs"]
        t0 = time.time()

        train_loss = run_epoch(model, train_dl, optimizer, criterion,
                               CONFIG["grad_clip"], tf, device, train=True)
        val_loss   = run_epoch(model, val_dl,   optimizer, criterion,
                               CONFIG["grad_clip"], tf, device, train=False)
        scheduler.step(val_loss)

        print(f"  Epoch {epoch:02d}/{CONFIG['epochs']}  "
              f"train={train_loss:.4f} (ppl {math.exp(min(train_loss,20)):.1f})  "
              f"val={val_loss:.4f} (ppl {math.exp(min(val_loss,20)):.1f})  "
              f"tf={tf:.2f}  {time.time()-t0:.0f}s")

        if val_loss < best_val:
            best_val = val_loss
            torch.save({"model_state": model.state_dict(), "config": CONFIG},
                       CONFIG["model_path"])
            print(f"  ✓ Best saved → {CONFIG['model_path']}\n")

    # ── Sample predictions ────────────────────────────────────────────────────
    ckpt = torch.load(CONFIG["model_path"], map_location=device)
    model.load_state_dict(ckpt["model_state"])

    print("\n─── Sample Predictions ─────────────────────────────────────")
    samples = [
        "the president of the united states announced a new",
        "scientists have discovered a new species of",
        "the stock market fell sharply today after",
        "apple released a new version of its",
        "the football team won the championship after",
    ]
    for text in samples:
        ctx    = tokenize(text)[-CONFIG["context_len"]:]
        preds  = model.predict(ctx, num_words=CONFIG["predict_len"])
        print(f"  IN  : ...{' '.join(ctx)}")
        print(f"  OUT : {' '.join(preds)}\n")

    print(f"✓ Done. Best val loss: {best_val:.4f}")


if __name__ == "__main__":
    main()