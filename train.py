"""
train.py — Seq2Seq Abstractive Summarizer (Encoder-Decoder + Attention)
RTX 5060 Ti compatible (CUDA / CPU fallback)

Uses CNN/DailyMail-style article→highlight pairs via HuggingFace datasets.
Saves: model.pt, vocab.pkl

Run:
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
    pip install datasets transformers rouge-score tqdm
    python train.py
"""

import math, time, pickle, random, re
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from collections import Counter
from tqdm import tqdm

# ─── Config ───────────────────────────────────────────────────────────────────
CONFIG = {
    # Data
    "num_train":       20000,     # more data = better generalisation
    "max_src_len":     300,       # slightly longer articles
    "max_tgt_len":     60,        # slightly longer summaries
    "min_freq":        2,
    "max_vocab":       15000,     # bigger vocab but still capped

    # Model  — one step up from the last run
    "embed_dim":       256,
    "hidden_dim":      512,
    "num_enc_layers":  2,
    "num_dec_layers":  2,
    "dropout":         0.4,       # slightly higher dropout to prevent overfit

    # Training
    "epochs":          15,
    "batch_size":      64,
    "lr":              1e-3,
    "grad_clip":       5.0,
    "teacher_forcing": 0.5,

    # Paths
    "model_path":      "model.pt",
    "vocab_path":      "vocab.pkl",
}

# ─── Special tokens ───────────────────────────────────────────────────────────
PAD, UNK, SOS, EOS = "<pad>", "<unk>", "<sos>", "<eos>"


# ─── Vocabulary ───────────────────────────────────────────────────────────────
class Vocab:
    def __init__(self):
        self.w2i = {PAD: 0, UNK: 1, SOS: 2, EOS: 3}
        self.i2w = {0: PAD, 1: UNK, 2: SOS, 3: EOS}

    def build(self, sentences, min_freq, max_vocab=None):
        counter = Counter(w for sent in sentences for w in sent)
        most_common = counter.most_common(max_vocab) if max_vocab else counter.items()
        for w, c in most_common:
            if c >= min_freq and w not in self.w2i:
                idx = len(self.w2i)
                self.w2i[w] = idx
                self.i2w[idx] = w
        print(f"Vocab size: {len(self.w2i):,}  (capped at {max_vocab or 'unlimited'})")

    def encode(self, tokens, max_len=None):
        ids = [self.w2i.get(t, 1) for t in tokens]
        if max_len:
            ids = ids[:max_len]
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


def tokenize(text):
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s'.,!?]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text.split()


# ─── Dataset ──────────────────────────────────────────────────────────────────
class SumDataset(Dataset):
    def __init__(self, pairs, vocab, src_len, tgt_len):
        self.data    = pairs
        self.vocab   = vocab
        self.src_len = src_len
        self.tgt_len = tgt_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        src_tokens, tgt_tokens = self.data[idx]
        src_ids = self.vocab.encode(src_tokens, self.src_len)
        tgt_ids = ([self.vocab.w2i[SOS]]
                   + self.vocab.encode(tgt_tokens, self.tgt_len)
                   + [self.vocab.w2i[EOS]])
        return torch.tensor(src_ids, dtype=torch.long), torch.tensor(tgt_ids, dtype=torch.long)


def collate(batch):
    srcs, tgts = zip(*batch)
    src_pad = pad_sequence(srcs, batch_first=True, padding_value=0)
    tgt_pad = pad_sequence(tgts, batch_first=True, padding_value=0)
    return src_pad, tgt_pad


# ─── Model ────────────────────────────────────────────────────────────────────
class Encoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_layers, dropout):
        super().__init__()
        self.embed      = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.dropout    = nn.Dropout(dropout)
        self.lstm       = nn.LSTM(embed_dim, hidden_dim, num_layers,
                                  batch_first=True,
                                  dropout=dropout if num_layers > 1 else 0,
                                  bidirectional=True)
        self.fc_h       = nn.Linear(hidden_dim * 2, hidden_dim)
        self.fc_c       = nn.Linear(hidden_dim * 2, hidden_dim)
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
        T         = enc_out.shape[1]
        dec_h_exp = dec_h.unsqueeze(1).repeat(1, T, 1)
        energy    = torch.tanh(self.attn(torch.cat([dec_h_exp, enc_out], dim=-1)))
        scores    = self.v(energy).squeeze(-1)
        if src_mask is not None:
            scores = scores.masked_fill(src_mask == 0, -1e10)
        return F.softmax(scores, dim=-1)


class Decoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, enc_hid, num_layers, dropout):
        super().__init__()
        self.embed     = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.dropout   = nn.Dropout(dropout)
        self.attention = Attention(enc_hid, hidden_dim)
        self.lstm      = nn.LSTM(embed_dim + enc_hid, hidden_dim, num_layers,
                                 batch_first=True,
                                 dropout=dropout if num_layers > 1 else 0)
        self.fc_out    = nn.Linear(hidden_dim + enc_hid + embed_dim, vocab_size)

    def forward_step(self, token, hidden, cell, enc_out, src_mask=None):
        emb     = self.dropout(self.embed(token.unsqueeze(1)))
        a       = self.attention(hidden[-1], enc_out, src_mask)
        ctx     = (a.unsqueeze(1) @ enc_out)
        lstm_in = torch.cat([emb, ctx], dim=-1)
        out, (h, c) = self.lstm(lstm_in, (hidden, cell))
        pred    = self.fc_out(torch.cat([out, ctx, emb], dim=-1))
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

    def forward(self, src, tgt, teacher_forcing=0.5):
        B, T_tgt = tgt.shape
        V        = len(self.vocab)
        enc_out, (h, c) = self.encoder(src)
        src_mask = (src != self.pad_idx)
        outputs  = torch.zeros(B, T_tgt, V).to(src.device)
        token    = tgt[:, 0]

        for t in range(1, T_tgt):
            pred, h, c = self.decoder.forward_step(token, h, c, enc_out, src_mask)
            outputs[:, t] = pred
            token = tgt[:, t] if random.random() < teacher_forcing else pred.argmax(dim=-1)

        return outputs

    @torch.no_grad()
    def summarize(self, src_ids, max_len=60):
        """Greedy decode with repetition penalty."""
        self.eval()
        src      = torch.tensor([src_ids], dtype=torch.long).to(next(self.parameters()).device)
        src_mask = (src != self.pad_idx)
        enc_out, (h, c) = self.encoder(src)
        token    = torch.tensor([self.sos_idx]).to(src.device)
        result   = []
        seen     = {}   # token_id → last step seen

        for step in range(max_len):
            pred, h, c = self.decoder.forward_step(token, h, c, enc_out, src_mask)

            # Penalise recently repeated tokens
            for tok_id, last_step in seen.items():
                penalty = 3.0 if step - last_step < 4 else 1.3
                pred[0, tok_id] /= penalty

            idx = pred.argmax(dim=-1).item()
            if idx == self.eos_idx:
                break
            w = self.vocab.i2w.get(idx, UNK)
            if w not in (PAD, SOS, UNK):
                result.append(w)
            seen[idx] = step
            token = torch.tensor([idx]).to(src.device)

        return " ".join(result)


# ─── Training loop ────────────────────────────────────────────────────────────
def train_epoch(model, loader, optimizer, criterion, clip, tf_ratio, device, epoch, total_epochs):
    model.train()
    total_loss = 0

    pbar = tqdm(loader,
                desc=f"  Epoch {epoch:02d}/{total_epochs} [Train]",
                unit="batch", ncols=100, leave=True)

    for src, tgt in pbar:
        src, tgt = src.to(device), tgt.to(device)
        optimizer.zero_grad()
        output   = model(src, tgt, teacher_forcing=tf_ratio)
        out_flat = output[:, 1:].reshape(-1, output.shape[-1])
        tgt_flat = tgt[:, 1:].reshape(-1)
        loss     = criterion(out_flat, tgt_flat)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        total_loss += loss.item()
        pbar.set_postfix(loss=f"{loss.item():.4f}")

    return total_loss / len(loader)


def evaluate(model, loader, criterion, device, epoch, total_epochs):
    model.eval()
    total_loss = 0

    pbar = tqdm(loader,
                desc=f"  Epoch {epoch:02d}/{total_epochs} [Val]  ",
                unit="batch", ncols=100, leave=True)

    with torch.no_grad():
        for src, tgt in pbar:
            src, tgt = src.to(device), tgt.to(device)
            output   = model(src, tgt, teacher_forcing=0)
            out_flat = output[:, 1:].reshape(-1, output.shape[-1])
            tgt_flat = tgt[:, 1:].reshape(-1)
            loss     = criterion(out_flat, tgt_flat)
            total_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}")

    return total_loss / len(loader)


# ─── Main ─────────────────────────────────────────────────────────────────────
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU   : {torch.cuda.get_device_name(0)}")

    print("\n[1/5] Loading CNN/DailyMail dataset...")
    from datasets import load_dataset
    raw = load_dataset("cnn_dailymail", "3.0.0", split="train")

    print(f"[2/5] Preprocessing {CONFIG['num_train']} samples...")
    pairs = []
    for item in tqdm(raw, desc="  Tokenizing", unit="doc", ncols=100,
                     total=CONFIG["num_train"]):
        src_tok = tokenize(item["article"])
        tgt_tok = tokenize(item["highlights"])
        if len(src_tok) > 20 and len(tgt_tok) > 5:
            pairs.append((src_tok, tgt_tok))
        if len(pairs) >= CONFIG["num_train"]:
            break

    random.shuffle(pairs)
    split = int(len(pairs) * 0.9)
    train_pairs, val_pairs = pairs[:split], pairs[split:]
    print(f"  Train: {len(train_pairs):,}  |  Val: {len(val_pairs):,}")

    print("[3/5] Building vocabulary...")
    vocab = Vocab()
    vocab.build(
        [p[0] for p in train_pairs] + [p[1] for p in train_pairs],
        min_freq=CONFIG["min_freq"],
        max_vocab=CONFIG["max_vocab"],
    )
    with open(CONFIG["vocab_path"], "wb") as f:
        pickle.dump(vocab, f)
    print(f"  Saved → {CONFIG['vocab_path']}")

    print("[4/5] Building model...")
    enc = Encoder(len(vocab), CONFIG["embed_dim"], CONFIG["hidden_dim"],
                  CONFIG["num_enc_layers"], CONFIG["dropout"])
    dec = Decoder(len(vocab), CONFIG["embed_dim"], CONFIG["hidden_dim"],
                  CONFIG["hidden_dim"] * 2,
                  CONFIG["num_dec_layers"], CONFIG["dropout"])
    model = Seq2Seq(enc, dec, vocab).to(device)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Parameters: {total_params:,}")

    train_ds = SumDataset(train_pairs, vocab, CONFIG["max_src_len"], CONFIG["max_tgt_len"])
    val_ds   = SumDataset(val_pairs,   vocab, CONFIG["max_src_len"], CONFIG["max_tgt_len"])
    train_dl = DataLoader(train_ds, batch_size=CONFIG["batch_size"],
                          shuffle=True, collate_fn=collate,
                          num_workers=0, pin_memory=(device.type == "cuda"))
    val_dl   = DataLoader(val_ds, batch_size=CONFIG["batch_size"] * 2,
                          collate_fn=collate,
                          num_workers=0, pin_memory=(device.type == "cuda"))

    optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG["lr"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=CONFIG["epochs"], eta_min=1e-5)
    criterion = nn.CrossEntropyLoss(ignore_index=0)

    print(f"[5/5] Training for {CONFIG['epochs']} epochs on {device}...\n")
    best_val = float("inf")
    history  = []

    for epoch in range(1, CONFIG["epochs"] + 1):
        t0 = time.time()

        train_loss = train_epoch(model, train_dl, optimizer, criterion,
                                 CONFIG["grad_clip"], CONFIG["teacher_forcing"],
                                 device, epoch, CONFIG["epochs"])

        val_loss = evaluate(model, val_dl, criterion,
                            device, epoch, CONFIG["epochs"])

        scheduler.step()
        elapsed = time.time() - t0

        train_ppl = math.exp(min(train_loss, 20))
        val_ppl   = math.exp(min(val_loss,   20))
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})

        print(f"  → Epoch {epoch:02d}/{CONFIG['epochs']}  "
              f"Train Loss: {train_loss:.4f} (ppl {train_ppl:.1f})  "
              f"Val Loss: {val_loss:.4f} (ppl {val_ppl:.1f})  "
              f"LR: {scheduler.get_last_lr()[0]:.2e}  "
              f"Time: {elapsed:.1f}s")

        if val_loss < best_val:
            best_val = val_loss
            torch.save({
                "model_state": model.state_dict(),
                "config":      CONFIG,
                "vocab_size":  len(vocab),
                "history":     history,
            }, CONFIG["model_path"])
            print(f"  ✓ Best model saved → {CONFIG['model_path']}\n")
        else:
            print()

    # ─── Sample inference ─────────────────────────────────────────────────────
    print("\n─── Sample Summaries ───────────────────────────────────────")
    sample_texts = [p[0] for p in val_pairs[:3]]
    sample_refs  = [vocab.decode([vocab.w2i.get(w, 1) for w in p[1]]) for p in val_pairs[:3]]

    ckpt = torch.load(CONFIG["model_path"], map_location=device)
    model.load_state_dict(ckpt["model_state"])

    for i, (src_tok, ref) in enumerate(zip(sample_texts, sample_refs)):
        src_ids = vocab.encode(src_tok, CONFIG["max_src_len"])
        pred    = model.summarize(src_ids, max_len=CONFIG["max_tgt_len"])
        print(f"\n[{i+1}] Source (first 30 words): {' '.join(src_tok[:30])}...")
        print(f"    Reference : {ref}")
        print(f"    Predicted : {pred}")

    print(f"\n✓ Training complete.")
    print(f"  Best Val Loss : {best_val:.4f}  (ppl {math.exp(min(best_val, 20)):.1f})")
    print(f"  Saved: {CONFIG['model_path']}, {CONFIG['vocab_path']}")


# ─── Entry point ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    main()