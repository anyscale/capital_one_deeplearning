"""Build the card-transaction corpus used by notebooks 01 and 02.

Not imported by anything. This is a one-off provenance script: it records how
`datasets/card_transactions/train.parquet` on the tutorial's public S3 mirror was
produced, so the corpus can be rebuilt or resized later.

Source
------
IBM TabFormer synthetic credit-card transactions (Apache-2.0):
    https://github.com/IBM/TabFormer  ->  data/credit_card/transactions.tgz
~24.4M transactions over ~2,000 synthetic cardholders. Synthetic, so there is no
PII and nothing sensitive goes on screen in front of a room.

What it produces
----------------
Consecutive transactions are grouped per (User, Card) and rendered one per line:

    2002-09-01 06:21 | $134.09 | swipe | MCC 5300 | La Verne, CA

`TXNS_PER_SEQ` of those lines become one training example. At 4 lines that is
~107 gpt2 tokens, which fits the notebooks' SEQ_LEN=128 without truncation --
raise SEQ_LEN in the notebooks before raising TXNS_PER_SEQ here.

The output schema matches the AG News parquet this corpus replaced, so the
notebooks' Ray Data path is unchanged: `text` (string), `label` (int64).
`label` is 1 when any transaction in the sequence is flagged fraud. Nothing in
the notebooks reads it; it exists so the "fine-tune a fraud head on top of this
pretrained model" narrative has something concrete to point at.

Usage
-----
    python common/build_card_transactions.py

Then publish (unsigned reads must work -- every notebook syncs with
--no-sign-request):

    aws s3 sync /mnt/cluster_storage/datasets/card_transactions \\
        s3://anyscale-public-materials-use2/ray_summit_foundation_model_training_2026/datasets/card_transactions
"""

from __future__ import annotations

import os
import subprocess
import tarfile

import pandas as pd

TGZ_URL = "https://media.githubusercontent.com/media/IBM/TabFormer/main/data/credit_card/transactions.tgz"

WORK_DIR = "/mnt/local_storage/tabformer"
OUT_DIR = "/mnt/cluster_storage/datasets/card_transactions"

TXNS_PER_SEQ = 4        # ~107 gpt2 tokens; fits SEQ_LEN=128 without truncation
TARGET_SEQS = 24_000    # ~1.8 MB parquet, fast for a room of 60 to sync
NROWS = 2_000_000       # first ~200 cardholders; the CSV is sorted by User

COLS = ["User", "Card", "Year", "Month", "Day", "Time", "Amount",
        "Use Chip", "Merchant City", "Merchant State", "MCC", "Is Fraud?"]

CHIP = {"Swipe Transaction": "swipe", "Chip Transaction": "chip",
        "Online Transaction": "online"}


def fetch_csv() -> str:
    """Download and extract the TabFormer CSV, skipping work already done."""
    os.makedirs(WORK_DIR, exist_ok=True)
    tgz = os.path.join(WORK_DIR, "transactions.tgz")
    csv = os.path.join(WORK_DIR, "card_transaction.v1.csv")
    if not os.path.exists(csv):
        if not os.path.exists(tgz):
            # The repo file is a git-LFS pointer; media.githubusercontent.com
            # serves the real 278 MB object.
            subprocess.run(["curl", "-sL", "--fail", "-o", tgz, TGZ_URL], check=True)
        with tarfile.open(tgz) as t:
            t.extractall(WORK_DIR)
    return csv


def render(r) -> str:
    """One transaction -> one line of text."""
    where = "ONLINE" if str(r.city).strip().upper() == "ONLINE" else (
        f"{r.city}, {r.state}" if isinstance(r.state, str) else str(r.city))
    return (f"{r.Year:04d}-{r.Month:02d}-{r.Day:02d} {r.Time} | {r.Amount} | "
            f"{CHIP.get(r.chip, 'other')} | MCC {r.MCC} | {where}")


def main() -> None:
    df = pd.read_csv(fetch_csv(), nrows=NROWS, usecols=COLS)
    df = df.rename(columns={"Use Chip": "chip", "Merchant City": "city",
                            "Merchant State": "state", "Is Fraud?": "fraud"})
    df["line"] = [render(r) for r in df.itertuples(index=False)]
    df["is_fraud"] = (df["fraud"].astype(str).str.strip().str.lower() == "yes").astype("int64")

    texts, labels = [], []
    for _, g in df.groupby(["User", "Card"], sort=True):
        lines, flags = g["line"].tolist(), g["is_fraud"].tolist()
        for i in range(0, len(lines) - TXNS_PER_SEQ + 1, TXNS_PER_SEQ):
            texts.append("\n".join(lines[i:i + TXNS_PER_SEQ]))
            labels.append(int(max(flags[i:i + TXNS_PER_SEQ])))
            if len(texts) >= TARGET_SEQS:
                break
        if len(texts) >= TARGET_SEQS:
            break

    os.makedirs(OUT_DIR, exist_ok=True)
    out_path = os.path.join(OUT_DIR, "train.parquet")
    pd.DataFrame({"text": texts, "label": labels}).to_parquet(out_path, index=False)

    print(f"wrote {out_path}")
    print(f"  sequences={len(texts)}  with_fraud={sum(labels)}  "
          f"bytes={os.path.getsize(out_path):,}")
    print("--- first sequence ---")
    print(texts[0])


if __name__ == "__main__":
    main()
