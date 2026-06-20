---
title: "Lab: Big Data Challenges, Made Concrete -- CSV vs Parquet on a Million Rows"
unit: "II"
lesson: "03"
type: lab
tags: [big-data, volume, csv, parquet, columnar, benchmark]
difficulty: introductory
duration: "75 mins"
---

**Goal:** stop *believing* that storage format matters and **measure it**. You will build a
million-row table, save it as CSV and as Parquet, and see the "volume" challenge of big data
made concrete -- on size and on how much the computer must read to answer one question. Pairs
with the concept note
[Big Data: Definitions, the Vs, and the Single-Machine Ceiling](l03_concept_big_data_challenges.qmd).

> This page is the read-only view. To run the lab, open the notebook (`l03_lab_big_data_challenges.ipynb`) -- in Colab via the badge below, or locally. Every output below comes from actually running the cells.
>
> [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/devomh/cdat4001-2026/blob/main/u02_big_data/l03_lab_big_data_challenges.ipynb)

## Prerequisites & Setup

Run the setup cells first. The data is **built here, seeded**, so everyone gets identical
numbers -- there is no external download (real data acquisition is L06). Fifty stores and six
categories make the columns deliberately repetitive, which is exactly what columnar
compression is good at.

```python
# Setup cell 1 of 2: install only (Colab resets on open; run this first)
%pip install -q numpy pandas pyarrow
```

```python
# Setup cell 2 of 2: imports + build the seeded synthetic transactions table
import numpy as np
import pandas as pd
import os

rng = np.random.default_rng(7)               # fixed seed -> identical data for everyone
n = 1_000_000
categories = ["grocery", "fuel", "dining", "retail", "pharmacy", "online"]
txns = pd.DataFrame({
    "txn_id":   np.arange(n, dtype=np.int64),
    "store_id": rng.integers(1, 51, n),                 # 50 stores: repetitive, compresses well
    "category": rng.choice(categories, n),              # 6 categories: repetitive
    "amount":   np.round(rng.gamma(2.0, 20.0, n), 2),   # transaction amount, dollars
    "ts":       pd.to_datetime("2025-01-01") + pd.to_timedelta(rng.integers(0, 90*24*3600, n), unit="s"),
})
print("rows:", len(txns), "| columns:", list(txns.columns))
```

~~~text
rows: 1000000 | columns: ['txn_id', 'store_id', 'category', 'amount', 'ts']
~~~

A peek at the first three rows -- five columns, a million rows:

```python
print(txns.head(3))
```

~~~text
   txn_id  store_id category  amount                  ts
0       0        48   retail   10.43 2025-03-08 11:00:19
1       1        32     fuel    5.61 2025-03-10 20:45:39
2       2        35     fuel   12.50 2025-03-16 02:07:54
~~~

## Step 1: Same Data, Two Formats -- Compare Size

Write the identical table as CSV (text, row by row) and as Parquet (binary, column by column),
then compare the file sizes. This is the **volume** V you can feel: same information, very
different footprint.

```python
os.makedirs("data", exist_ok=True)
txns.to_csv("data/txns.csv", index=False)
txns.to_parquet("data/txns.parquet", index=False)   # columnar, compressed (pyarrow)

csv_mb = os.path.getsize("data/txns.csv") / 1e6
pq_mb  = os.path.getsize("data/txns.parquet") / 1e6
print(f"CSV    : {csv_mb:5.1f} MB")
print(f"Parquet: {pq_mb:5.1f} MB  ({csv_mb / pq_mb:.1f}x smaller)")
```

~~~text
CSV    :  42.7 MB
Parquet:  15.7 MB  (2.7x smaller)
~~~

Same data, 2.7x less disk -- that is per-column compression working on repetitive columns.

## Step 2: Why Columnar Is Cheaper to Read

Parquet stores each column separately, so a reader can pull **just the columns a question
needs**. Ask for two of the five:

```python
two_cols = pd.read_parquet("data/txns.parquet", columns=["category", "amount"])
print("columns read from Parquet:", list(two_cols.columns), "| shape:", two_cols.shape)
```

~~~text
columns read from Parquet: ['category', 'amount'] | shape: (1000000, 2)
~~~

The other three columns were never read off disk. CSV cannot do this: it is text stored row by
row, so to reach `amount` it must still parse every byte of every row.

Now answer one real question -- *average amount per category* (the same `groupby` you used in
L02) -- from **both** formats, and confirm the answer is identical. The format is an
implementation detail; the result must not change.

```python
from_csv = pd.read_csv("data/txns.csv").groupby("category")["amount"].mean().round(2)
from_pq  = (pd.read_parquet("data/txns.parquet", columns=["category", "amount"])
              .groupby("category")["amount"].mean().round(2))
print("CSV and Parquet give the same answer:", from_csv.equals(from_pq))
print(from_pq.sort_values(ascending=False))
```

~~~text
CSV and Parquet give the same answer: True
category
online      40.07
fuel        40.05
retail      40.05
dining      40.00
grocery     39.96
pharmacy    39.95
Name: amount, dtype: float64
~~~

(All categories average about 40 by construction -- the lesson here is the *cost* of getting
the answer, not the values.)

## Step 3: Completion Problem

Your turn to finish one line. Pull **only** the `amount` column from Parquet (one column out
of five) and print its overall mean. Uncomment the two lines, complete the `columns=[...]`
list, and run.

```python
# one_col = pd.read_parquet("data/txns.parquet", columns=[__________])
# print("overall mean amount:", round(one_col["amount"].mean(), 2))
```

<details>
<summary>Expected output</summary>

~~~text
overall mean amount: 40.01
~~~

</details>

## Your Turn (Independent)

### Measure it yourself -- which format is faster?

Parquet's edge is that it **reads less**. This cell times the same two-column average on each
format and reports which won on your machine. The absolute milliseconds depend on your
computer and will change run to run; the point is *which* format the engine reads less of.

```python
import time

def best_ms(load, reps=3):
    best = float("inf")
    for _ in range(reps):
        t = time.perf_counter()
        load()
        best = min(best, time.perf_counter() - t)
    return best * 1000

# Note: usecols still parses the whole CSV (it is row-by-row text); it only avoids
# building the columns we discard -- so this stays a fair "reads less" comparison.
csv_ms = best_ms(lambda: pd.read_csv("data/txns.csv", usecols=["category", "amount"])
                          .groupby("category")["amount"].mean())
pq_ms  = best_ms(lambda: pd.read_parquet("data/txns.parquet", columns=["category", "amount"])
                          .groupby("category")["amount"].mean())
print("Faster format on this machine:", "Parquet" if pq_ms < csv_ms else "CSV")
print("(Exact milliseconds vary by machine and run -- columnar reads only the needed columns.)")
```

~~~text
Faster format on this machine: Parquet
(Exact milliseconds vary by machine and run -- columnar reads only the needed columns.)
~~~

### When would you still hand someone a CSV?

In one or two sentences, name a realistic situation in this course where you would hand a
classmate or instructor a **CSV** instead of Parquet, and why.

<details>
<summary>One acceptable answer</summary>

Sharing a small results table that someone will eyeball or open in a spreadsheet:
interchange and human-readability beat scan speed when the file is tiny and read once.

</details>

## Summary

| Task | Tool / call |
|------|-------------|
| Build a seeded table | `pd.DataFrame({...})` with `np.random.default_rng(7)` |
| Write CSV / Parquet | `df.to_csv(...)`, `df.to_parquet(...)` |
| File size on disk | `os.path.getsize(path) / 1e6` |
| Read only some columns | `pd.read_parquet(path, columns=[...])` |
| Group and summarize | `df.groupby("k")["v"].mean()` |

- **Volume, made concrete:** the same million rows took **2.7x** less disk as Parquet -- for
  free, just by choosing the format.
- **Why columnar reads less:** Parquet stores columns separately, so a query reads only the
  columns it needs; CSV is row-text and must scan everything.
- **Rule of thumb:** Parquet for repeated analysis on large data; CSV for interchange and
  small, human-readable files. Neither of these is "big data" yet -- that line is **L04**.
