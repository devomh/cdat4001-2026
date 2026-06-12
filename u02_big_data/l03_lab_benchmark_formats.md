---
title: "Lab: Measuring CSV vs Parquet"
unit: "II"
lesson: "03"
type: lab
tags: [csv, parquet, duckdb, benchmark]
difficulty: introductory
duration: "45 mins"
---

**Goal:** stop *believing* Parquet is smaller and faster — **measure it** on a million
rows. Pairs with the concept note
[CSV vs Parquet: Choosing a Storage Format](l03_concept_csv_vs_parquet.qmd).

> This page is the read-only view. To run the lab, open the notebook
> (`l03_lab_benchmark_formats.ipynb`) — in Colab via the badge on the concept page, or locally.

> **Note:** timings below are from one machine. **Your absolute numbers will differ; the
> ratios are the lesson.**

## Prerequisites & Setup

Run this first. It builds the dataset itself (seeded — everyone gets identical *data*;
your *timings* will vary). 50 stores and 6 categories make the columns deliberately
repetitive, which is exactly what columnar compression loves.

```python
# Setup: run this cell first (required for Colab — it resets on open)
%pip install -q duckdb==1.4.2 pandas pyarrow numpy   # duckdb pinned; rest float (Colab preinstalls)

import numpy as np, pandas as pd, duckdb, os, time
os.makedirs("data", exist_ok=True)

rng = np.random.default_rng(7)                       # fixed seed: identical data everywhere
n = 1_000_000
cats = ["grocery", "fuel", "dining", "retail", "pharmacy", "online"]
txns = pd.DataFrame({
    "txn_id":   np.arange(n, dtype=np.int64),
    "store_id": rng.integers(1, 51, n),              # 50 stores: repetitive, compresses well
    "category": rng.choice(cats, n),                 # 6 categories: repetitive
    "amount":   np.round(rng.gamma(2.0, 20.0, n), 2),
    "ts":       pd.to_datetime("2025-01-01") + pd.to_timedelta(rng.integers(0, 90*24*3600, n), unit="s"),
})
print(txns.shape)
txns.head(3)
```

## Step 1: Write the Same Data in Both Formats, Compare Size

```python
txns.to_csv("data/txns.csv", index=False)
txns.to_parquet("data/txns.parquet", index=False)    # pyarrow writer, snappy compression

csv_mb = os.path.getsize("data/txns.csv") / 1e6
pq_mb  = os.path.getsize("data/txns.parquet") / 1e6
print(f"CSV    : {csv_mb:5.1f} MB")
print(f"Parquet: {pq_mb:5.1f} MB   ({csv_mb/pq_mb:.1f}x smaller)")
```

<details><summary>Expected Output (CSV size is exact; Parquet may vary slightly with the pyarrow version)</summary>

~~~text
CSV    :  42.7 MB
Parquet:  15.7 MB   (2.7x smaller)
~~~
*(Same information, 2.7x less disk — that's per-column compression on repetitive data.)*
</details>

## Step 2: Time the Identical Query on Each

We average `amount` by `category` on both files. `best_ms` runs each query three times
and keeps the best (warms the OS file cache so we time the engine, not the disk's mood).

```python
con = duckdb.connect()
query = "SELECT category, ROUND(AVG(amount),2) AS avg_amt FROM '{f}' GROUP BY category ORDER BY avg_amt DESC"

def best_ms(path, q=query, reps=3):
    best = float("inf")
    for _ in range(reps):
        t = time.perf_counter()
        con.sql(q.format(f=path)).fetchall()
        best = min(best, time.perf_counter() - t)
    return best * 1000

print(f"CSV    : {best_ms('data/txns.csv'):6.1f} ms")
print(f"Parquet: {best_ms('data/txns.parquet'):6.1f} ms")
```

<details><summary>Expected Output (machine-dependent; the ratio is the point)</summary>

~~~text
CSV    :   84.6 ms
Parquet:    5.4 ms
~~~
*(~16x faster: the query touches two columns, so Parquet reads two columns — CSV has to
parse all five, every byte, as text.)*
</details>

## Step 3: Confirm the Answers Are Identical

The format is an implementation detail — the *result* must be the same.

```python
con.sql(query.format(f="data/txns.parquet")).df()
```

<details><summary>Expected Output</summary>

~~~text
   category  avg_amt
0    online    40.07
1      fuel    40.05
2    retail    40.05
3    dining    40.00
4   grocery    39.96
5  pharmacy    39.95
~~~
*(All categories average about 40 by construction — the point of this lab is the size and
speed columns, not these values.)*
</details>

## Your Turn

### Exercise 1 — Does the Speed Gap Survive `SELECT *`? — *completion problem*

**Predict first:** Parquet's edge comes from *skipping columns*. If you ask for **all**
columns, should its advantage be bigger or smaller? Then **uncomment, complete, and run**:

```python
# star = "SELECT * FROM '{f}'"
# print(f"CSV    : {best_ms('data/txns.csv',     q=____):6.1f} ms")
# print(f"Parquet: {best_ms('data/txns.parquet', q=____):6.1f} ms")
```

<details><summary>Expected Output (machine-dependent) + interpretation</summary>

~~~text
CSV    :  507.5 ms
Parquet:  440.6 ms
~~~
*(The 16x advantage collapses to ~1.2x: with every column requested, nothing can be
skipped — Parquet keeps only its decompression edge. Column-skipping IS the speed story.)*
</details>

### Exercise 2 — when CSV is the right call

**Task:** in one or two sentences, name a realistic situation in this course where you'd
still hand someone a **CSV** instead of Parquet, and why.

<details><summary>One acceptable answer</summary>

Sharing a small results table with a classmate or instructor who'll eyeball it or open it
in a spreadsheet — interchange and human-readability beat scan speed when the file is tiny
and read once.
</details>

## Summary

- Parquet was **2.7x smaller** and the two-column aggregate **~16x faster** here — for
  free, just by choosing the format.
- The speed comes from **columnar layout + compression**: the engine reads only the
  columns a query touches (Exercise 1 is the proof — ask for everything and the gap
  collapses).
- Reach for Parquet for repeated analysis on large data; keep CSV for interchange and
  human-readable files.
