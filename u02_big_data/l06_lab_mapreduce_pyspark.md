---
title: "Lab: MapReduce by Hand, then PySpark"
unit: "II"
lesson: "06"
type: lab
tags: [mapreduce, pyspark, spark, parquet, lazy-evaluation]
difficulty: intermediate
duration: "60 mins"
---

**Goal:** build the MapReduce paradigm with your bare hands (fifteen lines of plain
Python), then run the same idea industrially -- PySpark in local mode over the 1M-row
transactions table, reading the lazy plan before triggering it. Pairs with the concept
note [Computing at Scale: MapReduce & Spark](l06_concept_scaling_spark.qmd).

> This page is the read-only view. To run the lab, open the notebook
> (`l06_lab_mapreduce_pyspark.ipynb`) -- in Colab via the badge on the concept page, or
> locally (needs Java 11+; Colab has it preinstalled).

## Prerequisites & Setup

Run this first. It installs PySpark, checks Java, and **builds both datasets itself**: a
small text corpus (for hand-rolled MapReduce) and the seeded 1M-row transactions table
written to Parquet (for Spark).

```python
# Setup: run this cell first (required for Colab -- it resets on open)
%pip install -q pyspark==4.1.2 pandas numpy pyarrow   # pyspark pinned (API we teach); rest float -- Colab preinstalls them

import os, shutil
import numpy as np, pandas as pd

print("java found:", shutil.which("java") is not None)   # Spark needs a JVM
os.makedirs("data", exist_ok=True)

corpus = """Data science is the disciplined process for getting from a fuzzy question
to a defensible answer. The process is a loop: question, acquire, clean, explore,
communicate. Most of the work is not modeling. The data lives in files, in databases,
and behind web APIs. The format you choose changes the size of the data and the speed
of every query. When the data outgrows one machine, the work must be split across many
machines, and the answer must be assembled from the pieces. Splitting the work and
assembling the answer is the whole idea of computing at scale.
"""
with open("data/corpus.txt", "w") as f:
    f.write(corpus)

rng = np.random.default_rng(7)                       # fixed seed -> reproducible table
n = 1_000_000
CATEGORIES  = np.array(["grocery", "fuel", "pharmacy", "electronics", "clothing", "restaurant"])
MEAN_AMOUNT = np.array([ 32.0,      38.0,   24.0,       95.0,          48.0,       28.0])

cat_idx = rng.integers(0, len(CATEGORIES), n)
txns = pd.DataFrame({
    "txn_id":   np.arange(n),
    "store_id": rng.integers(1, 51, n),
    "category": CATEGORIES[cat_idx],
    "amount":   np.round(rng.gamma(2.0, MEAN_AMOUNT[cat_idx] / 2.0), 2),
    "ts":       pd.to_datetime("2025-01-01") + pd.to_timedelta(rng.integers(0, 365*24*3600, n), unit="s"),
})
txns["ts"] = txns["ts"].astype("datetime64[us]")     # Spark can't read pandas' nanosecond default
txns.to_parquet("data/txns.parquet", index=False)
print("wrote data/corpus.txt and data/txns.parquet:", len(txns), "rows")
```

~~~text
java found: True
wrote data/corpus.txt and data/txns.parquet: 1000000 rows
~~~

## Step 1: MapReduce with Your Bare Hands

Count word frequencies in the corpus using exactly the three phases from the concept
note -- map, shuffle, reduce -- in plain Python. No framework, nothing hidden.

```python
with open("data/corpus.txt") as f:
    lines = f.readlines()

def map_words(line):
    return [(word.strip(".,:"), 1) for word in line.lower().split()]

mapped = []
for line in lines:                       # MAP: one record in, many (key, value) pairs out
    mapped.extend(map_words(line))
print("mapped pairs:", len(mapped), "| first 4:", mapped[:4])

groups = {}
for word, one in mapped:                 # SHUFFLE: route every pair to its key's bucket
    groups.setdefault(word, []).append(one)
print("distinct words:", len(groups))

counts = {word: sum(ones) for word, ones in groups.items()}   # REDUCE: collapse buckets
top5 = sorted(counts, key=counts.get, reverse=True)[:5]       # five most frequent words
for word in top5:
    print(f"{word:12s} {counts[word]}")
```

<details><summary>Expected Output</summary>

~~~text
mapped pairs: 98 | first 4: [('data', 1), ('science', 1), ('is', 1), ('the', 1)]
distinct words: 60
the          15
data         4
is           4
of           4
and          4
~~~
*(Now picture the parallel version: each MAP loop iteration is independent, so fifty
machines could map fifty chunks at once without talking to each other. The SHUFFLE dict
is the only place data crosses machines -- pairs travel to their key's bucket. Each
REDUCE sum is again independent per key. The phases ARE the parallelism plan.)*
</details>

## Step 2: The Same Idea, Industrialized -- PySpark

Start a Spark session in **local mode**: driver and executors all on this machine, using
every core -- a one-machine rehearsal of a cluster.

```python
from pyspark.sql import SparkSession
from pyspark.sql import functions as F

spark = (SparkSession.builder
         .master("local[*]")                              # all local cores as executors
         .appName("CDAT4001-L06")
         .config("spark.ui.showConsoleProgress", "false") # keep the notebook quiet
         .getOrCreate())
spark.sparkContext.setLogLevel("ERROR")
print("Spark", spark.version, "ready")
```

~~~text
Spark 4.1.2 ready
~~~

*(Startup prints a few WARN lines about hostnames and native libraries -- normal noise,
not errors.)*

Read the Parquet table and check what arrived:

```python
txns_s = spark.read.parquet("data/txns.parquet")
print(txns_s.count(), "rows")
txns_s.printSchema()
```

<details><summary>Expected Output</summary>

~~~text
1000000 rows
root
 |-- txn_id: long (nullable = true)
 |-- store_id: long (nullable = true)
 |-- category: string (nullable = true)
 |-- amount: double (nullable = true)
 |-- ts: timestamp_ntz (nullable = true)
~~~
*(No type re-inference: the schema came straight out of the Parquet file -- L03's
schema-preservation payoff, now feeding a cluster engine.)*
</details>

Now the L03 question at Spark scale -- average amount by category. Note the shape:
`groupBy` + `agg` is SQL's GROUP BY wearing a Python method chain.

```python
result = (txns_s.groupBy("category")
          .agg(F.round(F.avg("amount"), 2).alias("avg_amount"),
               F.count("*").alias("n"))
          .orderBy("avg_amount", ascending=False))
print("returned instantly -- but where is the answer?")
```

~~~text
returned instantly -- but where is the answer?
~~~

That cell finished in milliseconds on a million rows. Suspicious -- and it's the lesson's
last big idea: **nothing has run yet.**

## Step 3: Lazy Evaluation -- Read the Plan, Then Trigger It

`result` is not a table; it's a **plan**. Ask Spark to show it:

```python
result.explain()
```

<details><summary>Expected Output</summary>

~~~text
== Physical Plan ==
AdaptiveSparkPlan isFinalPlan=false
+- Sort [avg_amount#14 DESC NULLS LAST], true, 0
   +- Exchange rangepartitioning(avg_amount#14 DESC NULLS LAST, 200), ENSURE_REQUIREMENTS, [plan_id=54]
      +- HashAggregate(keys=[category#2], functions=[avg(amount#3), count(1)])
         +- Exchange hashpartitioning(category#2, 200), ENSURE_REQUIREMENTS, [plan_id=51]
            +- HashAggregate(keys=[category#2], functions=[partial_avg(amount#3), partial_count(1)])
               +- FileScan parquet [category#2,amount#3] Batched: true, DataFilters: [], Format: Parquet, ... ReadSchema: struct<category:string,amount:double>
~~~
*(Read it bottom-up -- it is map, shuffle, reduce in costume. `FileScan` + the first
`HashAggregate(partial_avg)` = MAP: each executor pre-aggregates its own chunk.
`Exchange hashpartitioning(category)` = SHUFFLE: partial results travel to their key's
machine. The second `HashAggregate` = REDUCE. And the last line is L03 vindicated at
scale: `ReadSchema` lists only `category` and `amount` -- Spark reads two of five columns
from the Parquet file.)*
</details>

An **action** triggers the plan. `.show()` is one:

```python
result.show()
```

<details><summary>Expected Output</summary>

~~~text
+-----------+----------+------+
|   category|avg_amount|     n|
+-----------+----------+------+
|electronics|     95.17|166818|
|   clothing|     47.97|167523|
|       fuel|     37.92|166169|
|    grocery|     32.06|166757|
| restaurant|     28.01|166265|
|   pharmacy|     24.01|166468|
+-----------+----------+------+
~~~
*(NOW it ran -- a few seconds, because executors actually scanned the million rows.
Electronics averages roughly four times groceries, with each category near one sixth of
the data. The same code would run unchanged on a forty-machine cluster against forty
terabytes; only `master(...)` changes.)*
</details>

## Your Turn

### Exercise 1 -- Busiest Stores: *completion problem*

**Task:** which 5 stores processed the most transactions? **Uncomment and fill the two
`____` blanks**, then run.

```python
# Uncomment and fill the two ____ blanks, then run:
# (txns_s.groupBy(____)                            # which column identifies a store?
#  .agg(F.count("*").alias("n_txns"))
#  .orderBy(____, ascending=False)                 # biggest first (use the alias string)
#  .limit(5)
#  .show())
```

<details><summary>Expected Output</summary>

~~~text
+--------+------+
|store_id|n_txns|
+--------+------+
|      34| 20363|
|       6| 20199|
|       9| 20199|
|      11| 20197|
|      37| 20173|
+--------+------+
~~~
*(Every store hovers near 20,000 -- one millionth of the data over 50 stores -- because
the generator assigns stores uniformly. A real retailer's table would be lumpy; flat
counts are the signature of synthetic data, worth recognizing.)*
</details>

### Exercise 2 -- Revenue, From Scratch

**Task:** compute each category's **total revenue** (the sum of `amount`), largest
first -- the full chain (group, aggregate, order, show) written by you, no scaffold.
**Hint:** `F.round(F.sum("amount"), 0).cast("long").alias("revenue")` gives clean
integer dollars; the chaining itself is yours.

```python
# TODO: your code here
```

<details><summary>Expected Output</summary>

~~~text
+-----------+--------+
|   category| revenue|
+-----------+--------+
|electronics|15876559|
|   clothing| 8035301|
|       fuel| 6301665|
|    grocery| 5345564|
| restaurant| 4657872|
|   pharmacy| 3997617|
+-----------+--------+
~~~
*(Electronics earns roughly double clothing and four times pharmacy -- yet Exercise 1
showed traffic is uniform. Cross-check Step 3: it is the average ticket, not the number
of transactions, driving this ranking. Two aggregations, one story.)*
</details>

### Exercise 3 -- Your Call: Laptop or Cluster?

**Task:** for each scenario, decide **laptop** (DuckDB/pandas) or **cluster** (Spark) and
justify it in one or two sentences, using the concept note's "when NOT" list.

- **Scenario A:** a 200 MB CSV of monthly sales; you need one summary table per month,
  refreshed daily.
- **Scenario B:** 40 TB of clickstream logs in distributed storage; the whole history is
  re-processed every night.

*Your answer here.*

<details><summary>One Good Answer</summary>

~~~text
A: Laptop. 200 MB is far below one machine's ceiling -- DuckDB aggregates it in well
under a second, and Spark's startup overhead alone would cost more than the whole job.
Convert to Parquet if it is reread daily.

B: Cluster. 40 TB exceeds any single machine's disk and RAM, and the data already lives
in distributed storage -- moving it out nightly would cost more than computing where it
sits. This is exactly the move-computation-to-the-data case.
~~~
</details>

## Summary

- **MapReduce in one line:** map = transform each chunk where it lives, shuffle = the
  only data movement (route by key), reduce = collapse each key's bucket.
- Your L02 `GROUP BY` had this shape all along -- `.explain()` showed the costume.
- **Lazy evaluation:** transformations build a plan; only actions (`.show()`,
  `.count()`, writes) run it -- which is what lets Spark optimize (and prune columns,
  the L03 payoff).
- Local mode is a rehearsal: the same code scales out by changing `master(...)`.
- Default order of escalation: sample, laptop, one big machine, cluster -- most projects
  stop early, and that is good judgment, not timidity.
