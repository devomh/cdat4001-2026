---
title: "Lab: Querying Files with DuckDB"
unit: "II"
lesson: "02"
type: lab
tags: [duckdb, sql, group-by, python]
difficulty: introductory
duration: "60 mins"
---

**Goal:** answer real questions about a `students` table using SQL — without ever loading
the whole file into memory. Pairs with the concept note
[Databases & SQL: Querying Where the Data Lives](l02_concept_databases_sql.qmd).

> This page is the read-only view. To run the lab, open the notebook (`l02_lab_querying_duckdb.ipynb`) — in Colab via the badge below, or locally.
>
> [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/devomh/cdat4001-2026/blob/main/u02_big_data/l02_lab_querying_duckdb.ipynb)

## Prerequisites & Setup

Run this first. It installs the packages, then **builds the dataset itself** (seeded, so
everyone gets identical numbers) and writes `data/students.csv`. No external download —
the lab is self-contained.

```python
# Setup: run this cell first (required for Colab — it resets on open)
%pip install -q duckdb==1.4.2 pandas numpy   # duckdb pinned (API we teach); pandas/numpy float — Colab preinstalls them
```

```python
import numpy as np, pandas as pd, duckdb, os
os.makedirs("data", exist_ok=True)

rng = np.random.default_rng(42)                      # fixed seed → reproducible
depts, n = ["CS", "Math", "Biology", "Business"], 60
students = pd.DataFrame({
    "student_id": np.arange(1001, 1001 + n),
    "department": rng.choice(depts, n, p=[.35, .25, .20, .20]),
    "gpa":        np.round(rng.normal(3.1, 0.45, n).clip(0, 4), 2),
    "credits":    rng.integers(0, 130, n),
    "enrolled":   rng.random(n) > 0.15,
    "admit_date": pd.to_datetime("2021-08-01") + pd.to_timedelta(rng.integers(0, 1200, n), unit="D"),
})
students.to_csv("data/students.csv", index=False)
print("wrote data/students.csv:", len(students), "rows")
```

## Step 1: Query a File Directly — the Naive Way vs. the DuckDB Way

The question: *the top 5 enrolled CS students with GPA ≥ 3.5, best first.*

**The naive way** — load the *entire* file into pandas, then keep what you need. Fine for
60 rows; a disaster for 60 million (you pay to read every byte into RAM first).

```python
# Naive: read everything into memory, then filter/sort in Python
df = pd.read_csv("data/students.csv")
(df[(df["department"] == "CS") & (df["enrolled"]) & (df["gpa"] >= 3.5)]
   [["student_id", "gpa", "credits"]]
   .sort_values("gpa", ascending=False)
   .head(5))
```

**The better way** — same question, but let DuckDB read the file and **push the filter
down**, returning only the rows you asked for. The same SQL would run unchanged on a
60-million-row file too big for RAM.

```python
con = duckdb.connect()                                  # in-memory analytical engine
con.sql("""
    SELECT student_id, gpa, credits
    FROM 'data/students.csv'                            -- query the file directly
    WHERE department = 'CS' AND enrolled AND gpa >= 3.5
    ORDER BY gpa DESC
    LIMIT 5
""").df()                                               # return result as a pandas DataFrame
```

<details><summary>Expected Output (both ways)</summary>

~~~text
    student_id   gpa  credits
43        1044  3.68       63     ← naive (pandas keeps the original row index, 43)
~~~
~~~text
   student_id   gpa  credits
0        1044  3.68       63     ← DuckDB (result is a fresh table, numbered from 0)
~~~
*(Same single row both ways — one student clears all three filters. The difference isn't
the answer; it's that DuckDB never loaded the other 59 rows.)*
</details>

## Step 2: Aggregate with GROUP BY — *completion problem*

We want **one row per department** with its count and average GPA, highest first. Most of
the query is written — **uncomment and complete the two marked clauses**, then run.

```python
# Uncomment and fill the two ____ blanks, then run:
# con.sql("""
#     SELECT department,
#            COUNT(*)           AS n,
#            ROUND(AVG(gpa), 2) AS avg_gpa
#     FROM 'data/students.csv'
#     GROUP BY ____            -- ← which column collapses into one row each?
#     ORDER BY ____ DESC       -- ← sort so the top average is first
# """).df()
```

<details><summary>Expected Output</summary>

~~~text
  department   n  avg_gpa
0         CS  18     3.06
1    Biology  17     3.00
2       Math  14     2.98
3   Business  11     2.97
~~~
*(CS is both the largest department and the highest-averaging — and the spread across
departments is small, about 0.09 GPA points.)*
</details>

## Your Turn

### Exercise 1 — filter + sort

**Task:** list the `student_id` and `credits` of students with **120 or more credits**
(near graduation), most credits first. **Hint:** `WHERE … >= 120`, then `ORDER BY … DESC`.

```python
# TODO: your query here — uncomment the scaffold and complete it
# con.sql("""
#     SELECT ...
#     FROM 'data/students.csv'
#     ...
# """).df()
```

<details><summary>Expected Output</summary>

~~~text
   student_id  credits
0        1025      128
1        1034      124
2        1053      123
3        1046      121
4        1024      120
~~~
</details>

### Exercise 2 — aggregate with a condition

**Task:** for **enrolled** students only, return each department's **maximum** GPA (call
it `best_gpa`), **sorted alphabetically by department** so the result is reproducible.
**Hint:** `WHERE enrolled`, `GROUP BY department`, `MAX(gpa)`, `ORDER BY department`.

```python
# TODO: your query here
```

<details><summary>Expected Output</summary>

~~~text
  department  best_gpa
0    Biology      3.46
1   Business      3.43
2         CS      3.68
3       Math      3.54
~~~
*(Numbers follow from the seed in the setup cell; the shape is one row per department.)*
</details>

## Summary

- A query engine lets you **ask the file a question** instead of loading it all into memory.
- `SELECT … FROM … WHERE … GROUP BY … ORDER BY` — filter rows, then collapse groups, then sort.
- The *same SQL* scales from a 60-row CSV to a file bigger than RAM — a payoff we'll
  quantify later in Unit II.
