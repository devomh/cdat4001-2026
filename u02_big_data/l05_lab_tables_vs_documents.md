---
title: "Lab: Tables vs Documents"
unit: "II"
lesson: "05"
type: lab
tags: [nosql, tinydb, duckdb, json-normalize, schema-on-read]
difficulty: introductory
duration: "45 mins"
---

**Goal:** store one deliberately ragged course catalog twice -- as a flattened relational
table (DuckDB) and as documents kept whole (TinyDB) -- run the identical question both
ways, and feel the trade. Pairs with the concept note
[NoSQL: When Tables Don't Fit](l05_concept_nosql.qmd).

> This page is the read-only view. To run the lab, open the notebook (`l05_lab_tables_vs_documents.ipynb`) -- in Colab via the badge below, or locally.
>
> [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/devomh/cdat4001-2026/blob/main/u02_big_data/l05_lab_tables_vs_documents.ipynb)

## Prerequisites & Setup

Run this first. It **builds the dataset itself** (seeded, so everyone gets identical
documents) and writes `data/courses.json`: 60 course documents where some have a
`prerequisites` list, some charge a `lab_fee`, and all nest their `schedule` -- ragged on
purpose, because that raggedness is the whole lesson.

```python
# Setup: run this cell first (required for Colab -- it resets on open)
%pip install -q tinydb==4.8.2 duckdb==1.4.2 pandas numpy   # tinydb/duckdb pinned (APIs we teach); pandas/numpy float -- Colab preinstalls them

import json, os
import numpy as np, pandas as pd, duckdb
from tinydb import TinyDB, Query

os.makedirs("data", exist_ok=True)

rng = np.random.default_rng(11)                     # fixed seed -> reproducible catalog
DEPTS  = ["CS", "Math", "Biology", "Business"]
PREFIX = {"CS": "COMP", "Math": "MATH", "Biology": "BIOL", "Business": "BUSI"}
TOPICS = {
    "CS":       ["Programming", "Data Structures", "Databases", "Networks", "AI"],
    "Math":     ["Calculus", "Algebra", "Statistics", "Geometry", "Discrete Math"],
    "Biology":  ["Genetics", "Ecology", "Microbiology", "Botany", "Zoology"],
    "Business": ["Accounting", "Marketing", "Finance", "Management", "Economics"],
}
LEVELS = ["Introduction to", "Topics in", "Advanced"]

courses = []
for i in range(60):
    dept = str(rng.choice(DEPTS, p=[.35, .25, .20, .20]))
    course = {
        "code": f"{PREFIX[dept]}{3000 + i}",
        "title": f"{rng.choice(LEVELS)} {rng.choice(TOPICS[dept])}",
        "department": dept,
        "credits": int(rng.integers(1, 5)),
        "schedule": {"days": str(rng.choice(["MWF", "TTh"])),
                     "start": f"{int(rng.integers(7, 20)):02d}:00"},
    }
    same_dept = [c["code"] for c in courses if c["department"] == dept]
    if same_dept and rng.random() < 0.40:           # SOME courses have prerequisites
        k = min(int(rng.integers(1, 3)), len(same_dept))
        course["prerequisites"] = list(rng.choice(same_dept, k, replace=False))
    if rng.random() < (0.45 if dept in ("CS", "Biology") else 0.15):
        course["lab_fee"] = float(rng.choice([25.0, 40.0, 60.0, 75.0]))   # SOME charge a fee
    courses.append(course)

with open("data/courses.json", "w") as f:
    json.dump(courses, f, indent=1)
print("wrote data/courses.json:", len(courses), "course documents")
```

~~~text
wrote data/courses.json: 60 course documents
~~~

## Step 1: Inspect the Raggedness

How uneven is this data, really? Count the optional fields, then put a maximal and a
minimal document side by side.

```python
print("with prerequisites:", sum("prerequisites" in c for c in courses),
      "| with lab_fee:", sum("lab_fee" in c for c in courses), "| total:", len(courses))

full = [c for c in courses if "prerequisites" in c and "lab_fee" in c][0]
mini = [c for c in courses if "prerequisites" not in c and "lab_fee" not in c][0]
print(json.dumps(full, indent=2))
print(json.dumps(mini, indent=2))
```

<details><summary>Expected Output</summary>

~~~text
with prerequisites: 30 | with lab_fee: 17 | total: 60
{
  "code": "COMP3006",
  "title": "Introduction to Programming",
  "department": "CS",
  "credits": 4,
  "schedule": {
    "days": "TTh",
    "start": "10:00"
  },
  "prerequisites": [
    "COMP3003"
  ],
  "lab_fee": 60.0
}
{
  "code": "BUSI3001",
  "title": "Topics in Accounting",
  "department": "Business",
  "credits": 3,
  "schedule": {
    "days": "MWF",
    "start": "16:00"
  }
}
~~~
*(Two documents, two shapes: one has seven fields including a list, the other five. Half
the catalog has prerequisites, under a third charges a fee. A fixed-schema table must
carry every column for every row regardless -- watch what that does in Step 2.)*
</details>

## Step 2: Approach A -- Flatten into a Table, then SQL

First, the flattening (L04's move, now on ragged data):

```python
flat = pd.json_normalize(courses)
print(flat.shape)
print(flat.head(3).to_string())
```

<details><summary>Expected Output</summary>

~~~text
(60, 8)
       code                 title department  credits  lab_fee schedule.days schedule.start prerequisites
0  COMP3000    Advanced Databases         CS        3     25.0           TTh          16:00           NaN
1  BUSI3001  Topics in Accounting   Business        3      NaN           MWF          16:00           NaN
2  MATH3002     Advanced Calculus       Math        3      NaN           MWF          15:00           NaN
~~~
*(Schema-on-write's price, visible: the nested `schedule` became dotted columns -- fine --
but every document now carries ALL eight columns, with `NaN` filling what it never had,
and the `prerequisites` lists sit whole inside single cells, where SQL can't easily reach
them. DuckDB reads pandas `NaN` as SQL `NULL`.)*
</details>

Now the shared question -- *average credits of lab-fee courses, by department*. Most of
the query is written; **uncomment and fill the two `____` blanks**, then run.

```python
con = duckdb.connect()
# Uncomment and fill the two ____ blanks, then run:
# con.sql("""
#     SELECT department,
#            ROUND(AVG(credits), 2) AS avg_credits,
#            COUNT(*)               AS n_courses
#     FROM flat
#     WHERE ____ IS NOT NULL       -- only courses that actually charge a fee
#     GROUP BY ____                -- one row per department
#     ORDER BY avg_credits DESC
# """).df()
```

<details><summary>Expected Output</summary>

~~~text
  department  avg_credits  n_courses
0   Business         4.00          1
1         CS         3.00          9
2    Biology         2.50          4
3       Math         2.33          3
~~~
*(One declarative clause per idea -- filter, group, sort: L02 unchanged. But read
`n_courses` before trusting the ranking: Business "wins" on a single course. Small groups
make loud averages -- the same honesty-about-coverage habit as L04.)*
</details>

## Step 3: Approach B -- Query the Documents in Place

Same question, no flattening. TinyDB stores the documents as-is; `Query()` asks questions
about their content -- including the shape question itself, *does this field exist?*,
which a flat table can only approximate after NULL-padding (where "absent" and "empty"
blur together). The aggregation, though, is now yours to write.

```python
dbfile = "data/catalog_db.json"
if os.path.exists(dbfile):
    os.remove(dbfile)                       # fresh database on every run
db = TinyDB(dbfile)
db.insert_multiple(courses)

Course = Query()
with_fee = db.search(Course.lab_fee.exists())          # schema-on-read superpower
print(len(with_fee), "documents have a lab_fee")

totals = {}
for c in with_fee:                                     # the GROUP BY, hand-rolled
    totals.setdefault(c["department"], []).append(c["credits"])
avgs = {dept: sum(v) / len(v) for dept, v in totals.items()}
for dept in sorted(avgs, key=avgs.get, reverse=True):  # highest average first
    print(f"{dept:10s} avg_credits={avgs[dept]:.2f}  n_courses={len(totals[dept])}")
```

<details><summary>Expected Output</summary>

~~~text
17 documents have a lab_fee
Business   avg_credits=4.00  n_courses=1
CS         avg_credits=3.00  n_courses=9
Biology    avg_credits=2.50  n_courses=4
Math       avg_credits=2.33  n_courses=3
~~~
*(Identical numbers -- the comparison is fair. But look at the code: `lab_fee.exists()`
replaced the flatten-then-NULL dance in one readable line, while SQL's one-clause GROUP BY
became four lines of bookkeeping loop. Flexible shapes in, more query work out -- the
concept note's bargain, measured.)*
</details>

## The Trade, Side by Side

| | A: flatten + DuckDB | B: TinyDB documents |
|---|---|---|
| Preparation | `json_normalize` first | none -- stored as received |
| Ragged fields | NULL-padded columns | each document keeps its shape |
| "Has this field?" | `IS NOT NULL` after padding | `Course.lab_fee.exists()` |
| Aggregation | one `GROUP BY` clause | hand-written loop |
| Lists (prerequisites) | stuck inside cells | first-class (Exercise 1) |

## Your Turn

### Exercise 1 -- Query Inside the Lists

**Task:** find the courses that require **more than one prerequisite**: print how many
there are, then the `code`, `title`, and `prerequisites` of the first three. Note this
runs on the *documents* -- the lists the flat table left stranded inside cells.
**Hint:** `Course.prerequisites.test(lambda p: len(p) > 1)` -- `.test()` runs your
function on the field's value; documents without the field simply don't match.

```python
# TODO: your query here
```

<details><summary>Expected Output</summary>

~~~text
11 courses require more than one prerequisite
  COMP3018  Introduction to Data Structures  needs ['COMP3015', 'COMP3003']
  BUSI3024  Advanced Management  needs ['BUSI3019', 'BUSI3001']
  COMP3025  Topics in Databases  needs ['COMP3006', 'COMP3012']
~~~
*(The document side answers a list question in one line. The relational answer would
need the lists pulled out into their own proper table first -- restructuring work the
document store never asked for. The reverse trade of Steps 2-3: here, flexible shape
WINS the query round.)*
</details>

### Exercise 2 -- Your Call: SQL or NoSQL?

**Task:** for each workload below, pick **relational (SQL)** or **document (NoSQL)** and
justify it in one or two sentences each, using schema-on-write/read in your answer.
Write your answer in the markdown cell.

- **Workload A:** the registrar's office stores final grades -- every record is exactly
  `(student_id, course_code, term, grade)` -- and runs GPA and enrollment statistics
  across millions of records every night.
- **Workload B:** your research group saves the raw JSON answers from four different web
  APIs (different providers, different shapes, fields appearing and disappearing as the
  providers evolve), mostly to re-read individual responses later.

*Your answer here.*

<details><summary>One Good Answer</summary>

~~~text
A: Relational. The shape is fixed and identical for every record (schema-on-write costs
nothing here), and the workload is nightly aggregations across millions of rows --
exactly what declarative GROUP BY engines are built for.

B: Document. Four evolving shapes is schema-on-read territory: storing payloads
as-received means no migration every time a provider changes, and "re-read individual
responses" is document retrieval, not aggregation -- the hand-rolled-loop weakness
never gets exercised.
~~~
</details>

## Summary

- **Schema-on-write** (tables) pays at storage time: flattening, NULL padding, stranded
  lists -- and collects at query time with one-clause declarative aggregation.
- **Schema-on-read** (documents) stores anything as-is and asks shape questions
  (`.exists()`, list lengths) directly -- and pays at query time with hand-rolled loops.
- Same data, same answers, opposite bargains: choose by the **data's shape and the
  questions you'll ask**, not by fashion.
- Check `n_courses` before trusting an average -- small groups make loud rankings.
