---
title: "Lab: Engineering Features from the Ames Homes"
unit: "IV"
lesson: "11"
type: lab
tags: [feature-engineering, one-hot-encoding, scaling, binning, date-features, housing]
difficulty: introductory
duration: "75 mins"
---

**Goal:** turn the Ames table you explored in L10 into a numeric **feature matrix** -- encode a category,
put a numeric column on a sane scale, bin a continuous column, and derive new columns from a year and a
date. Same data, a new job: building the columns Unit IV will mine.

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/devomh/cdat4001-2026/blob/main/u04_data_mining/l11_lab_feature_engineering.ipynb)

> This page is the read-only view. To run the lab, open the notebook (`l11_lab_feature_engineering.ipynb`)
> -- in Colab via the badge above, or locally. Every text output below comes from actually running the
> cells. No random numbers -- everyone sees identical results.

> **Where this sits:** opens **Unit IV** -- builds straight on L10's Ames dataset and L07's variable
> types. Here we *create* features; **L13--L14** decide which to keep, and **L15** trains a model on them.

## Prerequisites & Setup

Run the setup cells first. The data is bundled as `data/ames_subset.csv` (the cell falls back to a
one-time OpenML download if the file is absent).

```python
# Setup cell 1 of 2: install only (Colab resets on open; run this first)
%pip install -q pandas numpy scikit-learn
```

```python
# Setup cell 2 of 2: imports + load the data (bundled, with an OpenML fallback)
import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

CACHE = "data/ames_subset.csv"
COLS = ["SalePrice", "GrLivArea", "LotArea", "YearBuilt", "OverallQual",
        "BedroomAbvGr", "FullBath", "GarageArea", "Neighborhood",
        "HouseStyle", "LotFrontage", "MasVnrArea"]

if os.path.exists(CACHE):
    df = pd.read_csv(CACHE)
else:
    from sklearn.datasets import fetch_openml
    raw = fetch_openml("house_prices", version=1, as_frame=True, parser="auto")
    df = raw.frame[COLS].copy()
    os.makedirs("data", exist_ok=True)
    df.to_csv(CACHE, index=False)

print(df.shape)
```

<details><summary>Expected Output</summary>

~~~text
(1460, 12)
~~~
</details>

## Step 1: Encode a Categorical -- and Watch It Explode

An algorithm cannot read `"2Story"`. **One-hot encoding** gives each category value its own 0/1 column.
`HouseStyle` has only a handful of values, so this is clean.

```python
# 1a. One-hot encode HouseStyle (a nominal category with few values)
house = pd.get_dummies(df["HouseStyle"], prefix="style", dtype=int)
print("HouseStyle distinct values:", df["HouseStyle"].nunique())
print("one-hot shape:", house.shape)
house.head(3)
```

<details><summary>Expected Output</summary>

~~~text
HouseStyle distinct values: 8
one-hot shape: (1460, 8)
~~~
*(...followed by the first three rows of the 8 new `style_*` columns: row 0 has a 1 in `style_2Story`,
row 1 in `style_1Story`, row 2 in `style_2Story`; every other cell is 0.)*
</details>

Eight values, eight tidy columns. Now run the **same call** on `Neighborhood` and measure what happens.

```python
# 1b. The high-cardinality trap: same call on Neighborhood
nb = pd.get_dummies(df["Neighborhood"], prefix="nb", dtype=int)
print("Neighborhood distinct values:", df["Neighborhood"].nunique())
print("one-hot turns 1 column into:", nb.shape[1])
```

<details><summary>Expected Output</summary>

~~~text
Neighborhood distinct values: 25
one-hot turns 1 column into: 25
~~~
</details>

One column became 25, most of them almost entirely zeros. That is the **high-cardinality** trap: harmless
here, ruinous on an ID column with thousands of values. The fix is to group rare values into `"Other"`
before encoding.

## Step 2: Scale the Numerics

`LotArea` lives in the ten-thousands; `FullBath` is 0--3. Distance- and gradient-based algorithms would
treat `LotArea` as thousands of times more important purely because its numbers are bigger. **Scaling**
removes that accident. Standardize `GrLivArea` to mean 0, std 1:

```python
# 2a. Standardize GrLivArea by hand (z-score). ddof=0 matches StandardScaler.
gla = df["GrLivArea"]
z = (gla - gla.mean()) / gla.std(ddof=0)
print("standardized mean:", round(z.mean(), 4), "std:", round(z.std(ddof=0), 4))
print("first 3 z-scores:", [round(v, 4) for v in z.head(3)])
```

<details><summary>Expected Output</summary>

~~~text
standardized mean: -0.0 std: 1.0
first 3 z-scores: [0.3703, -0.4825, 0.515]
~~~
</details>

**Min-max** scaling is the other common recipe -- it squeezes a column into `[0, 1]` instead of centering
it. Complete the denominator (the full range) and run it:

```python
# 2b. COMPLETION -- min-max scaling: x' = (x - min) / (max - min).
# Fill in the denominator (____) with the full range, then uncomment all three lines.
# mm = (gla - gla.min()) / (____)
# print("min-max range:", round(mm.min(), 4), "to", round(mm.max(), 4))
# print("first 3 min-max:", [round(v, 4) for v in mm.head(3)])
```

<details><summary>Expected Output (after completing <code>gla.max() - gla.min()</code>)</summary>

~~~text
min-max range: 0.0 to 1.0
first 3 min-max: [0.2592, 0.1748, 0.2735]
~~~
</details>

You rarely scale by hand in practice. scikit-learn's `StandardScaler` does exactly the same z-score in one
line -- `fit` learns the mean and std, `transform` applies them:

```python
# 2c. StandardScaler reproduces 2a in one line (the fit/transform pattern you will reuse in L13-L15)
X_scaled = StandardScaler().fit_transform(df[["GrLivArea"]])
print("StandardScaler first 3:", [round(v, 4) for v in X_scaled[:3].ravel()])
```

<details><summary>Expected Output</summary>

~~~text
StandardScaler first 3: [0.3703, -0.4825, 0.515]
~~~
</details>

Identical to your hand-rolled z-scores -- because it *is* the same recipe.

## Step 3: Bin a Column, Derive From a Date

**Binning** trades a continuous column for ordered groups. Cut `YearBuilt` into decades:

```python
# 3a. Bin YearBuilt into decades (fixed-width bins with pd.cut)
decade = pd.cut(df["YearBuilt"], bins=range(1870, 2011, 10))
print(decade.value_counts().sort_index())
```

<details><summary>Expected Output</summary>

~~~text
YearBuilt
(1870, 1880]      6
(1880, 1890]      5
(1890, 1900]     14
(1900, 1910]     22
(1910, 1920]     71
(1920, 1930]     76
(1930, 1940]     63
(1940, 1950]     81
(1950, 1960]    164
(1960, 1970]    182
(1970, 1980]    174
(1980, 1990]     63
(1990, 2000]    175
(2000, 2010]    364
Name: count, dtype: int64
~~~
</details>

A **derived feature** transforms a column into something more informative. A raw year barely helps; its
*age* does:

```python
# 3b. Derive 'age' from YearBuilt (2010 = the year this data was collected)
age = 2010 - df["YearBuilt"]
print(age.describe().round(2))
```

<details><summary>Expected Output</summary>

~~~text
count    1460.00
mean       38.73
std        30.20
min         0.00
25%        10.00
50%        37.00
75%        56.00
max       138.00
Name: YearBuilt, dtype: float64
~~~
</details>

A true **timestamp** carries even more -- pull the parts out with `.dt`:

```python
# 3c. From a datetime, derive year / month / day-of-week / a weekend flag
events = pd.DataFrame({"ts": pd.to_datetime(["2021-03-15", "2021-07-04", "2021-12-25"])})
events["year"] = events["ts"].dt.year
events["month"] = events["ts"].dt.month
events["dayofweek"] = events["ts"].dt.dayofweek          # Monday=0 ... Sunday=6
events["is_weekend"] = events["ts"].dt.dayofweek >= 5
print(events.to_string(index=False))
```

<details><summary>Expected Output</summary>

~~~text
        ts  year  month  dayofweek  is_weekend
2021-03-15  2021      3          0       False
2021-07-04  2021      7          6        True
2021-12-25  2021     12          5        True
~~~
</details>

## Your Turn

**1. Should you one-hot encode `OverallQual`?** Look before you transform.

```python
# Your Turn 1: inspect OverallQual, then decide. (Uncomment to run.)
# print("OverallQual dtype:", df["OverallQual"].dtype)
# print("values:", sorted(df["OverallQual"].unique()))
# Write one sentence: encode it, or leave it as-is? Why?
```

<details><summary>Answer</summary>

`OverallQual` is `int64` with values `1`...`10` that *already encode their own order* (1 = poor, 10 =
excellent). Leave it alone -- one-hot encoding it would scatter the order across ten columns and **throw
away** the very information that makes it useful. One-hot is for *nominal* (unordered) categories.
</details>

**2. Split `GrLivArea` into four equal-count size tiers** with `pd.qcut` (not `pd.cut`), and read the
counts.

```python
# Your Turn 2: equal-count tiers with qcut. (Uncomment to run.)
# tiers = pd.qcut(df["GrLivArea"], 4)
# print(tiers.value_counts().sort_index())
```

<details><summary>Answer</summary>

~~~text
GrLivArea
(333.999, 1129.5]    365
(1129.5, 1464.0]     366
(1464.0, 1776.75]    364
(1776.75, 5642.0]    365
Name: count, dtype: int64
~~~
`qcut` draws the bin edges so each tier holds about the same number of homes (365 / 366 / 364 / 365) --
contrast Step 3a's `cut`, whose fixed-width decades ranged from 5 homes to 364. Use `qcut` when you want
balanced groups, `cut` when the edges themselves are meaningful (decades, price brackets).
</details>

**3. Legit feature or label leak?** For each candidate below, say whether it is a usable feature or a leak,
and why. (Target = `SalePrice`.)

- `price_per_sqft = SalePrice / GrLivArea`
- `rooms_per_floor` (bedrooms divided by number of stories)
- `age = 2010 - YearBuilt`

<details><summary>Answer</summary>

- `price_per_sqft` -- **leak.** It is built from `SalePrice`, the target; you could only compute it if you
  already knew the sale price. It would predict price perfectly in testing and be uncomputable for an
  unsold house.
- `rooms_per_floor` -- **fine.** Measurable for any house, sold or not; encodes layout, not the answer.
- `age` -- **fine.** Computable from `YearBuilt` alone, known long before a sale.
</details>

## Recap

You turned a readable table into the start of a feature matrix: **one-hot** for nominal categories (and
the high-cardinality trap to avoid), **standardize / min-max** to put numerics on one footing,
**`cut`/`qcut`** to bin, and **derived features** (`age`, date parts) to encode knowledge. You also met the
**fit/transform** pattern (`StandardScaler`) that the rest of Unit IV runs on. Next, **L12** takes the
hardest raw input -- free text -- and turns it into numbers too; then **L13--L14** decide *which* of all
these features are worth keeping.
