---
title: "Lab: The Data Scientist's Toolkit -- NumPy, pandas, and a First Clean"
unit: "I"
lesson: "02"
type: lab
tags: [numpy, pandas, dataframe, data-cleaning, tidy-data]
difficulty: introductory
duration: "75 mins"
---

**Goal:** put a small air-quality table into the two tools every later lesson uses --
**NumPy** for fast numbers and **pandas** for labeled tables -- and run the first **Clean**
stage of the data-science process for real. Pairs with the concept note
[The Data Scientist's Toolkit](l02_concept_data_science_toolkit.qmd).

> This page is the read-only view. To run the lab, open the notebook (`l02_lab_data_science_toolkit.ipynb`) -- in Colab via the badge below, or locally. Every output below comes from actually running the cells.
>
> [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/devomh/cdat4001-2026/blob/main/u01_introduction/l02_lab_data_science_toolkit.ipynb)

## Prerequisites & Setup

Run the setup cells first. The data is **built here, seeded**, so everyone gets identical
numbers -- there is no external download (real data acquisition is Unit II). The table is a
tiny, made-up stand-in for the air-quality question from L01.

```python
# Setup cell 1 of 2: install only (Colab resets on open; run this first)
%pip install -q numpy pandas
```

```python
# Setup cell 2 of 2: imports + build the seeded synthetic table
import numpy as np
import pandas as pd

rng = np.random.default_rng(42)              # fixed seed -> identical numbers for everyone

municipalities = ["Humacao", "Yabucoa", "Naguabo", "Maunabo"]
records = []
for town in municipalities:
    for day in range(1, 5):
        pm25 = round(max(0.0, float(rng.normal(14.0, 5.0))), 1)   # fine-particle reading
        temp = round(float(rng.normal(27.0, 1.5)), 1)             # temperature, deg C
        status = "ok" if rng.random() > 0.2 else "maintenance"
        records.append([town, day, pm25, temp, status])

aq = pd.DataFrame(records, columns=["municipality", "day", "pm25", "temp_c", "station_status"])
aq.loc[[2, 7, 11], "pm25"] = np.nan                          # three sensor gaps
aq["temp_c"] = aq["temp_c"].map(lambda c: str(c).replace(".", ","))   # temps came in with comma decimals
print("rows:", len(aq))
```

~~~text
rows: 16
~~~

## Step 1: NumPy -- Fast Numbers in Arrays

A NumPy **array** (`ndarray`) holds many numbers of one type and lets you compute on all of
them at once. Start with six PM2.5 readings:

```python
readings = np.array([12.0, 8.5, 23.1, 15.4, 9.8, 31.2])
print("shape:", readings.shape)
print("dtype:", readings.dtype)
```

~~~text
shape: (6,)
dtype: float64
~~~

`shape` `(6,)` means a 1-D array of length 6; `dtype` `float64` is the single element type.

### Naive vs. Better: averaging the readings

**Naive** -- a Python loop adds the numbers one at a time:

```python
total = 0.0
for r in readings:
    total = total + r
print("loop mean:", total / len(readings))
```

~~~text
loop mean: 16.666666666666668
~~~

**Better** -- a *vectorized* call does the whole array in one step (and runs far faster on
big arrays, because the loop happens in optimized C, not Python):

```python
print("numpy mean:", readings.mean())
```

~~~text
numpy mean: 16.666666666666668
~~~

Same answer, less code. Vectorized operations also work element-by-element. Flag every
reading above the World Health Organization 24-hour guideline of 15 ug/m3:

```python
above_guideline = readings > 15.0
print("above 15:", above_guideline)
print("count above 15:", (readings > 15.0).sum())
```

~~~text
above 15: [False False  True  True False  True]
count above 15: 3
~~~

The comparison returns a Boolean array; summing it counts the `True` values -- 3 readings
exceed the guideline.

A **matrix** is just a 2-D array (rows and columns). You can read its `shape` and index it
`[row, column]`:

```python
grid = np.array([[12.0, 8.5],
                 [23.1, 15.4]])
print("grid shape:", grid.shape)
print("grid[1, 0]:", grid[1, 0])
```

~~~text
grid shape: (2, 2)
grid[1, 0]: 23.1
~~~

> Multiplying matrices together (the dot product, matrix-vector, matrix-matrix) is its own
> topic with its own lesson -- L18, just before we need it for PCA and recommenders. Here we
> only treat arrays as containers of numbers.

## Step 2: pandas -- Labeled Tables

NumPy arrays are raw numbers with no labels. A pandas **DataFrame** is a table: named
columns, an index, and a different type allowed per column. `aq` is the table we built. Look
at the first rows:

```python
print(aq.head())
```

~~~text
  municipality  day  pm25 temp_c station_status
0      Humacao    1  15.5   25,4             ok
1      Humacao    2  18.7   24,1             ok
2      Humacao    3   NaN   26,5    maintenance
3      Humacao    4   9.7   28,3             ok
4      Yabucoa    1  14.3   28,7             ok
~~~

`.info()` is your first look at any table -- column names, how many non-null values each
has, and its dtype:

```python
aq.info()
```

~~~text
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 16 entries, 0 to 15
Data columns (total 5 columns):
 #   Column          Non-Null Count  Dtype
---  ------          --------------  -----
 0   municipality    16 non-null     object
 1   day             16 non-null     int64
 2   pm25            13 non-null     float64
 3   temp_c          16 non-null     object
 4   station_status  16 non-null     object
dtypes: float64(1), int64(1), object(3)
memory usage: 772.0+ bytes
~~~

Two warning signs already: `pm25` has only **13 non-null** of 16 (missing values), and
`temp_c` is **object** (text), not a number -- we will fix both in Step 3.

`.describe()` summarizes the numeric columns. Notice `temp_c` is absent: pandas will not
average a text column.

```python
print(aq.describe())
```

~~~text
             day       pm25
count  16.000000  13.000000
mean    2.500000  14.823077
std     1.154701   4.487047
min     1.000000   8.100000
25%     1.750000  11.900000
50%     2.500000  15.200000
75%     3.250000  17.300000
max     4.000000  24.700000
~~~

Select rows with a Boolean condition (the same idea as the NumPy comparison, now on a
table). Which readings exceed 20 ug/m3?

```python
print(aq[aq["pm25"] > 20.0][["municipality", "day", "pm25"]])
```

~~~text
   municipality  day  pm25
10      Naguabo    3  24.7
~~~

And a first "explore" step -- mean PM2.5 per municipality (pandas skips the missing values
for now):

```python
print(aq.groupby("municipality")["pm25"].mean())
```

~~~text
municipality
Humacao    14.633333
Maunabo    13.500000
Naguabo    17.466667
Yabucoa    14.133333
Name: pm25, dtype: float64
~~~

## Step 3: A First Clean

Real tables arrive messy. **Tidy** (rectangular) data means one row per observation, one
column per variable, each column a single proper type -- the shape every later tool expects.
Two problems to fix here: missing `pm25` and text `temp_c`.

First, count missing values per column:

```python
print(aq.isnull().sum())
```

~~~text
municipality      0
day               0
pm25              3
temp_c            0
station_status    0
dtype: int64
~~~

Fill the three missing `pm25` values with the column **median** (a robust middle value, less
swayed by extremes than the mean):

```python
median_pm25 = aq["pm25"].median()
aq["pm25"] = aq["pm25"].fillna(median_pm25)
print("median used:", median_pm25)
print("missing pm25 now:", int(aq["pm25"].isnull().sum()))
```

~~~text
median used: 15.2
missing pm25 now: 0
~~~

Now fix the type of `temp_c`. It is text because the temperatures came with comma decimals
("25,4"). Replace the comma with a dot, then convert to a real number:

```python
print("temp_c dtype before:", aq["temp_c"].dtype)
aq["temp_c"] = pd.to_numeric(aq["temp_c"].str.replace(",", ".", regex=False))
print("temp_c dtype after:", aq["temp_c"].dtype)
```

~~~text
temp_c dtype before: object
temp_c dtype after: float64
~~~

The table is now tidy -- no missing values and every column its proper type:

```python
print(aq.dtypes)
```

~~~text
municipality       object
day                 int64
pm25              float64
temp_c            float64
station_status     object
dtype: object
~~~

## Step 4: Completion Problem

Your turn to finish one line. Count how many readings came from each `station_status`.
Uncomment, complete the blank, and run.

```python
# Hint: a Series has a .value_counts() method.
# counts = aq["station_status"].__________()
# print(counts)
```

<details>
<summary>Expected output</summary>

~~~text
station_status
ok             13
maintenance     3
Name: count, dtype: int64
~~~

</details>

## Your Turn (Independent)

After cleaning, rank the municipalities by mean PM2.5, highest first. Compare the numbers to
the pre-cleaning means from Step 2 -- imputing the gaps moved them.

```python
# Hint: aq.groupby(...)[...].mean().sort_values(ascending=False)
# TODO: your code here
```

<details>
<summary>Expected output</summary>

~~~text
municipality
Naguabo    16.900
Humacao    14.775
Yabucoa    14.400
Maunabo    13.500
Name: pm25, dtype: float64
~~~

</details>

## Summary

| Task | Tool / call |
|------|-------------|
| Make a numeric array | `np.array([...])` |
| Array shape / type | `arr.shape`, `arr.dtype` |
| Vectorized math / compare | `arr.mean()`, `arr > 15` |
| Build / inspect a table | `pd.DataFrame(...)`, `.head()`, `.info()`, `.describe()` |
| Select / filter | `df["col"]`, `df[df["col"] > x]` |
| Group and summarize | `df.groupby("k")["v"].mean()` |
| Count missing | `df.isnull().sum()` |
| Fill missing | `df["c"].fillna(df["c"].median())` |
| Fix a column's type | `pd.to_numeric(df["c"])`, `.astype(...)` |

You loaded the PyData stack, did vectorized math in NumPy, built and explored a pandas
DataFrame, and turned a messy table into a tidy one. That tidy table is where every later
unit -- EDA, data mining, visualization -- begins.
