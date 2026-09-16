---
title: "Lab: EDA Mini-Project -- What Drives House Prices?"
unit: "III"
lesson: "10"
type: lab
tags: [eda, applied-project, data-cleaning, imputation, housing, capstone, matplotlib]
difficulty: introductory
duration: "90 mins"
---

**Goal:** run the **whole** EDA loop -- clean, explore each variable, read relationships, synthesize
findings -- on a dataset you have not seen before, and finish with **one figure worth sending to
somebody**. This is Unit III's capstone: the L07 mindset, the L08 numbers and one-variable plots, and
the L09 visual reading, applied end-to-end and ending in sentences.

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/devomh/cdat4001-2026/blob/main/u03_eda/l10_lab_eda_mini_project.ipynb)

> This page is the read-only view. To run the lab, open the notebook (`l10_lab_eda_mini_project.ipynb`)
> -- in Colab via the badge above, or locally. Every text output below comes from actually running the
> cells; the plots render live in the notebook.

> **Where this sits:** the capstone of **Unit III** -- builds on L07 (mindset), L08 (one-variable stats
> and plots), and L09 (visual methods). The single *number* for how two variables move together --
> correlation -- is **L19**; here we read relationships by eye and by group.

## Learning Objectives

By the end of this lab you will be able to:

1. **Apply** the full EDA loop -- clean, explore each variable, read relationships, synthesize -- to a
   dataset you have never seen.
2. **Handle** missing values with a strategy you can justify (median, zero, or drop) instead of a reflex.
3. **Read** each variable's shape and each pair's relationship **visually**: pair plot, scatter, grouped
   box plot, median-by-group.
4. **Construct** a figure that carries several variables at once -- color, marker size, reference lines,
   annotations -- and **save** it to a file.
5. **Synthesize** what you found into a short written data story that states its own limitations.

## The Problem

What drives house prices? Real estate is the largest financial asset most families own. In Puerto Rico,
the post-Maria market rebounded unevenly by municipality, and headline averages hid wildly different
local stories. A full EDA is exactly the tool that separates them. Today's dataset: **Ames Housing** --
1,460 home sales in Ames, Iowa (2006-2010; De Cock 2011, *Journal of Statistics Education*; via the
OpenML/Kaggle "House Prices" data, CC BY-NC-SA 4.0, educational use). We use a 12-column subset: a mix of
numeric and categorical variables, real missing values, and a clear target, `SalePrice`.

## Prerequisites & Setup

Run the setup cells first. The data is bundled as `data/ames_subset.csv` (the cell falls back to a
one-time OpenML download if the file is absent). No random numbers -- everyone sees identical results.

```python
# Setup cell 1 of 2: install only (Colab resets on open; run this first)
%pip install -q pandas matplotlib seaborn
```

```python
# Setup cell 2 of 2: imports + load the data (bundled, with an OpenML fallback)
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

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
df.head(3)
```

<details><summary>Expected Output</summary>

~~~text
(1460, 12)
~~~
*(...followed by the first three rows: two 2Story homes in CollgCr and a 1Story in
Veenker, with their features. The SalePrices are 208500, 181500, 223500.)*
</details>

### What the columns mean

A column name is not a definition. Before you compute anything, know what you are computing *on* --
the units matter as much as the numbers:

| Column | Meaning | Units |
|--------|---------|-------|
| `SalePrice` | what the house sold for -- **the target** | US dollars |
| `GrLivArea` | above-ground living area ("Gr" = ground, not gross) | square feet |
| `LotArea` | total size of the lot the house sits on | square feet |
| `LotFrontage` | length of street touching the lot | linear feet |
| `YearBuilt` | year of original construction | calendar year |
| `OverallQual` | overall rating of material and finish | 1 (very poor) to 10 (excellent) |
| `BedroomAbvGr` | bedrooms above ground (basement bedrooms excluded) | count |
| `FullBath` | full bathrooms above ground | count |
| `GarageArea` | garage size (0 means no garage) | square feet |
| `Neighborhood` | the named district of Ames the house is in | -- |
| `HouseStyle` | the dwelling's style (1Story, 2Story, SLvl, ...) | -- |
| `MasVnrArea` | masonry veneer -- a decorative brick or stone facing | square feet |

Two of these will bite us later, and the definitions are the warning: `GarageArea` uses **0 for "no
garage"** (a real value, not a missing one), and `LotFrontage` is a **measurement**, so a blank could
mean either "we never measured it" or "there is no street frontage." Keep both in mind.

## Step 1: First Look -- Structure, Quality, Headline Numbers

The L07 ritual on a dataset you have not seen: *what is here*, *what is broken*, *what are the headline
numbers*.

### 1a. Structure and types

```python
print(df.dtypes)
```

<details><summary>Expected Output</summary>

~~~text
SalePrice         int64
GrLivArea         int64
LotArea           int64
YearBuilt         int64
OverallQual       int64
BedroomAbvGr      int64
FullBath          int64
GarageArea        int64
Neighborhood     object
HouseStyle       object
LotFrontage     float64
MasVnrArea      float64
dtype: object
~~~
</details>

A dtype is not a variable type. `OverallQual` is stored as `int64` -- but is the step from quality 3 to
4 really the same size as 9 to 10? The dtype will not tell you that; only knowing what the column
*means* will (the L07 lesson). Classify each column the L07 way:

| Type | Meaning | Columns here |
|------|---------|--------------|
| Numeric -- continuous | any real value in a range | SalePrice, GrLivArea, LotArea, GarageArea, LotFrontage |
| Numeric -- discrete | whole-number counts | BedroomAbvGr, FullBath |
| Categorical -- nominal | unordered categories | HouseStyle |

That table covers eight of the twelve columns. **Uncomment and complete** the other four:

```python
# Classify each column. Choose from: Continuous / Discrete / Nominal / Ordinal
# MasVnrArea   (masonry veneer area, sq ft)  -> ____
# Neighborhood (neighbourhood name)          -> ____
# OverallQual  (quality rating 1-10)         -> ____
# YearBuilt    (year of construction)        -> ____
```

<details><summary>Expected Answers</summary>

~~~text
MasVnrArea   -> Continuous  (non-negative real; fractions of a sq ft possible)
Neighborhood -> Nominal     (named places; no ordering)
OverallQual  -> Ordinal     (1-10 ordered; gaps unequal)
YearBuilt    -> Discrete    (integer years; ordered; not a count)
~~~
</details>

### 1b. Missing values and duplicates

```python
# kind="stable": columns tied at 0 keep their table order
print(df.isnull().sum().sort_values(ascending=False, kind="stable"))
print("Duplicate rows:", df.duplicated().sum())
```

<details><summary>Expected Output</summary>

~~~text
LotFrontage     259
MasVnrArea        8
SalePrice         0
GrLivArea         0
LotArea           0
YearBuilt         0
OverallQual       0
BedroomAbvGr      0
FullBath          0
GarageArea        0
Neighborhood      0
HouseStyle        0
dtype: int64
Duplicate rows: 0
~~~
</details>

**Read it:** `LotFrontage` is missing for 259 of 1,460 houses (~18%); `MasVnrArea` for 8; no duplicate
rows. Count the damage *first* -- pandas silently skips NaNs in every statistic below (the L07 sentinel
lesson).

### 1c. Headline numbers

```python
print(df["SalePrice"].describe())
print(f"\nmean / median ratio: {df['SalePrice'].mean() / df['SalePrice'].median():.2f}")
```

<details><summary>Expected Output</summary>

~~~text
count      1460.000000
mean     180921.195890
std       79442.502883
min       34900.000000
25%      129975.000000
50%      163000.000000
75%      214000.000000
max      755000.000000
Name: SalePrice, dtype: float64

mean / median ratio: 1.11
~~~
</details>

**Read it:** the mean ($180,921) sits above the median ($163,000) -- right skew (the L08 signature: a tail
of expensive homes pulls the average up). The middle 50% of sales fall between $130k and $214k.

## Step 2: Clean -- Impute, Don't Drop

The naive fix for missing values is `dropna()`. See what it costs:

```python
print(f"Shape before cleaning: {df.shape}")
print(f"Shape after dropna():  {df.dropna().shape}")
print(f"Rows lost: {len(df) - len(df.dropna())}  ({(len(df) - len(df.dropna())) / len(df) * 100:.0f}%)")
```

<details><summary>Expected Output</summary>

~~~text
Shape before cleaning: (1460, 12)
Shape after dropna():  (1195, 12)
Rows lost: 265  (18%)
~~~
</details>

Dropping any row with a NaN throws away 18% of the data -- almost all because of `LotFrontage` alone.
Targeted imputation keeps every row:

```python
# LotFrontage: fill with the median (robust for a skewed column, the L08 rule)
# MasVnrArea:  fill with 0 -- a legitimate value (no masonry veneer on that house)
df["LotFrontage"] = df["LotFrontage"].fillna(df["LotFrontage"].median())
df["MasVnrArea"] = df["MasVnrArea"].fillna(0)
print("Missing values remaining:", df.isnull().sum().sum())
print("Shape preserved:", df.shape)
```

<details><summary>Expected Output</summary>

~~~text
Missing values remaining: 0
Shape preserved: (1460, 12)
~~~
</details>

**Why median, not mean, for LotFrontage?** It is right-skewed -- the median is the robust center (L08).
**Why 0 for MasVnrArea?** Zero is a real value (no veneer), and only 8 rows are affected; median-filling
would invent a veneer those houses do not have.

## Step 3: Univariate -- Read Each Variable

### 3a. The target, SalePrice (worked)

```python
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
sns.histplot(df["SalePrice"], bins=40, kde=True, color="steelblue", ax=axes[0])
axes[0].axvline(df["SalePrice"].mean(), color="crimson", ls="--", lw=2, label="mean")
axes[0].axvline(df["SalePrice"].median(), color="forestgreen", ls="-", lw=2, label="median")
axes[0].set_title("SalePrice is right-skewed"); axes[0].set_xlabel("SalePrice ($)"); axes[0].legend()
sns.boxplot(x=df["SalePrice"], color="lightsteelblue", ax=axes[1])
axes[1].set_title("SalePrice: long upper tail"); axes[1].set_xlabel("SalePrice ($)")
fig.tight_layout(); plt.show()
```

<details><summary>Expected Output</summary>

*(Two panels side by side. Left: a histogram with a KDE curve over it, its bulk between $130k and $214k,
a dashed crimson vertical line sitting to the right of a solid green one, and a long thin tail reaching
past $500k. Right: a horizontal box plot whose median line sits left of center, with a left whisker that
stops at about $35k, a longer right one, and a crowd of individually drawn points past its right end --
that crowd, more than the whiskers, is what the skew looks like on a box plot.)*
</details>

**Read it:** both panels tell the same story from different angles -- the histogram shows the crimson
mean line pulled to the right of the green median by the expensive tail, and the box plot shows that
tail as a long right whisker with a crowd of 1.5xIQR-flagged points beyond it. Right skew, twice.

> **Naming a color.** Every color above is a **name**: `steelblue`, `crimson`, `forestgreen`,
> `lightsteelblue`. Matplotlib accepts about 150 CSS color names, the ten `tab:` cycle colors
> (`tab:blue`, `tab:orange`, ...), and any hex string like `#c0504d`. Reach for a **name** when any
> reasonable blue will do -- it is readable, and a reader of your code knows instantly what they are
> getting. Reach for **hex** when you need one exact shade that no name gives you: a brand color, or a
> shade you must match across every figure in a report. That `#c0504d` is the one hex in this lab; it
> turns up in Step 5, where we say why it earns its place.

### 3b. OverallQual, the ordinal predictor (worked)

```python
counts = df["OverallQual"].value_counts().sort_index()
print(counts)
counts.plot(kind="bar", color="steelblue", title="Most homes rate 5, 6, or 7")
plt.xlabel("Overall Quality (1=worst, 10=best)"); plt.ylabel("count")
plt.xticks(rotation=0); plt.tight_layout(); plt.show()
```

<details><summary>Expected Output</summary>

~~~text
OverallQual
1       2
2       3
3      20
4     116
5     397
6     374
7     319
8     168
9      43
10     18
Name: count, dtype: int64
~~~
*(A bar chart centered on 5-6; very few homes earn 9 (43) or 10 (18) -- those drive the price extremes
from Step 3a.)*
</details>

### 3c. GrLivArea -- completion

`GrLivArea` is above-ground living area. Print its summary, then **uncomment and complete** a histogram
(choose a bin count between 20 and 50, add the KDE):

```python
print(df["GrLivArea"].describe())
print(f"skew = {df['GrLivArea'].skew():.2f}")

# Uncomment and fill the two ____ blanks, then run:
# use 30 bins, and kde wants True or False:
# sns.histplot(df["GrLivArea"], bins=____, kde=____, color="steelblue")
# plt.title("GrLivArea distribution"); plt.xlabel("Above-ground living area (sq ft)")
# plt.tight_layout(); plt.show()
```

<details><summary>Expected Output</summary>

~~~text
count    1460.000000
mean     1515.463699
std       525.480383
min       334.000000
25%      1129.500000
50%      1464.000000
75%      1776.750000
max      5642.000000
Name: GrLivArea, dtype: float64
skew = 1.37
~~~
*(Right-skewed, skew 1.37: mean 1,515 > median 1,464 sq ft; most homes 1,000-2,000 sq ft with a long
tail to ~5,600.)*
</details>

## Step 4: Bivariate -- Read Relationships VISUALLY

> **No correlation coefficient here, and no trend line.** The single number that measures how two
> variables move together -- **correlation (Pearson's r)** -- is **L19**, and the fitted line that goes
> with it is Unit IV. In this lesson we read relationships the L09 way: with the **eye** (scatter) and
> with **group summaries** (median per group).

The move in this step is **scan, then zoom**: look at every pair at once, pick the pair that looks most
promising, and study that one properly.

### 4a. The pair plot -- scan every pair at once (worked)

L09 gave you two ways to see many variables at once (parallel coordinates and the radar plot). Here is
the third, and the one you will reach for most: a **pair plot** (also called a scatter matrix) draws a
scatter for every pair of numeric columns, with each column's own distribution down the diagonal.

```python
sns.pairplot(df[["SalePrice", "GrLivArea", "LotArea", "YearBuilt"]],
             corner=True, diag_kind="kde",
             plot_kws={"s": 8, "alpha": 0.3, "color": "steelblue"})
plt.show()
```

<details><summary>Expected Output</summary>

*(A lower-triangle grid of panels. Down the diagonal, top to bottom: four KDE curves -- SalePrice
sharply right-skewed, GrLivArea mildly so, LotArea extremely so (one thin spike), and YearBuilt
bimodal, with a broad mid-century hump and a taller recent peak near 2000. Off the diagonal, GrLivArea
vs SalePrice is the **tightest upward band** in the grid; every LotArea panel is crushed into a thin
strip along one edge by a handful of enormous lots, with no pattern visible inside the strip; and in the
YearBuilt panels the newest homes reach the highest prices, but every era shows a wide spread.)*
</details>

**Three mechanics worth stealing:**

- **`corner=True`** drops the upper triangle. The panel for (x, y) and the panel for (y, x) show the
  same thing mirrored, so half the grid is decoration -- and the corner form renders in half the time.
- **`s=8, alpha=0.3` are not optional** at 1,460 rows. With default marker size and no transparency,
  every panel is a solid blob and you learn nothing. Small, see-through markers let density show.
- **`diag_kind="kde"`** puts L08's density curve on the diagonal, so the scan gives you the one-variable
  shapes and the pairwise relationships in a single figure.

**Read it:** only four columns and one figure, and the ranking is already visible -- living area tracks
price closely, year built tracks it loosely, lot area barely at all. That is the whole point of a scan:
it tells you **where to spend your attention next**.

### 4b. Zoom in: living area vs price, colored by quality (worked)

The pair plot said the GrLivArea-SalePrice panel was the tightest. Zoom in -- with living area on the x
axis this time, since it is the candidate driver -- and add a third variable as color:

```python
fig, ax = plt.subplots(figsize=(9, 5))
sc = ax.scatter(df["GrLivArea"], df["SalePrice"], c=df["OverallQual"], cmap="RdYlGn", alpha=0.5, s=15)
fig.colorbar(sc, ax=ax, label="OverallQual")
ax.set_xlabel("Above-ground living area (sq ft)"); ax.set_ylabel("SalePrice ($)")
ax.set_title("Living area vs price, colored by quality"); plt.tight_layout(); plt.show()
```

<details><summary>Expected Output</summary>

*(A scatter of 1,460 semi-transparent points rising from the bottom-left and widening as it climbs, with
a colorbar down the right labelled OverallQual, running red through yellow to green. At any given living
area, the green points sit near the top of the cloud and the red ones near its bottom.)*
</details>

**Read it:** the trend is positive and roughly linear -- bigger homes cost more -- but the cloud **fans
out** as it rises: among homes near 2,000 sq ft, prices run from about $125k to $440k. Size alone does
not fix the price. The color is what explains the spread: green (high quality) dots ride the top of the
band and red (low quality) the bottom, so the vertical scatter at any given size is largely a quality
story. All of that read by eye -- no number computed.

### 4c. Price by quality -- the group-summary story (worked)

The scatter hinted that quality matters. Make it explicit with a **median per group** (no coefficient):

```python
med_by_qual = df.groupby("OverallQual")["SalePrice"].median()
print(med_by_qual)
med_by_qual.plot(kind="bar", color="steelblue", title="Median SalePrice rises with quality")
plt.xlabel("Overall Quality"); plt.ylabel("median SalePrice ($)")
plt.xticks(rotation=0); plt.tight_layout(); plt.show()
```

<details><summary>Expected Output</summary>

~~~text
OverallQual
1      50150.0
2      60000.0
3      86250.0
4     108000.0
5     133000.0
6     160000.0
7     200141.0
8     269750.0
9     345000.0
10    432390.0
Name: SalePrice, dtype: float64
~~~
*(Median price climbs monotonically with quality: a quality-9 home's median ($345,000) is about **2.6x**
a quality-5 home's ($133,000). "Quality drives price" -- shown with group medians, not a correlation.)*
</details>

### 4d. Price by house style -- grouped box plot (worked)

A **grouped box plot** is just L08's box plot drawn once **per category**, side by side -- so you compare
a numeric variable's distribution across groups at a glance:

```python
order = df.groupby("HouseStyle")["SalePrice"].median().sort_values(ascending=False).index
sns.boxplot(data=df, x="HouseStyle", y="SalePrice", hue="HouseStyle",
            order=order, palette="Blues_r", legend=False)
plt.title("SalePrice by house style (ordered by median)")
plt.xlabel("House style"); plt.ylabel("SalePrice ($)")
plt.xticks(rotation=15); plt.tight_layout(); plt.show()
```

<details><summary>Expected Output</summary>

*(Eight boxes ordered by median: 2.5Fin (~$194k) and 2Story (~$190k) highest -- but 2.5Fin rests on only
8 homes, so trust the widest, best-sampled boxes most -- and 1.5Unf (~$111k) lowest. 1Story dominates by
count (726) and shows a wide interquartile range: lots of price variability even within one style.)*
</details>

## Step 5: One Figure, Many Encodings -- and How to Save It

Every figure so far carried one or two variables. A finished figure -- the one that goes in the report --
usually carries more, and says what it means **on the figure itself**. Here we build one plot with
**four** variables and three kinds of annotation, then save it to a file.

The encodings: living area on **x**, price on **y**, quality as **color** (two tiers), garage size as
**marker size**. First, the split:

```python
premium = df["OverallQual"] >= 8      # a boolean mask -- True/False per row
print(f"premium (quality 8-10): {premium.sum()} homes")
print(f"the rest  (quality 1-7): {(~premium).sum()} homes")
```

<details><summary>Expected Output</summary>

~~~text
premium (quality 8-10): 229 homes
the rest  (quality 1-7): 1231 homes
~~~
</details>

You have filtered rows with a single condition since L02 (`df[df["pm25"] > 20]`). Three small extensions
carry the rest of this step: **`~mask`** flips a mask (so `~premium` is "everything else"), **`&`**
combines two masks and each condition needs **its own parentheses** (`(a > 1) & (b < 2)` -- without them
Python's operator precedence raises an error), and **`df.loc[mask, "col"]`** pulls one column for just
the rows the mask selects.

The two houses we are about to point at -- find them first, do not eyeball them:

```python
odd = df[(df["GrLivArea"] > 4000) & (df["SalePrice"] < 300000)]
print(odd[["GrLivArea", "SalePrice", "OverallQual", "Neighborhood"]])
```

<details><summary>Expected Output</summary>

~~~text
      GrLivArea  SalePrice  OverallQual Neighborhood
523        4676     184750           10      Edwards
1298       5642     160000           10      Edwards
~~~
</details>

Two of the largest houses in the dataset, both rated quality **10**, both in Edwards, and both sold for
**less than the median-quality home of half the size**. That is not a market insight -- that is a data
problem worth flagging, and the figure should say so out loud.

```python
fig, ax = plt.subplots(figsize=(10, 6))

# marker size carries GarageArea; the +10 floor keeps the 81 garage-less homes visible
sizes = 10 + df["GarageArea"] / 15

ax.scatter(df.loc[~premium, "GrLivArea"], df.loc[~premium, "SalePrice"],
           s=sizes[~premium], color="lightsteelblue", alpha=0.6,
           edgecolor="white", linewidth=0.4, label="quality 1-7")
ax.scatter(df.loc[premium, "GrLivArea"], df.loc[premium, "SalePrice"],
           s=sizes[premium], color="#c0504d", alpha=0.75,
           edgecolor="white", linewidth=0.4, label="quality 8-10")

# reference lines at the two medians, splitting the plot into four regions
ax.axhline(df["SalePrice"].median(), color="gray", ls=":", lw=1)
ax.axvline(df["GrLivArea"].median(), color="gray", ls=":", lw=1)
ax.text(600, 330000, "small, expensive", color="gray", fontsize=9)
ax.text(4300, 600000, "big, expensive", color="gray", fontsize=9)
ax.text(600, 60000, "small, cheap", color="gray", fontsize=9)
ax.text(4850, 60000, "big, cheap", color="gray", fontsize=9)

# one note, two arrows: write the text once, then add a text-less annotation for the second point
note_at = (2900, 38000)      # where the note's text sits
# the 2nd arrow starts to the RIGHT of that text, not inside it
tail_2 = (3980, 46000)
arrow = dict(arrowstyle="->", color="black", lw=1)
ax.annotate("top quality, huge, sold cheap\n-- check these records",
            xy=(odd.iloc[0]["GrLivArea"], odd.iloc[0]["SalePrice"]),
            xytext=note_at, arrowprops=arrow, fontsize=9)
ax.annotate("", xy=(odd.iloc[1]["GrLivArea"], odd.iloc[1]["SalePrice"]),
            xytext=tail_2, arrowprops=arrow)

ax.set_xlabel("Above-ground living area (sq ft)")
ax.set_ylabel("SalePrice ($)")
ax.set_title("Bigger and better-built sells for more -- with two glaring exceptions\n"
             "(marker size = garage area; dotted lines = the two medians)")
ax.legend(title="Overall quality", loc="upper left")

# save BEFORE plt.show() -- show() clears the figure and you would save a blank file
fig.savefig("ames_price_vs_area.png", dpi=150, bbox_inches="tight")
plt.show()
print("saved:", os.path.exists("ames_price_vs_area.png"))
```

<details><summary>Expected Output</summary>

~~~text
saved: True
~~~
*(One scatter carrying four variables: the pale blue quality-1-7 cloud rises from bottom-left, the brick-
red quality-8-10 points sit above and to the right of it, and marker sizes vary visibly -- the biggest
dots are the biggest garages. Dotted gray lines cross at (1,464 sq ft, $163,000), labelling the four
regions, and a single note low on the right fans two black arrows out to the Edwards pair, which sit far
below every other house their size.)*
</details>

**Read it:** the two encodings agree -- moving right (bigger) and moving from pale blue to brick red
(better built) both move you up the price axis, and the "big, expensive" region is almost entirely brick
red. The figure also *earns its annotations*: without the arrows, the Edwards pair are just two dots; with
them, the reader knows the analyst saw them and had a question.

**The one hex in this lab, and why.** Every other color here is a name, but the premium points are
`#c0504d`. This is the case Step 3a described: the shade is fixed by something outside the plot -- it is
the brick red used for "premium" in every other figure of the report this plot belongs to, so it has to
match exactly. The nearest CSS name, `indianred` (`#cd5c5c`), is noticeably lighter and would make this
figure the odd one out. When an exact shade matters, hex is the honest tool; when it does not, a name is
the readable one.

**Marker size is a rough channel -- do not overload it.** `s` sets the marker's **area**, so doubling
`s` does *not* mean "twice as much garage" to the eye. Size is for a supporting variable a reader should
notice, never for the variable your conclusion depends on -- that one belongs on an axis. And the `+ 10`
floor is not cosmetic: 81 houses have `GarageArea == 0`, and without the floor they would plot at size
zero and vanish from the figure entirely.

**Four things about `savefig`:**

- **Call it BEFORE `plt.show()`.** `plt.show()` renders and then clears the figure; saving afterwards
  writes a blank image. This is the single most common way to get an empty PNG.
- **`dpi=150`** sets the resolution of a raster image. 100 is screen-ish, 150 is comfortable, 300 is
  print.
- **`bbox_inches="tight"`** trims the whitespace and, more importantly, stops long axis labels and
  titles from being cropped off the saved file (they often look fine on screen and get cut in the file).
- **The extension chooses the format.** `.png` for slides and the web, `.pdf` or `.svg` for a report or
  a poster -- those are **vector**, so they stay sharp at any zoom.

> **In Colab** the file lands in the session's own filesystem -- open the folder icon in the left sidebar
> to see it, or pull it to your machine with `from google.colab import files;
> files.download("ames_price_vs_area.png")`. The session filesystem is wiped when the runtime closes, so
> download anything you want to keep.

## Step 6: Synthesis -- Your Findings

EDA ends in **sentences**, not just plots. **Uncomment and fill** three findings (one sentence each):

```python
# Uncomment and fill the three strings, then run:
# finding_1 = "____"   # the strongest driver of SalePrice, and why (hint: Steps 4b-4c)
# finding_2 = "____"   # one surprising or counter-intuitive result
# finding_3 = "____"   # one limitation of this analysis or dataset
# print("=== EDA Findings: Ames Housing ===")
# print(f"1. Strongest driver: {finding_1}")
# print(f"2. Surprising:       {finding_2}")
# print(f"3. A limitation:     {finding_3}")
```

<details><summary>Sample Answers</summary>

~~~text
=== EDA Findings: Ames Housing ===
1. Strongest driver: OverallQual -- median price climbs steadily with it, from $133k at 5 to $345k at 9.
2. Surprising:       Two quality-10 Edwards houses over 4,000 sq ft sold below $185k -- likely not
                     ordinary sales, and they would distort any average they enter.
3. A limitation:     One Midwestern city, 2006-2010 (a crash window); patterns may not transfer to PR.
~~~
</details>

## Your Turn

Seven exercises in four tiers. **A and B** read what is already on the screen, **C** asks you to build a
figure to a specification, and **D** is open-ended -- if you run short of time in the session, tiers A-C
are the core and **tier D is the one to finish at home**.

### Tier A -- Read the numbers

#### Exercise A1 -- Which neighborhood ranking do you trust?

Compute the **median `SalePrice` per neighborhood together with the number of sales behind it**, sort by
median, and look at the top six. Then answer in two sentences: which neighborhood would you name as
"the most expensive," and which of those six rankings would you trust **least**, and why?

```python
# TODO: your code here (one groupby with two aggregations, sorted), then two sentences as a comment
```

<details><summary>Expected Output</summary>

~~~text
                median  size
Neighborhood                
NridgHt       315000.0    77
NoRidge       301500.0    41
StoneBr       278000.0    25
Timber        228475.0    38
Somerst       225500.0    86
Veenker       218000.0    11
~~~
*(NridgHt tops the table at $315,000 and it is also well sampled (77 sales), so that ranking is
believable. The rows below it are not equally trustworthy: StoneBr's $278,000 rests on 25 sales and
Veenker's $218,000 on 11, so either could move a lot with a handful of different houses -- the same
small-n caution as the 2.5Fin box in Step 4d. A median is only as stable as the count behind it, which
is why you print `size` next to it.)*
</details>

#### Exercise A2 -- A mean below its median

Every skewed column so far has had its **mean above its median**. Print `GarageArea`'s mean, its median,
and the number of houses with `GarageArea == 0`. The mean comes out *below* the median here -- explain
the mechanism in two sentences, and say whether those zeros are outliers to remove or a category to keep.

```python
# TODO: your code here (three prints), then two sentences as a comment
```

<details><summary>Expected Output</summary>

~~~text
mean:   473.0
median: 480.0
houses with GarageArea == 0: 81
~~~
*(81 houses have no garage at all, and a pile of exact zeros sits far below the bulk of the data -- a
left-hand tail. The mean is dragged toward it while the median barely moves, which is the same
mean-vs-median mechanism as SalePrice running in the opposite direction. Those zeros are **not**
outliers to delete: "no garage" is a real, meaningful category of house, exactly like the `MasVnrArea`
zeros in Step 2. The honest fix is to report the two groups separately -- "81 houses have no garage;
among the rest the median is larger" -- not to erase them.)*
</details>

### Tier B -- Read the plots

#### Exercise B1 -- What the pair plot told you

Go back to the Step 4a pair plot. In two or three sentences and **no numbers at all**: which pair shows
the tightest band, which panel shows almost no usable relationship, and what would you have missed if
you had jumped straight to a single scatter of your favourite variable?

<details><summary>Expected Answer</summary>

~~~text
GrLivArea vs SalePrice is the tightest band -- the cloud rises clearly and stays reasonably narrow.
The LotArea panels are the weakest: a few enormous lots crush every other house into a thin strip along
one edge, and inside that strip price shows no visible pattern. The scan is what tells you the ranking;
had you gone straight to a LotArea scatter you would have spent your time on the one panel with the
least to say, and never known that a better variable was sitting next to it.
~~~
</details>

#### Exercise B2 -- LotArea, two ways

First the numbers: print `describe()` and `skew()` for `LotArea`. Then the picture: plot `LotArea` (x)
against `SalePrice` (y) and read it the L09 way -- direction, form, strength, unusual points -- in two
sentences. (No correlation number and no fitted line; both are later lessons.)

```python
# TODO: your code here (describe + skew, then one scatter), then two sentences as a comment
```

<details><summary>Expected Output</summary>

~~~text
count      1460.000000
mean      10516.828082
std        9981.264932
min        1300.000000
25%        7553.500000
50%        9478.500000
75%       11601.500000
max      215245.000000
Name: LotArea, dtype: float64
skew = 12.21
~~~
*(A skew of 12.21 is extreme -- compare GrLivArea's 1.37. The max lot (215,245 sq ft, roughly five acres)
is more than twenty times the median (9,478.5), so the mean sits well above the median and a handful of
rural parcels stretch the axis. The scatter is correspondingly **wide and noisy**: a weak, positive,
loosely linear trend among ordinary lots, crowded into the left edge. The clearest reading is what is
*missing* from the top-right -- the most expensive homes in Ames do not sit on the biggest lots at all
(the two priciest, at $755,000 and $745,000, sit on fairly ordinary 21,535 and 15,623 sq ft lots), while
the extreme lots land only mid-range in price and two of them sold for $160,000. A bigger lot does not
reliably mean a higher price.)*
</details>

### Tier C -- Reproduce a figure from its specification

#### Exercise C1 -- Build this figure

Below is the figure you have to produce. Work only from this specification -- do not copy code from
above:

- a **horizontal** bar chart (`ax.barh`) of **median `SalePrice`**,
- for the **8 neighborhoods with the most sales** (most sales, not highest price),
- sorted so the **cheapest bar is at the bottom** and the most expensive at the top,
- bars in `steelblue`,
- the **overall median SalePrice** drawn as a **dashed gray vertical line**,
- title `Where the expensive houses are`, x-label `median SalePrice ($)`, `figsize=(8, 5)`,
- and finally **save it** as `neighborhood_medians.png` at `dpi=150` with `bbox_inches="tight"`.

![Target figure: a horizontal bar chart titled "Where the expensive houses are", with eight neighborhood bars in steelblue ordered from OldTown at the bottom (about $119,000) up to NridgHt at the top (about $315,000), and a dashed gray vertical line at the overall median of $163,000 crossing the four most expensive bars (NridgHt down to Gilbert), while the four cheapest stop short of it.](https://raw.githubusercontent.com/devomh/cdat4001-2026/main/u03_eda/assets/l10_reproduce_target.png)

```python
# TODO: your code here
```

<details><summary>Expected Output</summary>

~~~python
top8 = df["Neighborhood"].value_counts().head(8).index
med8 = df[df["Neighborhood"].isin(top8)].groupby("Neighborhood")["SalePrice"].median().sort_values()

fig, ax = plt.subplots(figsize=(8, 5))
ax.barh(med8.index, med8.values, color="steelblue")
ax.axvline(df["SalePrice"].median(), color="gray", ls="--", lw=1.5)
ax.set_title("Where the expensive houses are")
ax.set_xlabel("median SalePrice ($)")
fig.tight_layout()
# save before show(), as in Step 5
fig.savefig("neighborhood_medians.png", dpi=150, bbox_inches="tight")
plt.show()
~~~
*(Three moves carry this: `value_counts().head(8).index` picks the neighborhoods by **count**, `isin`
filters to them, and `.sort_values()` on the medians is what puts the bars in order -- `barh` draws the
first row at the bottom, so sorting ascending gives the cheapest-at-the-bottom layout. The bars run
OldTown $119,000, Edwards $121,750, Sawyer $135,000, NAmes $140,000, Gilbert $181,000, CollgCr $197,200,
Somerst $225,500, NridgHt $315,000, and the dashed line at $163,000 falls between NAmes and Gilbert --
so half of the eight best-sampled neighborhoods sit below the town-wide median.)*
</details>

### Tier D -- Open-ended (finish at home)

#### Exercise D1 -- Written: imputation assumptions

`LotFrontage` was filled with the median. Under what missingness assumption is that reasonable? If the
blanks instead meant "this lot genuinely has no street frontage," why would median-filling mislead, and
what would you do instead? *(Hint: a value can be missing at random, or missing for a reason. Filling 69
ft for a lot with truly zero frontage is not neutral -- and Exercise A2 just showed you what a pile of
real zeros does to a mean.)*

#### Exercise D2 -- Explore a column we skipped

Pick one numeric column we did not feature (`BedroomAbvGr`, `FullBath`, `YearBuilt`, `MasVnrArea`, ...).
Make one plot -- a histogram, or a scatter against `SalePrice` -- and write two sentences on what you
see: shape, or the visual relationship to price. Then say which of Step 5's encodings (color or size)
you would have used to add it to the big figure, and why.

```python
# TODO: your code here
```

## Summary

| Step | Key commands | What you did |
|------|--------------|--------------|
| First look | `dtypes`, `isnull().sum()`, `describe()` | structure, quality, headline numbers |
| Clean | `fillna(median())`, `fillna(0)` | targeted impute beats `dropna()` (kept 265 rows) |
| Univariate | `histplot`, `boxplot`, `value_counts` + bar | shape, spread, skew per variable |
| Scan | `pairplot(corner=True, diag_kind="kde")` | every pair at once -- where to spend attention next |
| Bivariate (visual) | `scatter`, `groupby().median()` + bar, grouped `boxplot` | relationships by eye and by group -- no `r` |
| Finished figure | `scatter(c=, s=)`, `axhline`/`axvline`, `text`, `annotate` | four variables and their meaning in one plot |
| Save it | `savefig(dpi=, bbox_inches="tight")` **before** `show()` | a file you can put in a report |
| Synthesis | written findings | EDA ends in sentences, with stated limits |

You ran the full Unit III loop on an unfamiliar dataset and produced a figure you could hand to someone.
**Unit IV** turns these patterns into *features* and *models* -- and **L19** finally puts a number
(correlation) on the relationships you read by eye here.
