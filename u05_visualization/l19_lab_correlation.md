---
title: "Lab: Penguin Relationships & a Paradox"
unit: "V"
lesson: "19"
type: lab
tags: [correlation, pearson, spearman, prediction, simpsons-paradox, autocorrelation]
difficulty: introductory
duration: "75 mins"
---

**Goal:** quantify a relationship in the penguins data three ways -- Pearson's r, the
same r built from L18's dot product (and read as an angle), and a best-fit line that
predicts -- then read the whole correlation matrix and catch a real Simpson's paradox in
the act. The exercises carry the same tools to two new datasets: Anscombe's quartet and
L17's San Juan temperatures. Pairs with the concept note
[Relationships Between Two Variables: Correlation](l19_concept_correlation.qmd).

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/devomh/cdat4001-2026/blob/main/u05_visualization/l19_lab_correlation.ipynb)

> This page is the read-only view. To run the lab, open the notebook
> (`l19_lab_correlation.ipynb`) -- in Colab via the badge above, or locally. Every text
> output below comes from actually running the cells.

## Prerequisites & Setup

Two datasets load here. **Palmer Penguins** (Gorman, Williams & Fraser 2014, *PLoS ONE*;
CC0 public domain), 344 penguins -- the same dataset you explored in Unit III. And
**Anscombe's quartet** (Anscombe 1973, *The American Statistician*), the 44 points from the
concept note's opening figure, for Exercise 2. Exercise 4 loads its own data. No random
numbers in this lab, so there is nothing to seed.

```python
# Setup cell 1 of 2: install only (Colab resets on open; run this first)
%pip install -q pandas numpy matplotlib seaborn
```

```python
# Setup cell 2 of 2: imports + load the data (bundled, with a live fallback)
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

SEABORN_DATA = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/"

# Palmer Penguins -- Gorman, Williams & Fraser (2014), PLoS ONE; CC0 public domain
LOCAL = "data/penguins.csv"
penguins = pd.read_csv(LOCAL if os.path.exists(LOCAL)
                       else SEABORN_DATA + "penguins.csv")

# Anscombe's quartet -- Anscombe (1973), The American Statistician 27(1)
ANSCOMBE_LOCAL = "data/anscombe.csv"
anscombe = pd.read_csv(ANSCOMBE_LOCAL if os.path.exists(ANSCOMBE_LOCAL)
                       else SEABORN_DATA + "anscombe.csv")

print("penguins:", penguins.shape)
print("anscombe:", anscombe.shape)
```

<details><summary>Expected Output</summary>

~~~text
penguins: (344, 7)
anscombe: (44, 3)
~~~
</details>

## Know the Columns

Correlation compares measurements, so it pays to know exactly what each one measures.
The penguins were measured on three islands of the Palmer Archipelago, Antarctica.

| Column | What it records | Values |
|--------|-----------------|--------|
| `species` | Adelie, Chinstrap or Gentoo | 152 / 68 / 124 birds |
| `island` | Biscoe, Dream or Torgersen | category |
| `bill_length_mm` | bill length, from the tip to where it meets the head | mm |
| `bill_depth_mm` | bill **height**, top to bottom, at its base | mm |
| `flipper_length_mm` | flipper (wing) length | mm |
| `body_mass_g` | body mass | grams |
| `sex` | MALE or FEMALE | some missing |

"Depth" is the word that trips people up: it is how *tall* the bill is, not how long.
A deep bill is a thick, chunky one. The ranges and the missing values:

```python
print(penguins.describe().loc[["min", "max"]].to_string())
print()
print(penguins.isna().sum().to_string())
```

<details><summary>Expected Output</summary>

~~~text
     bill_length_mm  bill_depth_mm  flipper_length_mm  body_mass_g
min            32.1           13.1              172.0       2700.0
max            59.6           21.5              231.0       6300.0

species               0
island                0
bill_length_mm        2
bill_depth_mm         2
flipper_length_mm     2
body_mass_g           2
sex                  11
~~~
</details>

> **Read it:** every measurement is a positive number -- remember that for Step 2. Two
> birds have no measurements at all (the same two in every column), and pandas' `corr`
> skips them automatically. The 11 missing `sex` values do not matter here: this lab
> never uses `sex`.

## Step 1: One Relationship -- Scatter + r (Worked)

Flipper length vs body mass: the scatter is the L09 read (direction, form, strength,
unusual points), then the number.

```python
sns.scatterplot(data=penguins, x="flipper_length_mm", y="body_mass_g")
plt.title("Flipper length vs body mass")
plt.show()

r = penguins["flipper_length_mm"].corr(penguins["body_mass_g"])
print(f"Pearson r = {r:+.2f}")
```

<details><summary>Expected Output</summary>

~~~text
Pearson r = +0.87
~~~
</details>

> **Read it:** direction positive (longer flippers, heavier birds), form roughly linear,
> strength high -- the cloud is tight, no isolated stragglers. The number agrees: +0.87,
> strong positive. (pandas quietly skipped the rows with missing values.)

A cheap insurance check the concept note insisted on -- color the same scatter by species,
to make sure no subgroup is hiding:

```python
sns.scatterplot(data=penguins, x="flipper_length_mm", y="body_mass_g",
                hue="species", alpha=0.8)
plt.title("Same scatter, colored by species")
plt.show()
```

> **Read it:** three clouds -- Gentoo up-right (bigger on both axes), Adelie and Chinstrap
> overlapping lower-left. The positive trend points the same way *inside* each cloud, so
> here the pooled number is honest. Remember this check; in Step 4 it saves us from a wrong
> conclusion.

## Step 2: The Same r, Built from L18 (Worked)

The concept note's claim: Pearson's r is L18's **cosine of the two mean-centered columns**,
and covariance is their **dot product**. Build it by hand and check it against pandas.

```python
pair = penguins[["flipper_length_mm", "body_mass_g"]].dropna()
x = pair["flipper_length_mm"].to_numpy()
y = pair["body_mass_g"].to_numpy()

xc = x - x.mean()            # center each column (subtract its mean)
yc = y - y.mean()
covariance = (xc @ yc) / (len(x) - 1)                       # L18 dot product, averaged
r_cosine = (xc @ yc) / (np.linalg.norm(xc) * np.linalg.norm(yc))   # L18 cosine

print(f"rows after dropna: {len(x)}")
print(f"covariance (xc @ yc / n-1): {covariance:,.1f}")
print(f"r via L18 cosine:  {r_cosine:+.6f}")
print(f"r via pandas corr: {pair['flipper_length_mm'].corr(pair['body_mass_g']):+.6f}")
print(f"match: {np.isclose(r_cosine, pair['flipper_length_mm'].corr(pair['body_mass_g']))}")
```

<details><summary>Expected Output</summary>

~~~text
rows after dropna: 342
covariance (xc @ yc / n-1): 9,824.4
r via L18 cosine:  +0.871202
r via pandas corr: +0.871202
match: True
~~~
</details>

> **Read it:** identical to six decimals. Correlation is not new machinery -- it is L18's
> dot product (covariance) turned into L18's cosine (r) by dividing out the two vectors'
> lengths.

**Why center first?** In L18, Ana and Carla's raw cosine was +0.52 although their tastes
are opposite: all-positive ratings live in the *positive corner*, where no two arrows can
be more than 90 degrees apart. Penguin measurements are all positive too. Compare the raw
cosine with r for two pairs, and turn each r into an angle with `np.arccos`:

```python
def cosine(a, b):
    return a @ b / (np.linalg.norm(a) * np.linalg.norm(b))

for col in ["flipper_length_mm", "bill_depth_mm"]:
    both = penguins[[col, "body_mass_g"]].dropna()
    a = both[col].to_numpy()
    b = both["body_mass_g"].to_numpy()
    cos_raw = cosine(a, b)
    r_c = cosine(a - a.mean(), b - b.mean())
    angle = np.degrees(np.arccos(r_c))
    print(f"{col} vs body_mass_g")
    print(f"  raw cosine {cos_raw:+.3f}, r {r_c:+.3f}, "
          f"angle {angle:.1f} degrees")
```

<details><summary>Expected Output</summary>

~~~text
flipper_length_mm vs body_mass_g
  raw cosine +0.991, r +0.871, angle 29.4 degrees
bill_depth_mm vs body_mass_g
  raw cosine +0.966, r -0.472, angle 118.2 degrees
~~~
</details>

> **Read it:** the raw cosines are both near 1 -- they only say "both columns are positive
> numbers". After centering, the two pairs separate: flipper and mass are 29.4 degrees
> apart (a tight, rising cloud), while bill depth and mass are 118.2 degrees apart --
> **over 90, so r is negative**, a sign the raw cosine could never show. Centering is not a
> detail; it is what makes r measure a relationship. Keep that -0.472 in mind: Step 4
> investigates it.

## Step 3: Correlation as a Predictive Tool (Worked + Completion)

If two variables correlate, a straight line can **predict** one from the other.
`np.polyfit(x, y, 1)` returns the best line's slope and intercept:

```python
slope, intercept = np.polyfit(x, y, 1)
print(f"slope = {slope:.1f} g per mm, intercept = {intercept:.1f} g")
print(f"predicted mass at flipper=210mm: {slope*210 + intercept:,.0f} g")
print(f"r-squared = {r**2:.3f}")

sns.scatterplot(data=penguins, x="flipper_length_mm", y="body_mass_g", alpha=0.5)
xs = np.array([penguins['flipper_length_mm'].min(), penguins['flipper_length_mm'].max()])
plt.plot(xs, slope*xs + intercept, color="crimson", linewidth=2, label="best-fit line")
plt.legend(); plt.title("Flipper length predicts body mass")
plt.show()
```

<details><summary>Expected Output</summary>

~~~text
slope = 49.7 g per mm, intercept = -5780.8 g
predicted mass at flipper=210mm: 4,653 g
r-squared = 0.759
~~~
</details>

> **Read it:** the line predicts about 50 g of body mass per extra millimetre of flipper,
> and **r-squared = 0.76** says it accounts for about three-quarters of the variation in
> mass -- a strong predictor, with a quarter left to everything else. A higher |r| would
> mean a tighter cloud and a better line.

Your turn -- predict the mass of a penguin with a 200 mm flipper.

```python
# Uncomment and fill the ____ : reuse slope and intercept from above.
# print(f"predicted mass at flipper=200mm: {slope*____ + intercept:,.0f} g")
```

<details><summary>Expected Output</summary>

~~~text
predicted mass at flipper=200mm: 4,156 g
~~~
*(200 mm is inside the observed flipper range, so the line is trustworthy here. Predict at
300 mm and you would have hope, not evidence -- never extrapolate the fit past the data.)*
</details>

## Step 4: All Pairs, and a Paradox (Worked + Completion)

Six pairs of measurements is too many scatters to eyeball. Compute the full matrix:

```python
numeric = penguins.select_dtypes("number")
print(numeric.corr().round(2).to_string())
```

<details><summary>Expected Output</summary>

~~~text
                   bill_length_mm  bill_depth_mm  flipper_length_mm  body_mass_g
bill_length_mm               1.00          -0.24               0.66         0.60
bill_depth_mm               -0.24           1.00              -0.58        -0.47
flipper_length_mm            0.66          -0.58               1.00         0.87
body_mass_g                  0.60          -0.47               0.87         1.00
~~~
</details>

A heatmap makes the matrix scannable. **Uncomment and fill the two ____ blanks**: the
first is the matrix, the second turns on the in-cell numbers (True/False). Note the
colormap -- `coolwarm` is *diverging* because r has a meaningful midpoint at 0 (the L16
color principle).

```python
# Uncomment and fill the two ____ blanks, then run:
# sns.heatmap(____, annot=____, cmap="coolwarm", vmin=-1, vmax=1)
# plt.title("Pearson correlation, penguin measurements")
# plt.show()
```

<details><summary>Expected Output</summary>

*(A 4x4 heatmap: deep-red diagonal of 1.00, warm red 0.87 for flipper/body-mass, and the
bill-depth row/column in blues -- including -0.47 against body mass. Cool colors where
everything "should" grow together is the anomaly worth chasing.)*
</details>

The bill-depth row is negative against everything, including **-0.47** with body mass.
Deeper bills on *lighter* penguins? Check it the way Step 1 taught -- overall, then within
each species:

```python
pair_bd = ["bill_depth_mm", "body_mass_g"]
print(f"overall    r = {penguins[pair_bd[0]].corr(penguins[pair_bd[1]]):+.2f}")
for species, group in penguins.groupby("species"):
    print(f"{species:10s} r = {group[pair_bd[0]].corr(group[pair_bd[1]]):+.2f}")
```

<details><summary>Expected Output</summary>

~~~text
overall    r = -0.47
Adelie     r = +0.58
Chinstrap  r = +0.60
Gentoo     r = +0.72
~~~
*(Pooled it is moderately NEGATIVE; inside every species it is POSITIVE. Both correct, and
they contradict each other -- Simpson's paradox in the wild.)*
</details>

Now **see** it: the species-colored scatter, one fitted line per species (solid), and one
line fitted to all three pooled (dashed black).

```python
bd = penguins[["species", "bill_depth_mm", "body_mass_g"]].dropna()
colors = sns.color_palette()
sns.scatterplot(data=bd, x="bill_depth_mm", y="body_mass_g",
                hue="species", alpha=0.6)

# One line per species (groupby order = the hue order, so colors match)
for i, (species, group) in enumerate(bd.groupby("species")):
    gx, gy = group["bill_depth_mm"], group["body_mass_g"]
    s, b = np.polyfit(gx, gy, 1)
    ends = np.array([gx.min(), gx.max()])
    plt.plot(ends, s * ends + b, color=colors[i], linewidth=2.5)
    print(f"{species:10s} slope {s:+.0f} g per mm")

# One line for all three species pooled
s, b = np.polyfit(bd["bill_depth_mm"], bd["body_mass_g"], 1)
ends = np.array([bd["bill_depth_mm"].min(), bd["bill_depth_mm"].max()])
plt.plot(ends, s * ends + b, color="black", linestyle="--", linewidth=2)
print(f"{'pooled':10s} slope {s:+.0f} g per mm")
plt.title("Bill depth vs body mass: three lines up, one line down")
plt.show()
```

<details><summary>Expected Output</summary>

~~~text
Adelie     slope +217 g per mm
Chinstrap  slope +205 g per mm
Gentoo     slope +369 g per mm
pooled     slope -192 g per mm
~~~
</details>

> **Read it:** within any one species, a deeper bill means a heavier bird -- every solid
> line rises, by 205 to 369 g per extra millimetre. But the Gentoo cloud sits high and to
> the left (heavy birds, shallow bills), so the one dashed line through all three clouds
> runs *downhill*. The lurking variable is species. Report the pooled -0.47 and you have
> described penguin taxonomy, not penguin anatomy. In Step 2's angle language: 118 degrees
> apart pooled, under 90 inside every species.

## Your Turn (Exercises)

### Exercise 1 -- Group-check the strong one

Step 1's flipper/body-mass r was +0.87 pooled. Is it honest, or another paradox? Compute
the **per-species r** for flipper vs body mass (reuse Step 4's `groupby` loop with the
flipper/mass pair). One sentence: does the +0.87 survive the group check?

```python
# TODO: your code here
```

<details><summary>Expected Output (one possible answer -- match the values, not the print format)</summary>

~~~text
Adelie     r = +0.47
Chinstrap  r = +0.64
Gentoo     r = +0.70
~~~
*(All positive -- weaker than the pooled +0.87 (part of that number was just "Gentoo is
big"), but the direction survives in every species. NOT a paradox this time: the group
check confirms the pooled story instead of reversing it.)*
</details>

### Exercise 2 -- Transfer: Anscombe's Quartet

`anscombe` (loaded in setup) holds the concept note's opening figure: 44 rows, a
`dataset` column (`I` to `IV`) and the columns `x` and `y`.

**(a)** For each dataset, compute Pearson's r and Spearman's r.

**(b)** Spearman works on ranks. For which dataset does Spearman come out well **above**
Pearson -- rescuing it -- and why there? Why does it *not* rescue dataset II? And why does
IV's Spearman *fall*? (Look back at the concept's quartet figure.)

**(c)** Dataset III is a perfect line plus one outlier (its largest `y`). Drop that row and
recompute Pearson. Then do the same for dataset IV, dropping its row with the largest `x`.
What does pandas return for IV, and why? (Hint: L18's Eva.)

> **Hint:** loop with `for name, g in anscombe.groupby("dataset"):`; `g["x"].corr(g["y"])`
> is Pearson, add `method="spearman"` for Spearman. To drop one row:
> `g.drop(g["y"].idxmax())`.

```python
# TODO: your code here
```

<details><summary>Expected Output (one possible answer -- match the values, not the print format)</summary>

~~~text
dataset I    Pearson +0.816  Spearman +0.818
dataset II   Pearson +0.816  Spearman +0.691
dataset III  Pearson +0.816  Spearman +0.991
dataset IV   Pearson +0.817  Spearman +0.500
III without its outlier: Pearson +1.000
IV without its outlier: Pearson nan
~~~
**(a)** Four identical Pearson values -- the concept's warning, now computed by you.

**(b)** Spearman rescues **III** (0.816 up to 0.991): ranks tame the one outlier, so it
reads the near-perfect line underneath. It cannot rescue **II**, because II is a curve that
rises and then falls -- not monotone -- so its ranks do not line up either (0.691, even
lower than Pearson). **IV** falls furthest of all (0.817 down to 0.500): ten of its x
values are tied at 8, so the ranks have almost nothing to order, and they too rest on the
single far point. For I, a plain linear cloud, the two agree. Spearman helps with outliers
on a monotone pattern -- not with a curve that turns, and not when x barely varies.

**(c)** Without its outlier, III is exactly a line: r = 1.000. IV returns **nan**: once
its one far point is gone, all ten remaining x values equal 8, so the centered x column is
the zero vector -- like Eva in L18, it has no direction, and r is 0 divided by 0. pandas
may also print "RuntimeWarning: invalid value encountered in divide", which is that 0 / 0.
IV's whole +0.817 came from one point.
</details>

### Exercise 3 -- Name the Confounder

Ice-cream sales correlate strongly with drowning deaths, month by month. In two or three
sentences: name the confounder, explain the mechanism, and give the analogous statement
for the bill-depth paradox of Step 4 (what plays the role of "summer", and what would
"banning ice cream" correspond to?).

<details><summary>One acceptable answer</summary>

The confounder is summer: hot weather independently raises both ice-cream sales and
swimming (hence drownings); neither causes the other. In Step 4 the role of summer is
played by species -- being a Gentoo independently makes a penguin heavier AND
shallower-billed, manufacturing the pooled negative correlation. "Banning ice cream to
prevent drownings" corresponds to filing down bills to fatten penguins: intervening on a
variable that never caused the outcome.
</details>

### Exercise 4 -- Autocorrelation: Does San Juan Repeat Itself?

L17 saw San Juan's 12-month temperature cycle in a picture and promised a number for it.
The number is the **autocorrelation**: r between a series and a copy of itself shifted by
`k` steps (the *lag*). The cell below rebuilds L17's monthly mean temperature; run it as is.

```python
# Given: L17's San Juan record -> monthly mean temperature
import urllib.request

# NOAA NCEI GHCN-Daily, San Juan L. M. Marin Intl Airport (public domain)
WEATHER_LOCAL = "data/sju_weather_1960_2024.csv"
WEATHER_URL = ("https://www.ncei.noaa.gov/access/services/data/v1"
               "?dataset=daily-summaries&stations=RQW00011641"
               "&startDate=1960-01-01&endDate=2024-12-31"
               "&dataTypes=TMAX,TMIN,PRCP&format=csv&units=metric")

if not os.path.exists(WEATHER_LOCAL):
    os.makedirs("data", exist_ok=True)
    urllib.request.urlretrieve(WEATHER_URL, WEATHER_LOCAL)

weather = pd.read_csv(WEATHER_LOCAL, parse_dates=["DATE"])
weather = weather.set_index("DATE").sort_index()
weather["TMID"] = (weather["TMAX"] + weather["TMIN"]) / 2
temp_m = weather["TMID"].resample("ME").mean()
print(len(temp_m), "months")
```

<details><summary>Expected Output</summary>

~~~text
780 months
~~~
</details>

Now compute `temp_m.autocorr(lag=k)` for `k` = 1, 3, 6 and 12, and turn the lag-6 and
lag-12 values into angles as in Step 2. Then explain in a sentence each: **(a)** why lag
12 is the highest; **(b)** why lag 6 is negative; **(c)** lag 1 is high too -- why does
that *not* mean the cycle repeats every month?

> **Hint:** `temp_m.autocorr(lag=12)` is the same number as
> `temp_m.corr(temp_m.shift(12))`: pair every month with the month 12 steps earlier, then
> take ordinary r.

```python
# TODO: your code here
```

<details><summary>Expected Output (one possible answer -- match the values, not the print format)</summary>

~~~text
lag  1: r = +0.835
lag  3: r = +0.142
lag  6: r = -0.598
lag 12: r = +0.869
lag  6: angle 126.7 degrees
lag 12: angle 29.7 degrees
~~~
**(a)** Lag 12 pairs each month with the same month a year earlier -- August with
August -- and the seasonal cycle's period is 12 months, so the pairs agree best: only
29.7 degrees apart, about as close as flipper length and body mass.

**(b)** Lag 6 pairs each month with the opposite half of the cycle -- a warm August with
the cool February before it -- so when one is above its average the other tends to be
below: negative r, an angle over 90 degrees.

**(c)** Neighboring months are similar because temperature changes slowly, not because
anything repeats. The period shows where r climbs back up after dropping: down to +0.142
at lag 3, negative at 6, and back to its highest at 12.
</details>

## Summary

| Move | Key commands | What you learned |
|------|-------------|------------------|
| Quantify a relationship | `x.corr(y)` | Pearson r = +0.87 for flipper vs body mass |
| Build r from L18 | `xc @ yc / (norm(xc)*norm(yc))` | r IS the cosine of the mean-centered columns (=0.871202) |
| Center first, read the angle | `np.degrees(np.arccos(r))` | raw cosines are near 1; centered, bill depth/mass is 118.2 degrees apart (r < 0) |
| Predict with it | `np.polyfit(x, y, 1)`; r-squared | the best-fit line; r-squared = 0.76 of variance explained |
| Scale to all pairs | `df.corr()` + diverging heatmap | -0.47 bill-depth/mass is the lead to chase |
| Group-check it | `groupby` + per-group r and lines | Simpson's paradox: -0.47 pooled, positive within every species |
| Pearson vs Spearman | `method="spearman"` | ranks tame an outlier (Anscombe III), not a curve that turns (II) or an x that barely varies (IV) |
| Autocorrelation | `s.autocorr(lag=k)` | San Juan: +0.869 at lag 12, -0.598 at lag 6 -- the 12-month period as a number |

Next lesson (L20): when there are too many variables for pairwise r, **PCA** projects them
down to a 2-D picture -- using L18's projection, the same dot product once more.
