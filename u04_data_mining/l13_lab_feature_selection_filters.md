---
title: "Lab: From 2,003 Features to the 30 That Matter"
unit: "IV"
lesson: "13"
type: lab
tags: [feature-selection, chi-square, mutual-information, variance-threshold, sklearn]
difficulty: introductory
duration: "85 mins"
---

**Goal:** take the 5,574 x 2,003 feature matrix you built in L12 and find the features that actually
matter -- without training any model. Drop the near-constant columns, rank the rest with a chi-square
filter, and confirm the leaders with a second, unrelated filter (mutual information). Then carry the
filters to a new dataset, where they do not all apply. Pairs with the concept note [Feature Selection I: Filter Methods](l13_concept_feature_selection_filters.qmd).

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/devomh/cdat4001-2026/blob/main/u04_data_mining/l13_lab_feature_selection_filters.ipynb)

> This page is the read-only view. To run the lab, open the notebook
> (`l13_lab_feature_selection_filters.ipynb`) -- in Colab via the badge above, or locally. Every text
> output below comes from actually running the cells. No model is trained here -- that is L14.

> **Where this sits:** L12 ended with a 2,003-column matrix and the question *which columns matter?* This
> lab answers it with **model-free filters**. **L14** then lets a decision tree select features while
> seeing them together.

## Prerequisites & Setup

Run this first. Same dataset as L12 -- the **SMS Spam Collection** (Almeida, Gomez Hidalgo & Yamakami 2011;
UCI Machine Learning Repository id 228, CC BY 4.0), bundled as `data/sms_spam.csv` with a UCI fallback.

```python
# Setup cell 1 of 2: install only (Colab resets on open; run this first)
%pip install -q pandas numpy scikit-learn scipy
```

```python
# Setup cell 2 of 2: imports + rebuild L12's feature matrix X (5574 x 2003) and the label y
import os
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from scipy.sparse import hstack, csr_matrix
from sklearn.feature_selection import VarianceThreshold, chi2, mutual_info_classif, SelectKBest

LOCAL = "data/sms_spam.csv"
URL = "https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip"

if os.path.exists(LOCAL):
    sms = pd.read_csv(LOCAL)
else:
    import io, zipfile, urllib.request
    with urllib.request.urlopen(URL) as r:
        z = zipfile.ZipFile(io.BytesIO(r.read()))
    sms = pd.read_table(z.open("SMSSpamCollection"), header=None,
                        names=["label", "message"], quoting=3)

# the three handcrafted features (L11/L12) + the 2,000-word bag of words (L12), stacked side by side
sms["length"] = sms["message"].str.len()
sms["digit_count"] = sms["message"].str.count(r"\d")
sms["exclam_count"] = sms["message"].str.count("!")

vectorizer = CountVectorizer(stop_words="english", max_features=2000)
X_counts = vectorizer.fit_transform(sms["message"])
handcrafted = ["length", "digit_count", "exclam_count"]
X = hstack([X_counts, csr_matrix(sms[handcrafted].values)]).tocsr()
feature_names = np.array(list(vectorizer.get_feature_names_out()) + handcrafted)
y = (sms["label"] == "spam").astype(int)

print("Feature matrix:", X.shape)
print("Label spam share:", round(y.mean(), 3))
```

<details><summary>Expected Output</summary>

~~~text
Feature matrix: (5574, 2003)
Label spam share: 0.134
~~~
</details>

2,003 candidate features, one label `y` (1 = spam). The question of the day: how few of these columns can
we keep and still tell spam from ham?

## Step 1: Variance Threshold -- Drop the Near-Constant Columns (Worked)

The cheapest filter needs no label at all: drop columns that barely vary. A word appearing in almost no
messages is a column of almost all zeros -- it cannot separate anything.

```python
vt = VarianceThreshold(threshold=0.001).fit(X)
print("Variance threshold kept", int(vt.get_support().sum()), "of", X.shape[1], "features")
```

<details><summary>Expected Output</summary>

~~~text
Variance threshold kept 1482 of 2003 features
~~~
</details>

> **Read it:** 521 near-dead columns gone in one line, before any label-based scoring. This is a first
> triage, not a final answer -- a rare word can still be a perfect spam tell, so we score the survivors
> against the label next.

## Step 2: Chi-Square -- Rank All 2,003 Against the Label (Worked + Completion)

The chi-square score asks, for each feature independently: do its values differ across the classes more
than chance would predict? One call scores all 2,003 candidates.

```python
scores, _ = chi2(X, y)
# kind="stable": tied scores keep a fixed (alphabetical) order
ranking = pd.Series(scores, index=feature_names)
ranking = ranking.sort_values(ascending=False, kind="stable")
print(ranking.head(15).round(1))
```

<details><summary>Expected Output</summary>

~~~text
digit_count     65270.3
length          36303.1
free             1049.0
txt               944.4
exclam_count      789.5
claim             730.2
mobile            707.4
www               616.7
prize             601.0
stop              555.4
uk                469.8
150p              458.8
text              438.8
reply             412.4
nokia             408.7
dtype: float64
~~~
</details>

> **Read it:** the three features *you handcrafted in L11/L12* -- `digit_count`, `length`, `exclam_count`
> -- all rank in the top five, ahead of 1,998 of the 2,000 vocabulary columns. The vocabulary that does
> rank high reads like a parody of spam: free, txt, claim, www, prize.
>
> **But read the size of the gap with care.** A word column counts a word a handful of times; `length`
> counts characters, in the hundreds. Chi-square treats every value as a count, so a column of bigger
> numbers earns a bigger score: measured in hundreds of characters, `length` would score 363 and drop from
> 2nd place to 18th. Chi-square scores compare fairly only among columns of the same kind (word counts with
> word counts). Step 3 checks the leaders with a filter that ignores units.

Keep the top 30 with `SelectKBest`:

```python
selector = SelectKBest(chi2, k=30).fit(X, y)
kept = feature_names[selector.get_support()]
print("Kept", int(selector.get_support().sum()), "of", X.shape[1], "features:")
print(sorted(kept.tolist()))
```

<details><summary>Expected Output</summary>

~~~text
Kept 30 of 2003 features:
['1000', '150p', '16', '18', '50', '500', 'cash', 'claim', 'contact', 'cs', 'customer', 'digit_count', 'exclam_count', 'free', 'guaranteed', 'length', 'mobile', 'nokia', 'prize', 'reply', 'service', 'stop', 'text', 'tone', 'txt', 'uk', 'urgent', 'win', 'won', 'www']
~~~
</details>

The other end of the ranking is just as instructive. Which features are most useless?

```python
# COMPLETION: show the 10 LOWEST-scoring features.
# Sort from smallest to largest, then take the first 10.
# Fill the ____ and uncomment the three code lines.
# weakest = ranking.sort_values(ascending=____,
#                               kind="stable").head(10)
# print(weakest.round(3))
```

<details><summary>Expected Output (after completing <code>True</code>)</summary>

~~~text
30            0.000
fri           0.000
try           0.000
luv           0.002
id            0.003
stay          0.003
cal           0.005
changed       0.005
definitely    0.005
feb           0.005
dtype: float64
~~~
*(The word "30" scores 0.000 to three decimals: it appears at almost exactly the same rate in spam and ham,
so it carries essentially no information about the label. A column can be frequent and still be worthless.
Several scores are exact ties -- "30"/"fri", "id"/"stay", and the last four rows, which are 4 of 17 words
tied at 0.005. The stable sort keeps tied words in alphabetical order, so everyone sees the same list.)*
</details>

## Step 3: Mutual Information -- A Second, Unrelated Opinion (Worked)

Mutual information measures the *shared information* between a feature and the label -- a different idea from
chi-square's observed-vs-expected counts. If two unrelated methods agree on the leaders, believe them.

```python
mi = mutual_info_classif(X, y, discrete_features=True, random_state=0)
mi_ranking = pd.Series(mi, index=feature_names).sort_values(ascending=False)
print(mi_ranking.head(12).round(4))
```

<details><summary>Expected Output</summary>

~~~text
digit_count     0.3033
length          0.1735
txt             0.0495
exclam_count    0.0483
free            0.0443
claim           0.0402
www             0.0347
mobile          0.0344
prize           0.0311
150p            0.0261
stop            0.0252
uk              0.0249
dtype: float64
~~~
</details>

```python
chi_top10 = set(ranking.head(10).index)
mi_top10 = set(mi_ranking.head(10).index)
print("Shared in both top-10s:", len(chi_top10 & mi_top10), "of 10")
print(sorted(chi_top10 & mi_top10))
```

<details><summary>Expected Output</summary>

~~~text
Shared in both top-10s: 9 of 10
['claim', 'digit_count', 'exclam_count', 'free', 'length', 'mobile', 'prize', 'txt', 'www']
~~~
</details>

> **Read it:** chi-square and mutual information share no machinery, yet nine of their top ten features are
> the same. That agreement is evidence the leaders are real signal, not an artifact of one method. And
> mutual information ignores units, so it settles Step 2's question: `digit_count` and `length` lead on
> their merits, far ahead of the best word. Domain insight beat the machinery, again -- chi-square only
> exaggerated by how much.
>
> **One caveat about `discrete_features=True`:** it treats every distinct value as a category. That suits
> word counts and flags, but `length` has 273 distinct values, and a many-valued column can pick up score
> by chance. Scored as a continuous measurement instead, `length` gets 0.15 rather than 0.17 -- still
> 2nd, so the ranking stands.

## Your Turn

> **Before the exercises:** the concept note's table in *When to Use Which Filter* lists, for each filter,
> which columns it accepts and what to watch out for. Exercise 1 needs it.

### Exercise 1 -- New data, same filters

The SMS matrix is a friendly case: every column is a non-negative count, so all three filters apply. Real
tables are rarely that tidy. Here is one you know from L08 -- the **Palmer Penguins** (Gorman, Williams &
Fraser 2014; CC0 public domain), bundled as `data/penguins.csv` with the same seaborn-data fallback as
L08. The label is `species`. Which columns tell the three species apart -- and which filter may you use to
find out? Run the setup cell first.

```python
# Your Turn 1, setup: load the penguins, drop incomplete rows
P_LOCAL = "data/penguins.csv"
P_URL = ("https://raw.githubusercontent.com/mwaskom/"
         "seaborn-data/master/penguins.csv")
peng = pd.read_csv(P_LOCAL if os.path.exists(P_LOCAL) else P_URL)
peng = peng.dropna()
# the four measurement columns, used in parts (b) and (c)
num = ["bill_length_mm", "bill_depth_mm",
       "flipper_length_mm", "body_mass_g"]
print(peng.shape)
print(peng["species"].value_counts())
print(peng.dtypes)
```

<details><summary>Expected Output</summary>

~~~text
(333, 7)
species
Adelie       146
Gentoo       119
Chinstrap     68
Name: count, dtype: int64
species               object
island                object
bill_length_mm       float64
bill_depth_mm        float64
flipper_length_mm    float64
body_mass_g          float64
sex                   object
dtype: object
~~~
</details>

**(a) Read the columns first.** Six columns could be features: `island`, the four measurements, and `sex`.
Using the table, decide which filter(s) fit each one, and why the others do not. Write your answer before
running anything.

<details><summary>Answer</summary>

- **The four measurements** (`bill_length_mm`, `bill_depth_mm`, `flipper_length_mm`, `body_mass_g`) are
  continuous and non-negative, in mm and g. **Mutual information** fits. Chi-square will run (nothing is
  negative), but it treats the measurements as counts, so its scores follow how big the numbers are, not
  how well each column separates the species. A variance threshold is meaningless here: body mass has a
  variance of about 648,000 (grams squared), bill depth about 3.9 (mm squared), so no single cutoff means
  the same thing for both.
- **`island` and `sex`** are text categories. No filter takes text: **one-hot encode** them first (L11).
  The 0/1 columns that result are what **chi-square** is built for (mutual information with
  `discrete_features=True` would fit too).

</details>

**(b) Chi-square on the four measurements.** Score them against `species`. Then measure body mass in
kilograms instead of grams and score it again. **Before running:** which column do you expect chi-square to
put first, and why? Then fill the two `____` and uncomment.

```python
# Your Turn 1(b): chi-square on the measurements
# chi_g = ____(peng[num], peng["species"])[0]
# chi_rank = pd.Series(chi_g, index=num)
# print(chi_rank.sort_values(ascending=False).round(1))
# mass_kg = peng["body_mass_g"] / ____
# print("body mass in kg:",
#       chi2(mass_kg.to_frame(), peng["species"])[0].round(1))
```

<details><summary>Answer (blanks: <code>chi2</code>, <code>1000</code>)</summary>

~~~text
body_mass_g          34511.1
flipper_length_mm      251.4
bill_length_mm         159.5
bill_depth_mm           50.7
dtype: float64
body mass in kg: [34.5]
~~~
Body mass "wins" by a mile -- because grams are big numbers. In kilograms the same column scores 34.5,
below bill depth's 50.7: from first place to last. Same penguins, same information; the ranking was set by
the unit. Even the three mm columns are not comparable: flippers measure about 200 mm, bill depths about
17 mm, and chi-square rewards big numbers -- subtract 13 mm from every bill depth and its score jumps from
50.7 to 209.1, with not one penguin changed. This is the table's chi-square warning: on measurements the
score follows the size of the numbers, so rank them with mutual information.
</details>

**(c) Mutual information, and the two text columns.** Score the measurements with mutual information
(they are continuous, so leave `discrete_features` at its default), then one-hot encode `island` and `sex`
and score those with chi-square. Fill the two `____` and uncomment.

```python
# Your Turn 1(c): MI on the measurements, chi2 on categories
# mi_p = ____(peng[num], peng["species"], random_state=0)
# mi_rank = pd.Series(mi_p, index=num)
# print(mi_rank.sort_values(ascending=False).round(3))
# cats = pd.____(peng[["island", "sex"]], dtype=int)
# chi_c = chi2(cats, peng["species"])[0]
# print(pd.Series(chi_c, index=cats.columns).round(1))
```

<details><summary>Answer (blanks: <code>mutual_info_classif</code>, <code>get_dummies</code>)</summary>

~~~text
flipper_length_mm    0.603
bill_depth_mm        0.575
bill_length_mm       0.531
body_mass_g          0.501
dtype: float64
island_Biscoe       107.2
island_Dream        117.2
island_Torgersen     60.2
sex_FEMALE            0.0
sex_MALE              0.0
dtype: float64
~~~
Body mass goes from **first** under chi-square to **last** under mutual information, which ignores units --
trust MI here. The four MI scores are close, so do not read much into flipper beating bill depth: every
measurement carries real information about the species. `island` scores high (Torgersen has only Adelie
penguins; no Gentoo lives on Dream). `sex` scores 0.0 (0.02 before rounding): every species has almost
exactly as many females as males (73/73, 34/34, 58/61), so a penguin's sex says nothing about its
species -- the penguin version of SMS's word "30".
</details>

### Exercise 2 -- See the blind spot

Every filter scores each feature alone. Build a tiny exclusive-or dataset -- two 0/1 flags where the label
is 1 exactly when the flags *disagree* -- and score each flag with both filters.

```python
# Your Turn 2: two flags; label = flag_a XOR flag_b. Run and read the scores.
# combos = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
# Xtoy = csr_matrix(np.repeat(combos, 100, axis=0))
# ytoy = Xtoy.toarray()[:, 0] ^ Xtoy.toarray()[:, 1]
# s_chi = chi2(Xtoy, ytoy)[0]
# s_mi = mutual_info_classif(Xtoy, ytoy, discrete_features=True,
#                            random_state=0)
# print("chi2 per flag:", [round(float(v), 4) for v in s_chi])
# print("MI per flag:  ", [round(float(v), 4) for v in s_mi])
```

<details><summary>Answer</summary>

~~~text
chi2 per flag: [0.0, 0.0]
MI per flag:   [0.0, 0.0]
~~~
Each flag alone is split 50/50 across the classes, so both filters score it `0.000` -- yet the two flags
*together* determine the label perfectly. A filter, judging each feature alone, would throw both away. This
is the blind spot **L14** fixes: a decision tree considers features together while it works.
</details>

### Exercise 3 -- Written: a low score is not always a reason to drop

A teammate proposes deleting every feature whose chi-square score is below the top 30. Argue for or against
in 3-4 sentences, naming at least one reason a low-scoring feature might still be worth keeping.

> **Hint:** think about two features that overlap (the concept note's `length` / `digit_count` twins), and
> about a variable a regulator or a scientific question requires you to keep regardless of its score.

## Summary

| Move | Key command | What you learned |
|------|-------------|------------------|
| Variance threshold | `VarianceThreshold(threshold=0.001)` | Drop near-constant columns label-free (2003 -> 1482) |
| Chi-square rank | `chi2`, `SelectKBest(k=30)` | `digit_count` and `length` lead (MI confirms; chi-square's gap is partly units) |
| Read the bottom | `ranking.sort_values()` | A word balanced across classes ("30", "fri") scores 0.000 |
| Second opinion | `mutual_info_classif` | An unrelated filter crowns 9 of the same top 10 |
| New dataset | penguins: `chi2` vs `mutual_info_classif` | Check column types first; chi-square ranks by units (body mass 1st in g, last in kg) |
| The blind spot | XOR toy | Filters miss features that only work together |

Next (**L14**): hand these features to a model that sees them *together* -- a decision tree you can read
aloud, and a random forest -- and measure what selection actually costs in accuracy.
