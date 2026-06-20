---
title: "Lab: From 2,003 Features to the 30 That Matter"
unit: "IV"
lesson: "13"
type: lab
tags: [feature-selection, chi-square, mutual-information, variance-threshold, sklearn]
difficulty: introductory
duration: "75 mins"
---

**Goal:** take the 5,574 x 2,003 feature matrix you built in L12 and find the features that actually
matter -- without training any model. Drop the near-constant columns, rank the rest with a chi-square
filter, and confirm the leaders with a second, unrelated filter (mutual information). Pairs with the
concept note [Feature Selection I: Filter Methods](l13_concept_feature_selection_filters.qmd).

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
Variance threshold kept 1483 of 2003 features
~~~
</details>

> **Read it:** 520 near-dead columns gone in one line, before any label-based scoring. This is a first
> triage, not a final answer -- a rare word can still be a perfect spam tell, so we score the survivors
> against the label next.

## Step 2: Chi-Square -- Rank All 2,003 Against the Label (Worked + Completion)

The chi-square score asks, for each feature independently: do its values differ across the classes more
than chance would predict? One call scores all 2,003 candidates.

```python
scores, _ = chi2(X, y)
ranking = pd.Series(scores, index=feature_names).sort_values(ascending=False)
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
> -- all rank in the top five, ahead of 1,998 of the 2,000 vocabulary columns. `digit_count` scores far
> higher than `free`, the best word. The vocabulary that does rank high reads like a parody of spam: free,
> txt, claim, www, prize. Domain insight beat the machinery, again.

Keep the top 30 with `SelectKBest`:

```python
selector = SelectKBest(chi2, k=30).fit(X, y)
kept = feature_names[selector.get_support()]
print("Kept", int(selector.get_support().sum()), "of", X.shape[1], "features:")
print(sorted(kept))
```

<details><summary>Expected Output</summary>

~~~text
Kept 30 of 2003 features:
['1000', '150p', '16', '18', '50', '500', 'cash', 'claim', 'contact', 'cs', 'customer', 'digit_count', 'exclam_count', 'free', 'guaranteed', 'length', 'mobile', 'nokia', 'prize', 'reply', 'service', 'stop', 'text', 'tone', 'txt', 'uk', 'urgent', 'win', 'won', 'www']
~~~
</details>

The other end of the ranking is just as instructive. Which features are most useless?

```python
# COMPLETION: show the 10 LOWEST-scoring features. Sort the ranking from smallest to
# largest, then take the first 10. Fill the ____ and uncomment both lines.
# weakest = ranking.sort_values(ascending=____).head(10)
# print(weakest.round(3))
```

<details><summary>Expected Output (after completing <code>True</code>)</summary>

~~~text
30          0.000
fri         0.000
try         0.000
luv         0.002
stay        0.003
id          0.003
midnight    0.005
january     0.005
internet    0.005
cal         0.005
dtype: float64
~~~
*(The word "30" scores 0.000 to three decimals: it appears at almost exactly the same rate in spam and ham,
so it carries essentially no information about the label. A column can be frequent and still be worthless.)*
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
> the same. That agreement is evidence the leaders are real signal, not an artifact of one method.

## Your Turn

### Exercise 1 -- How few features hold the signal?

Re-run `SelectKBest(chi2, k=...)` with `k=10` and print the kept features sorted. Compare to the k=30 list
above: are the strongest features stable, or does the membership churn?

```python
# Your Turn 1: keep the top 10 by chi-square and print them sorted.
# kept10 = feature_names[SelectKBest(chi2, k=10).fit(X, y).get_support()]
# print(sorted(kept10))
```

<details><summary>Answer</summary>

~~~text
['claim', 'digit_count', 'exclam_count', 'free', 'length', 'mobile', 'prize', 'stop', 'txt', 'www']
~~~
The strongest features are a stable core: every one of these ten is also in the k=30 list. Selection does
not reshuffle the leaders as you change k -- it extends a settled ranking.
</details>

### Exercise 2 -- See the blind spot

Every filter scores each feature alone. Build a tiny exclusive-or dataset -- two 0/1 flags where the label
is 1 exactly when the flags *disagree* -- and score each flag with both filters.

```python
# Your Turn 2: two flags; label = flag_a XOR flag_b. Run and read the scores.
# combos = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
# Xtoy = csr_matrix(np.repeat(combos, 100, axis=0))
# ytoy = Xtoy.toarray()[:, 0] ^ Xtoy.toarray()[:, 1]
# print("chi2 per flag:", [round(v, 4) for v in chi2(Xtoy, ytoy)[0]])
# print("MI per flag:  ", [round(v, 4) for v in mutual_info_classif(Xtoy, ytoy, discrete_features=True, random_state=0)])
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
| Variance threshold | `VarianceThreshold(threshold=0.001)` | Drop near-constant columns label-free (2003 -> 1483) |
| Chi-square rank | `chi2`, `SelectKBest(k=30)` | Your handcrafted features top all 2,000 words |
| Read the bottom | `ranking.sort_values()` | A word balanced across classes ("30", "fri") scores 0.000 |
| Second opinion | `mutual_info_classif` | An unrelated filter crowns 9 of the same top 10 |
| The blind spot | XOR toy | Filters miss features that only work together |

Next (**L14**): hand these features to a model that sees them *together* -- a decision tree you can read
aloud, and a random forest -- and measure what selection actually costs in accuracy.
