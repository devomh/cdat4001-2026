---
title: "Lab: Data-Mining Mini-Project -- What Predicts Higher Income?"
unit: "IV"
lesson: "15"
type: lab
tags: [data-mining, capstone, feature-engineering, feature-selection, decision-trees, random-forests]
difficulty: introductory
duration: "75 mins"
---

**Goal:** run the **whole** Unit IV loop -- engineer features, select the informative ones, train a rule you
can read, and decide what is honest to conclude -- on a dataset you have not seen. This is Unit IV's
capstone: L11 (engineer), L13 (filter-select), and L14 (trees, forests, the honest yardstick), end to end,
ending in a judgment.

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/devomh/cdat4001-2026/blob/main/u04_data_mining/l15_lab_data_mining_mini_project.ipynb)

> This page is the read-only view. To run the lab, open the notebook
> (`l15_lab_data_mining_mini_project.ipynb`) -- in Colab via the badge above, or locally. Every text output
> below comes from actually running the cells. No random numbers beyond seeded splits/models -- everyone
> sees identical results.

> **Where this sits:** the capstone of **Unit IV** -- runs the tabular loop on L11 (features), L13
> (filters), and L14 (trees & forests); L12 was the unit's text track. It is the unit's integrative
> deliverable; the techniques are behind you, the judgment is the point.

## The Problem

What predicts whether a person earns more than $50,000 a year? Today's dataset: **Adult / Census Income** --
a 1994 U.S. census extract (UCI/OpenML "adult", id 1590; public domain). We use a deterministic 15,000-row
subset: a mix of categorical and numeric columns, a clear binary target (`class`: `<=50K` or `>50K`), and --
unavoidably -- columns about people (age, education, occupation, and also **race** and **sex**) that force a
fairness question we will not dodge. This is the Unit IV loop on a completely different shape of data from
the SMS messages: **engineer -> select -> model -> interpret.**

## Prerequisites & Setup

Run the setup cells first. The data is bundled as `data/adult_subset.csv` (the cell falls back to a one-time
OpenML download that reproduces the exact same 15,000-row subset).

```python
# Setup cell 1 of 2: install only (Colab resets on open; run this first)
%pip install -q pandas numpy scikit-learn
```

```python
# Setup cell 2 of 2: imports + load the data (bundled, with an OpenML fallback)
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import mutual_info_classif
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.ensemble import RandomForestClassifier

CACHE = "data/adult_subset.csv"
if os.path.exists(CACHE):
    df = pd.read_csv(CACHE)
else:
    from sklearn.datasets import fetch_openml
    raw = fetch_openml("adult", version=2, as_frame=True, parser="auto")
    # fnlwgt is a census sampling weight, not a property of the person -- drop it (L11)
    df = raw.frame.dropna().drop(columns=["fnlwgt"]).sample(n=15000, random_state=0).reset_index(drop=True)
    os.makedirs("data", exist_ok=True)
    df.to_csv(CACHE, index=False)

print("shape:", df.shape)
```

<details><summary>Expected Output</summary>

~~~text
shape: (15000, 14)
~~~
</details>

## Step 1: First Look -- Structure, Target, and What to Watch

The L07/L10 ritual on a new dataset: what is here, what is the headline target, what should worry us.

```python
print(df["class"].value_counts())
print("baseline (predict <=50K for everyone):", round((df["class"] == "<=50K").mean(), 3))
```

<details><summary>Expected Output</summary>

~~~text
class
<=50K    11240
>50K      3760
Name: count, dtype: int64
baseline (predict <=50K for everyone): 0.749
~~~
</details>

The classes are imbalanced -- about 75% earn `<=50K`, so "predict `<=50K` for everyone" already scores
**74.9%** without learning anything. That is the number any model must beat (the L14 baseline). Now sort the
columns by type, and notice what is in them:

```python
cat_cols = ["workclass", "education", "marital-status", "occupation",
            "relationship", "race", "sex", "native-country"]
num_cols = ["age", "education-num", "capital-gain", "capital-loss", "hours-per-week"]
print(len(cat_cols), "categorical features,", len(num_cols), "numeric features")
print("sensitive attributes present:", [c for c in cat_cols if c in ("race", "sex")])
```

<details><summary>Expected Output</summary>

~~~text
8 categorical features, 5 numeric features
sensitive attributes present: ['race', 'sex']
~~~
</details>

> **Two things up front.** The `fnlwgt` column was already dropped in setup: it is a census *sampling
> weight*, not a property of the person, and a feature must measure the observation (L11). And the table
> contains **race** and **sex** -- we will encode them like any other column, but flag the fairness question
> they raise and return to it at the end.

## Step 2: Engineer the Feature Matrix (L11)

An algorithm needs numbers. One-hot the eight categoricals; keep the five numerics as they are.

```python
X_cat = pd.get_dummies(df[cat_cols], dtype=int)
X = pd.concat([X_cat, df[num_cols]], axis=1)
y = (df["class"] == ">50K").astype(int)            # 1 = earns >50K
print("one-hot expanded", len(cat_cols), "categoricals into", X_cat.shape[1], "columns")
print("feature matrix X:", X.shape)
```

<details><summary>Expected Output</summary>

~~~text
one-hot expanded 8 categoricals into 97 columns
feature matrix X: (15000, 102)
~~~
</details>

> **Two engineering judgments.** (1) We did **not scale** the numerics. L11 scaled features because
> distance- and gradient-based algorithms are scale-sensitive -- but our models here are **decision trees
> and forests**, which split on thresholds and are *scale-invariant*: scaling `age` or `capital-gain` would
> not change a single split. Engineering is choosing the right transform, and here the right choice is to
> skip one. (2) `education` and `education-num` are the **same information twice** (a redundant pair, L13) --
> we keep both and let the importance ranking reveal the redundancy.

## Step 3: An Honest Yardstick -- Split Before You Select

Everything learned from here on -- the feature selection included -- is fit on the training 70% and judged on
the 30% it never saw (L14). Split first, so nothing downstream can peek at the test set.

```python
X_train, X_test, y_train, y_test = train_test_split(
    X.values, y, test_size=0.3, random_state=42, stratify=y)
print("train:", X_train.shape[0], " test:", X_test.shape[0])
print("share earning >50K -- train:", round(y_train.mean(), 3), " test:", round(y_test.mean(), 3))
```

<details><summary>Expected Output</summary>

~~~text
train: 10500  test: 4500
share earning >50K -- train: 0.251  test: 0.251
~~~
</details>

## Step 4: Select the Informative Features (L13 + L14)

Two ways to ask "which columns carry signal," fit on the training data only. First a **mutual-information**
filter -- and note *why not chi-square*: our matrix has continuous columns (`capital-gain`, `age`), and
chi-square would crown the biggest-magnitude column rather than the most informative one (L13). Mutual
information has no such bias.

```python
feature_names = np.array(X.columns)
discrete = np.array([set(np.unique(X_train[:, i])) <= {0, 1} for i in range(X_train.shape[1])])
mi = mutual_info_classif(X_train, y_train, discrete_features=discrete, random_state=0)
mi_rank = pd.Series(mi, index=feature_names).sort_values(ascending=False)
print(mi_rank.head(8).round(4))
```

<details><summary>Expected Output</summary>

~~~text
marital-status_Married-civ-spouse    0.1059
relationship_Husband                 0.0833
age                                  0.0688
capital-gain                         0.0670
marital-status_Never-married         0.0648
education-num                        0.0640
capital-loss                         0.0412
hours-per-week                       0.0401
dtype: float64
~~~
</details>

Now the **forest's importance** -- a model-based ranking from a completely different mechanism (L14). Train
the forest on the training data and read its top features:

```python
forest = RandomForestClassifier(n_estimators=200, random_state=42)
forest.fit(X_train, y_train)
importance = pd.Series(forest.feature_importances_, index=feature_names).sort_values(ascending=False)
print(importance.head(8).round(4))
```

<details><summary>Expected Output</summary>

~~~text
age                                  0.2193
hours-per-week                       0.1130
capital-gain                         0.0886
marital-status_Married-civ-spouse    0.0696
education-num                        0.0680
relationship_Husband                 0.0475
capital-loss                         0.0320
marital-status_Never-married         0.0266
dtype: float64
~~~
</details>

> **Read it:** the filter and the forest order things differently in the details, but they crown the same
> handful -- `marital-status`, `capital-gain`, `education-num`, `age`, `hours-per-week`, `relationship`. Two
> unrelated methods agreeing (L14) is the strongest evidence those features carry the real signal.

## Step 5: Train a Rule You Can Read, and a Forest (L14)

A depth-3 decision tree (three questions per person) and a 200-tree forest, scored on the held-out test set
against the 74.9% baseline.

```python
tree = DecisionTreeClassifier(max_depth=3, random_state=42)
tree.fit(X_train, y_train)
print("Decision tree (depth 3) test accuracy:", round(tree.score(X_test, y_test), 3))
print("Random forest (200 trees) test accuracy:", round(forest.score(X_test, y_test), 3))
print()
print(export_text(tree, feature_names=list(feature_names)))
```

<details><summary>Expected Output</summary>

~~~text
Decision tree (depth 3) test accuracy: 0.829
Random forest (200 trees) test accuracy: 0.832
|--- marital-status_Married-civ-spouse <= 0.50
|   |--- capital-gain <= 7139.50
|   |   |--- education-num <= 12.50
|   |   |   |--- class: 0
|   |   |--- education-num >  12.50
|   |   |   |--- class: 0
|   |--- capital-gain >  7139.50
|   |   |--- capital-gain <= 8028.50
|   |   |   |--- class: 1
|   |   |--- capital-gain >  8028.50
|   |   |   |--- class: 1
|--- marital-status_Married-civ-spouse >  0.50
|   |--- education-num <= 12.50
|   |   |--- capital-gain <= 5095.50
|   |   |   |--- class: 0
|   |   |--- capital-gain >  5095.50
|   |   |   |--- class: 1
|   |--- education-num >  12.50
|   |   |--- capital-gain <= 5095.50
|   |   |   |--- class: 1
|   |   |--- capital-gain >  5095.50
|   |   |   |--- class: 1
~~~
</details>

## Step 6: Interpret and Decide -- the Synthesis

**Read the rule aloud** (class 0 = `<=50K`, 1 = `>50K`): *"Not married? Then `<=50K` unless there is a large
capital gain. Married? Then `>50K` if there is more than a high-school-plus education or any sizable capital
gain, otherwise `<=50K`."* Three questions -- marital status, education, capital gains -- capture most of the
signal.

**Which model would you ship?** The forest scores **0.832**, the readable tree **0.829** -- the forest is
ahead by **0.3 of a point**, essentially a tie. L14 warned that a forest usually buys accuracy by giving up
readability; here that price has nearly vanished, so the **readable tree wins** -- it is as accurate as the
forest *and* you can hand it to a person. Both clear the 74.9% baseline by about eight points: a real,
honest result, not a 99% demo.

**The fairness flag.** Our feature matrix one-hot-encoded `race` and `sex`. The readable rule happens not to
use them -- its splits are marital status, education, and capital gains -- but a model that *predicts income
from demographic attributes* raises real ethical and legal questions, and "the tree didn't use them" is not
the same as "the data didn't carry them." We name the issue here and examine it properly in **Unit VIII --
Data Ethics (L27)**. Reading a feature's importance also never proves *causation*: "marital status ranks
high" describes who earned more in 1994, in this sample -- not a lever you can pull.

## Your Turn

**1. The redundant twin.** `education` (one-hot) and `education-num` carry the same information. Re-run the
depth-3 tree using `X` with the `education_*` one-hot columns dropped (keep `education-num`). Does test
accuracy move? What does that tell you about redundant features?

```python
# Your Turn 1: drop the education_* one-hot columns, refit the depth-3 tree, compare accuracy.
```

<details><summary>Answer</summary>

Accuracy does not move (0.829 either way): the depth-3 tree already prefers `education-num` over the 16
`education_*` dummies, so removing the dummies changes nothing it relied on. Redundant features split the
credit without adding signal -- dropping one twin is usually safe and always simplifies the matrix. The
honest read is "same information, fewer columns."
</details>

**2. Where the rankings agree.** Print the top-8 features by mutual information and the top-8 by forest
importance (you computed both in Step 4) and name the features that appear in *both* lists.

```python
# Your Turn 2: print mi_rank.head(8) and importance.head(8); name the features in both.
```

<details><summary>Answer</summary>

**All eight overlap** -- the two top-8 lists are the *same set* in a different order:
`marital-status_Married-civ-spouse`, `marital-status_Never-married`, `relationship_Husband`, `age`,
`capital-gain`, `capital-loss`, `education-num`, and `hours-per-week`. A filter (model-free) and a forest
(200 trees) share no machinery, so total agreement on the top eight is strong evidence about which features
matter.
</details>

**3. Written synthesis (3-5 sentences).** You must hand a policymaker a model and a one-paragraph summary.
Which model do you ship and why? And what is the single most important caveat about concluding "X causes
higher income" from this analysis? (Consider: associational vs causal, the data is from 1994, the sample is
bounded, and some features are sensitive proxies.)

## Recap

You ran the entire Unit IV loop on a fresh dataset: **engineered** a 102-column feature matrix (one-hot the
categoricals; *skip* scaling because trees are scale-invariant; drop a non-feature), **selected** the
drivers with a mutual-information filter and forest importance that agreed, **trained** a readable depth-3
tree (82.9%) that matched a forest (83.2%) and beat the 74.9% baseline, and **interpreted** the result
honestly -- which model to ship, what the rule says, and the fairness question the features raise. That last
step -- deciding what is honest to conclude -- is the skill the whole unit was building toward. Unit V turns
from *mining* patterns to *seeing* them: visualization.
