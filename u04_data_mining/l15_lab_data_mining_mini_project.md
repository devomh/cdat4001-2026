---
title: "Lab: Data-Mining Mini-Project -- What Predicts Higher Income?"
unit: "IV"
lesson: "15"
type: lab
tags: [data-mining, capstone, feature-engineering, feature-selection, decision-trees, random-forests]
difficulty: introductory
duration: "90 mins"
---

**Goal:** run the **whole** Unit IV loop -- engineer features, select the informative ones, train a rule you
can read, and decide what is honest to conclude -- on a dataset you have not seen. This is Unit IV's
capstone: L11 (engineer), L13 (filter-select), and L14 (trees, forests, the honest yardstick), end to end,
ending in a judgment -- and then a harder one: does the rule hold for everyone, group by group?

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

## Know the Columns

You cannot read a rule like "`education-num <= 12.5`" until you know what the column means. The
codebook for the 13 features and the target:

| Column | Meaning | Values |
|--------|---------|--------|
| `age` | age in years | 17 to 90 |
| `workclass` | kind of employer | 7 groups: Private, Self-emp-not-inc, Self-emp-inc, Federal-gov, Local-gov, State-gov, Without-pay |
| `education` | highest schooling completed | 16 levels, Preschool to Doctorate |
| `education-num` | the same level as a **rank code** (not years) | 1 to 16 -- see below |
| `marital-status` | legal marital status | Married-civ-spouse (spouse is a civilian), Never-married, Divorced, Separated, Widowed, Married-spouse-absent, Married-AF-spouse (spouse in the Armed Forces) |
| `occupation` | kind of work | 14 groups, e.g. Exec-managerial, Prof-specialty, Craft-repair |
| `relationship` | the person's role in their household, relative to the household's reference person (the first person listed) | Husband, Wife, Own-child, Other-relative, Unmarried, Not-in-family |
| `race` | race, in the 1994 census categories | White, Black, Asian-Pac-Islander, Amer-Indian-Eskimo, Other |
| `sex` | sex as recorded | Female, Male |
| `capital-gain` | profit from selling investments (stocks, property) that year -- **not wages** | US dollars; 0 for most people |
| `capital-loss` | loss from selling investments that year | US dollars; 0 for most people |
| `hours-per-week` | hours usually worked per week | 1 to 99 |
| `native-country` | country of origin | 40 countries; 91% United-States |
| `class` | **the target**: does the person's income exceed $50,000 a year? | `<=50K`, `>50K` |

The category labels are the census's own, kept as recorded (some, like "Amer-Indian-Eskimo", are dated
terms). Three things in this table will matter when we read the trees:

- **`education-num` is a rank, not years of school.** 9 = HS-grad, 10 = Some-college, 11 = Assoc-voc,
  12 = Assoc-acdm, **13 = Bachelors**, 14 = Masters, 15 = Prof-school, 16 = Doctorate (1-8 are Preschool
  through 12th grade). So a tree's cut at **9.5** means "more than high school", **12.5** means "a
  bachelor's degree or higher", and **13.5** means "a master's or higher".
- **`capital-gain` is investment income, and it is rare.** About 92% of people have 0. A handful sit at
  exactly 99,999 -- a suspicious ceiling, so read that value as "at least this much".
- **`sex` and `relationship` are different questions.** `sex` describes the person; `relationship` is
  their role in the household. `Husband` and `Wife` exist only for married people, so they carry sex and
  marriage together:

```python
# count the rows for each (relationship, sex) pair
print(pd.crosstab(df["relationship"], df["sex"]))
married = df["marital-status"] == "Married-civ-spouse"
print()
print("share married, by sex:")
print(married.groupby(df["sex"]).mean().round(3))
```

<details><summary>Expected Output</summary>

~~~text
sex             Female  Male
relationship                
Husband              0  6260
Not-in-family     1795  2096
Other-relative     191   255
Own-child          958  1164
Unmarried         1193   377
Wife               711     0

share married, by sex:
sex
Female    0.155
Male      0.624
Name: marital-status, dtype: float64
~~~
</details>

No woman is a `Husband` and no man is a `Wife`. And only 15.5% of the women here are married, against
62.4% of the men. That is a clue about **who is in the data**. The extraction conditions documented with
the UCI dataset kept only people older than 16 who usually worked at least one hour a week and had an
income above $100. Those conditions are part of the story, but they do not fully explain why so few women
here are married, and the dataset's documentation does not say what else shaped the extract. Whatever the
reason, the married women in this table are a small, selected group -- not "all married women" of 1994.
Keep that in mind in Step 7.

**What $50,000 meant.** The threshold is in 1994 dollars. By the U.S. Bureau of Labor Statistics consumer
price index (CPI-U annual averages: 148.2 in 1994, 321.943 in 2025), $50,000 then is about
**$108,600 in 2025 dollars** (50,000 x 321.943 / 148.2). So `>50K` does not mean "a middle income"; it
means a well-paid worker.

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
capital gain. Married? Then `>50K` with a bachelor's degree or higher (`education-num` 13+ -- see Know the
Columns) or with any sizable capital gain, otherwise `<=50K`."* Three questions -- marital status, education,
capital gains -- capture most of the signal.

**Which model would you ship?** The forest scores **0.832**, the readable tree **0.829** -- the forest is
ahead by **0.3 of a point**, essentially a tie. L14 warned that a forest usually buys accuracy by giving up
readability; here that price has nearly vanished, so the **readable tree wins** -- it is as accurate as the
forest *and* you can hand it to a person. Both clear the 74.9% baseline by about eight points: a real,
honest result, not a 99% demo.

**The fairness flag.** Our feature matrix one-hot-encoded `race` and `sex`. The readable rule happens not to
use them -- its splits are marital status, education, and capital gains -- but a model that *predicts income
from demographic attributes* raises real ethical and legal questions, and "the tree didn't use them" is not
the same as "the data didn't carry them." Step 7 and Your Turn 3 put that sentence to the test; the ethics of it gets a
full lesson in **Unit VIII -- Data Ethics (L27)**. Reading a feature's importance also never proves
*causation*: "marital status ranks high" describes who earned more in 1994, in this sample -- not a lever you
can pull.

## Step 7: Does the Rule Hold for Everyone?

The Step 5 tree describes all 15,000 people at once. But does the same rule hold *inside* each group? Fit
the same depth-3 tree separately for different groups, and compare the rules. We will not tell you in
advance what to expect -- the trees will.

### 7a: How Many People of Each Kind?

Before comparing groups, count them -- and count the right thing. A tree learns its splits mostly from the
**rarer class**, so the number of people earning `>50K` is the real sample size, not the row count.

```python
high_income = df["class"] == ">50K"
for col in ["race", "sex"]:
    # size = rows; sum of True = count >50K; mean = share
    table = high_income.groupby(df[col]).agg(
        ["size", "sum", "mean"])
    table.columns = ["rows", "over_50K", "share"]
    table = table.sort_values("rows", ascending=False)
    print(table.round(3))
    print()
```

<details><summary>Expected Output</summary>

~~~text
                     rows  over_50K  share
race                                      
White               12947      3436  0.265
Black                1361       177  0.130
Asian-Pac-Islander    435       119  0.274
Amer-Indian-Eskimo    129        12  0.093
Other                 128        16  0.125

         rows  over_50K  share
sex                           
Male    10152      3192  0.314
Female   4848       568  0.117
~~~
</details>

> **Read it:** White and Black are the only race groups with enough `>50K` people to grow a tree on;
> Asian-Pac-Islander has 119 in total (about 80 left for training after the split), and the other two have
> 12 and 16. So we compare **White vs Black**. Notice what a "White vs everyone else" split would have
> hidden: Asian-Pac-Islander (0.274) earns `>50K` as often as White (0.265), Black (0.130) far less often.
> How you draw the groups is itself an analysis choice. The sex split is large on both sides; it is yours
> in Your Turn 3.

### 7b: Worked Example -- White vs Black

One helper does the work for any group. It picks the group's rows through the group's **one-hot column**
(`race_White`, `race_Black`, ...) in the Step 3 split -- no new split, so the test rows stay unseen -- and
judges each tree against **its own group's baseline**, because groups with different `>50K` shares start
from different "predict `<=50K` for everyone" scores.

```python
names = list(feature_names)

def subgroup_tree(column):
    """Fit a depth-3 tree on one group's training rows."""
    in_train = X_train[:, names.index(column)] == 1
    in_test = X_test[:, names.index(column)] == 1
    tree_g = DecisionTreeClassifier(max_depth=3,
                                    random_state=42)
    tree_g.fit(X_train[in_train], y_train[in_train])
    y_group = y_test[in_test]
    baseline = max(y_group.mean(), 1 - y_group.mean())
    accuracy = tree_g.score(X_test[in_test], y_group)
    print(column, "--", in_train.sum(), "training rows,",
          round(y_train[in_train].mean(), 3), "earn >50K")
    print("test accuracy:", round(accuracy, 3),
          " own baseline:", round(baseline, 3))
    return tree_g, in_train

white_tree, white_rows = subgroup_tree("race_White")
print(export_text(white_tree, feature_names=names,
                  show_weights=True))
```

<details><summary>Expected Output</summary>

~~~text
race_White -- 9089 training rows, 0.266 earn >50K
test accuracy: 0.821  own baseline: 0.736
|--- marital-status_Married-civ-spouse <= 0.50
|   |--- capital-gain <= 7139.50
|   |   |--- education-num <= 12.50
|   |   |   |--- weights: [3439.00, 114.00] class: 0
|   |   |--- education-num >  12.50
|   |   |   |--- weights: [825.00, 155.00] class: 0
|   |--- capital-gain >  7139.50
|   |   |--- capital-gain <= 8028.50
|   |   |   |--- weights: [3.00, 5.00] class: 1
|   |   |--- capital-gain >  8028.50
|   |   |   |--- weights: [0.00, 75.00] class: 1
|--- marital-status_Married-civ-spouse >  0.50
|   |--- education-num <= 12.50
|   |   |--- capital-gain <= 5095.50
|   |   |   |--- weights: [2046.00, 927.00] class: 0
|   |   |--- capital-gain >  5095.50
|   |   |   |--- weights: [3.00, 135.00] class: 1
|   |--- education-num >  12.50
|   |   |--- capital-gain <= 5095.50
|   |   |   |--- weights: [356.00, 816.00] class: 1
|   |   |--- capital-gain >  5095.50
|   |   |   |--- weights: [0.00, 190.00] class: 1
~~~
</details>

`show_weights=True` adds each leaf's **counts**: `weights: [a, b]` means `a` training people earn `<=50K`
and `b` earn `>50K` in that leaf. Compare the splits with the Step 5 tree: they are **the same tree**. White
people are 86% of the rows, so "the" rule of Step 5 is, to a large degree, the majority group's rule.

```python
black_tree, black_rows = subgroup_tree("race_Black")
print(export_text(black_tree, feature_names=names,
                  show_weights=True))
```

<details><summary>Expected Output</summary>

~~~text
race_Black -- 922 training rows, 0.131 earn >50K
test accuracy: 0.9  own baseline: 0.872
|--- marital-status_Married-civ-spouse <= 0.50
|   |--- capital-gain <= 7731.50
|   |   |--- education-num <= 13.50
|   |   |   |--- weights: [623.00, 11.00] class: 0
|   |   |--- education-num >  13.50
|   |   |   |--- weights: [9.00, 5.00] class: 0
|   |--- capital-gain >  7731.50
|   |   |--- weights: [0.00, 9.00] class: 1
|--- marital-status_Married-civ-spouse >  0.50
|   |--- capital-gain <= 5095.50
|   |   |--- education-num <= 9.50
|   |   |   |--- weights: [118.00, 33.00] class: 0
|   |   |--- education-num >  9.50
|   |   |   |--- weights: [51.00, 47.00] class: 0
|   |--- capital-gain >  5095.50
|   |   |--- weights: [0.00, 16.00] class: 1
~~~
</details>

> **Read it:** the first question is the same (married or not), and so is the unmarried side's capital-gain
> question. The married side changes order: it asks about **capital gain before education**. Against its
> own baseline, the Black tree gains far less (0.9 vs 0.872, under 3 points) than the White tree (0.821
> vs 0.736, about 8.5 points). Now read the **counts**, not just the classes. Every married leaf without a
> capital gain says `class: 0` -- yet the share earning `>50K` goes from 33 of 151 (22%) at
> `education-num` 9 or less to 47 of 98 (48%) above it. Education matters here too; it just stops short of
> the 50% that would flip the label. **A leaf's class label hides its share -- read the counts.** And be
> careful comparing leaves *across* the two trees: they cut education in different places (9.5 here, 12.5
> in the White tree), so their leaves hold different people. Compare like with like, or not at all.

To compare the two rules at a glance, read the trees' own `feature_importances_` (L14) -- for a single tree,
it is how much each split it made reduced the mixing of classes:

```python
importance_by_group = pd.DataFrame({
    "White": white_tree.feature_importances_,
    "Black": black_tree.feature_importances_,
}, index=names)
used = (importance_by_group > 0).any(axis=1)
print(importance_by_group[used].round(2))
```

<details><summary>Expected Output</summary>

~~~text
                                   White  Black
marital-status_Married-civ-spouse   0.53   0.49
education-num                       0.25   0.14
capital-gain                        0.22   0.38
~~~
</details>

Two cautions before reading meaning into it. An importance of 0 means "this three-question tree never asked
about it", not "irrelevant". And inside a group, the group's own column is constant (every row in the Black
tree has `race_Black` = 1), so a group's tree can never split on it. The table suggests education matters
less in the Black tree (0.14 vs 0.25) and capital gain more. Is that real?

### 7c: Is the Difference Real?

A depth-3 tree grown on 922 people (121 of them `>50K`) may not be a stable object. Test it the way a forest
builds its trees (L14): draw a **random resample** of the group's training rows -- the same number of rows,
drawn **with replacement** -- refit, and see which questions survive. `export_text` counts the root as depth
0, so `max_depth=1` prints the root and the questions right under it; "truncated branch" marks the deeper
questions it hides.

```python
def stability_check(rows, seeds=(0, 1, 2)):
    """Refit on resamples; print the first two levels."""
    X_group = X_train[rows]
    y_group = np.asarray(y_train)[rows]
    for seed in seeds:
        rng = np.random.default_rng(seed)
        pick = rng.choice(len(y_group), size=len(y_group),
                          replace=True)
        tree_r = DecisionTreeClassifier(max_depth=3,
                                        random_state=42)
        tree_r.fit(X_group[pick], y_group[pick])
        print("resample", seed)
        print(export_text(tree_r, feature_names=names,
                          max_depth=1))

stability_check(black_rows)
```

<details><summary>Expected Output</summary>

~~~text
resample 0
|--- marital-status_Married-civ-spouse <= 0.50
|   |--- capital-gain <= 10758.50
|   |   |--- truncated branch of depth 2
|   |--- capital-gain >  10758.50
|   |   |--- class: 1
|--- marital-status_Married-civ-spouse >  0.50
|   |--- capital-gain <= 5095.50
|   |   |--- truncated branch of depth 2
|   |--- capital-gain >  5095.50
|   |   |--- class: 1

resample 1
|--- marital-status_Married-civ-spouse <= 0.50
|   |--- capital-gain <= 7731.50
|   |   |--- truncated branch of depth 2
|   |--- capital-gain >  7731.50
|   |   |--- class: 1
|--- marital-status_Married-civ-spouse >  0.50
|   |--- age <= 34.50
|   |   |--- truncated branch of depth 2
|   |--- age >  34.50
|   |   |--- truncated branch of depth 2

resample 2
|--- marital-status_Married-civ-spouse <= 0.50
|   |--- capital-gain <= 7731.50
|   |   |--- truncated branch of depth 2
|   |--- capital-gain >  7731.50
|   |   |--- class: 1
|--- marital-status_Married-civ-spouse >  0.50
|   |--- capital-gain <= 4479.50
|   |   |--- truncated branch of depth 2
|   |--- capital-gain >  4479.50
|   |   |--- class: 1
~~~
</details>

> **Read it:** in all three resamples the **first question survives** (married or not), and the unmarried
> side always asks about capital gain -- though its cut moves (10758.5, 7731.5, 7731.5). The married side
> does not hold still: capital gain, then **age**, then capital gain again. `education-num` never reaches
> the second level. So the root is solid, and the second level -- exactly where the Black and White trees
> differed -- is fragile at this sample size.

A forest does this resampling 200 times and averages. So the forest's importance, fit separately per group,
is the steadier way to compare what each group's model leans on:

```python
def forest_importance(groups):
    """Forest importances, one column per group."""
    table = {}
    for label, rows in groups.items():
        forest_g = RandomForestClassifier(n_estimators=200,
                                          random_state=42)
        forest_g.fit(X_train[rows], y_train[rows])
        table[label] = forest_g.feature_importances_
    return pd.DataFrame(table, index=names)

race_forests = forest_importance({"White": white_rows,
                                  "Black": black_rows})
top = race_forests.sort_values("White", ascending=False,
                               kind="stable")
print(top.head(8).round(3))
```

<details><summary>Expected Output</summary>

~~~text
                                   White  Black
age                                0.236  0.181
hours-per-week                     0.121  0.077
capital-gain                       0.088  0.093
education-num                      0.074  0.071
marital-status_Married-civ-spouse  0.067  0.073
relationship_Husband               0.038  0.058
capital-loss                       0.033  0.012
marital-status_Never-married       0.026  0.025
~~~
</details>

> **Read it:** the two forests lean on the **same features in nearly the same order**, and the education gap
> from the single trees (0.25 vs 0.14) is gone: `education-num` scores 0.074 vs 0.071. The forests also
> crown `age` and `hours-per-week`, which never appear in a depth-3 tree. They help in many small
> refinements deep inside the forest's trees, and this kind of importance favors columns with many possible
> cut points -- so a forest's ranking and a shallow tree's first questions answer different questions. And
> `relationship_Husband` takes a share in both groups next to `marital-status_Married-civ-spouse`: two
> overlapping columns splitting the credit (L14).

**What is honest to conclude?** In this 1994 sample, the same few questions -- marriage, capital gains,
education, age, hours -- drive the prediction for both groups, and the one difference the single trees
showed (education) does not survive the forest. The **big** difference is the one Step 7a showed before any
model: 26.6% of White and 13.1% of Black training rows earn `>50K`. The counts *describe* that gap; no tree
can *explain* it -- why the base rates differ is a question about history, jobs, and opportunity that
belongs to domain knowledge and to **L27**, not to the tree.

## Your Turn

**1. The redundant twin.** `education` (one-hot) and `education-num` carry the same information. Re-run the
depth-3 tree using `X` with the `education_*` one-hot columns dropped (keep `education-num`). Does test
accuracy move? What does that tell you about redundant features?

```python
# Your Turn 1: drop the education_* one-hot columns, refit the depth-3 tree, compare accuracy.
# Use NEW names (e.g. X_noedu, Xtr_noedu); do not overwrite X_train / X_test -- Your Turn 3 needs them.
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

**3. Male vs Female.** Run the Step 7 analysis on the sex split, using the three helpers. Then answer:

- **(a)** What is the *first* question each tree asks? Look it up in Know the Columns. What does that say
  about the Step 5 tree, which never splits on `sex`?
- **(b)** Which tree has the higher test accuracy, and which one *learned more*? (Compare each with its own
  baseline.)
- **(c)** In the married branch, where does each tree cut `education-num`, and what level of schooling does
  each cut mean? Run `stability_check` on the Female rows: does the Female tree's cut survive resampling?
- **(d)** What do the forests say about `education-num` in the two groups? Before you explain any
  difference, recall *who* the married women in this table are (Know the Columns).

```python
# Your Turn 3: uncomment and run, then answer (a)-(d).
# The helpers use Step 3's X_train / X_test (102 columns).
# male_tree, male_rows = subgroup_tree("sex_Male")
# print(export_text(male_tree, feature_names=names,
#                   show_weights=True))
# female_tree, female_rows = subgroup_tree("sex_Female")
# print(export_text(female_tree, feature_names=names,
#                   show_weights=True))
# stability_check(female_rows)
# sex_forests = forest_importance({"Male": male_rows,
#                                  "Female": female_rows})
# top = sex_forests.sort_values("Female", ascending=False,
#                               kind="stable")
# print(top.head(8).round(3))
```

<details><summary>Answer</summary>

**(a)** The Male tree's first question is `relationship_Husband`; the Female tree's is `relationship_Wife`.
Both mean "married, and of this sex" -- so the question the Step 5 tree asked first (married or not) is,
inside each sex, the same split under a sex-coded name. And the Step 5 tree's first question carries sex
too: in this table 6,334 of the 7,087 `Married-civ-spouse` people (89.4%) are men, against 48.2% of everyone
else (check with `pd.crosstab(df["marital-status"], df["sex"])`). So "married or not" is also, to a large
degree, a question about sex. The Step 5 tree never asks about `sex`, but other columns carry it anyway: a
**proxy**. "The tree didn't use it" is not "the data didn't carry it".

**(b)** The Female tree has the higher accuracy (0.922 vs 0.79) but **learned less**: its own baseline is
0.871 (87.1% of the women in the test rows earn `<=50K`), so it gains about 5 points; the Male tree gains
about 10 (0.79 vs 0.691). Comparing raw accuracies across groups with different base rates rewards the
group that was easiest to guess.

**(c)** Men: married with `education-num` above 12.5 -- a bachelor's degree or higher -- and that cut
holds in every resample of the Male rows (try `stability_check(male_rows)`). Women: the married side cuts at
9.5 -- more than high school. But `stability_check(female_rows)` keeps the first question
(`relationship_Wife`) in all three resamples while the married side changes every time: capital gain, then
`education-num` at 9.5, then `occupation_Other-service`. The 9.5 cut is not a stable finding. Read the
Female tree's small leaves the same way: the `education_Doctorate` split rests on 21 women, and the `age`
split peels off a leaf of just 35.

**(d)** The forests give `education-num` 0.078 (Male) vs 0.054 (Female) -- a gap that, unlike the race
comparison's, survives the averaging. The Female forest also leans
on `relationship_Wife` (0.091) and on being married (0.075). Before explaining that as "education pays off
less for women", recall Know the Columns: only 15.5% of the women here are married, the documented
extraction conditions do not fully explain why, and the data is from 1994. The married women in this table
are a small, selected group -- the trees describe that table, not women in general.
</details>

**4. Written synthesis (3-5 sentences).** You must hand a policymaker a model and a one-paragraph summary.
Which model do you ship and why? And what is the single most important caveat about concluding "X causes
higher income" from this analysis? (Consider: associational vs causal, the data is from 1994, the sample is
bounded, some features are sensitive proxies -- and what Step 7 and Your Turn 3 showed about groups.)

## Recap

You ran the entire Unit IV loop on a fresh dataset: **engineered** a 102-column feature matrix (one-hot the
categoricals; *skip* scaling because trees are scale-invariant; drop a non-feature), **selected** the
drivers with a mutual-information filter and forest importance that agreed, **trained** a readable depth-3
tree (82.9%) that matched a forest (83.2%) and beat the 74.9% baseline, and **interpreted** the result
honestly -- which model to ship, what the rule says, and the fairness question the features raise. Then you
**checked the rule group by group**: read the codebook before reading a split, counted the rarer class to
see which groups can support a tree, judged each group's tree against its own baseline, read leaf counts
instead of class labels, refit on resamples to see which questions survive, saw that a column like
`relationship` can carry `sex` (a proxy), and trusted the forest's average over one shallow tree when the
two disagreed.
That last move -- deciding what is honest to conclude -- is the skill the whole unit was building toward.
Unit V turns from *mining* patterns to *seeing* them: visualization.
