---
title: "Lab: From 2,003 Features to a Rule You Can Read Aloud"
unit: "IV"
lesson: "14"
type: lab
tags: [feature-selection, decision-trees, random-forests, feature-importance, sklearn]
difficulty: introductory
duration: "85 mins"
---

**Goal:** first *see* what a decision tree does, on a two-feature toy you can plot; then hand L13's
chi-square survivors to a model that sees features *together* -- a decision tree you can read aloud and a
random forest -- and measure what selection actually costs against an honest baseline.
Pairs with the concept note [Feature Selection II: Trees & Random Forests](l14_concept_feature_selection_trees.qmd).

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/devomh/cdat4001-2026/blob/main/u04_data_mining/l14_lab_feature_selection_trees.ipynb)

> This page is the read-only view. To run the lab, open the notebook
> (`l14_lab_feature_selection_trees.ipynb`) -- in Colab via the badge above, or locally. Every text output
> below comes from actually running the cells; the figures (Step 1's tree diagram and region map, Step 5's
> importance bar chart) render live in the notebook.

> **Where this sits:** **closes Unit IV.** L13 ranked features model-free; here a model selects them while
> seeing them together. The full engineer -> select -> model -> interpret loop is the mini-project (**L15**).

## Prerequisites & Setup

Run this first. Same dataset and matrix as L12/L13 -- the **SMS Spam Collection** (Almeida, Gomez Hidalgo &
Yamakami 2011; UCI id 228, CC BY 4.0), bundled as `data/sms_spam.csv` with a UCI fallback.

```python
# Setup cell 1 of 2: install only (Colab resets on open; run this first)
%pip install -q pandas numpy matplotlib scikit-learn scipy
```

```python
# Setup cell 2 of 2: imports + rebuild L12/L13's feature matrix X (5574 x 2003) and label y
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.feature_extraction.text import CountVectorizer
from scipy.sparse import hstack, csr_matrix
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.tree import plot_tree
from sklearn.ensemble import RandomForestClassifier

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
```

<details><summary>Expected Output</summary>

~~~text
Feature matrix: (5574, 2003)
~~~
</details>

## Step 1: A Tree You Can See -- Two Features, Two Classes (Worked)

Before 2,003 columns, two. Here are fourteen invented messages, each described by the two features you
handcrafted in L11/L12 -- `digit_count` and `length` -- and labeled ham or spam. With only two features,
every message is a point on a plane, and you can watch the tree decide.

```python
# 14 invented messages; their features are already counted
toy = pd.DataFrame({
    "digit_count": [0, 1, 0, 2, 3, 1, 0,
                    11, 8, 13, 6, 9, 7, 10],
    "length": [25, 40, 90, 130, 60, 180, 200,
               120, 95, 150, 70, 140, 190, 210],
    "label": ["ham"] * 7 + ["spam"] * 5 + ["ham"] * 2,
})
toy_features = ["digit_count", "length"]
X_toy = toy[toy_features]
y_toy = (toy["label"] == "spam").astype(int)
print(f"{(y_toy == 0).sum()} ham, {y_toy.sum()} spam")
print(f"Baseline (always 'ham'): {(y_toy == 0).mean():.1%}")
```

<details><summary>Expected Output</summary>

~~~text
9 ham, 5 spam
Baseline (always 'ham'): 64.3%
~~~
</details>

Fit a tree allowed one question (`max_depth=1`), then one allowed two (`max_depth=2`):

```python
for depth in [1, 2]:
    toy_tree = DecisionTreeClassifier(max_depth=depth,
                                      random_state=42)
    toy_tree.fit(X_toy, y_toy)
    n_right = (toy_tree.predict(X_toy) == y_toy).sum()
    print(f"max_depth={depth}: {n_right} of 14 right")
    print(export_text(toy_tree, feature_names=toy_features))
# the loop ends with the depth-2 tree in toy_tree
```

<details><summary>Expected Output</summary>

~~~text
max_depth=1: 12 of 14 right
|--- digit_count <= 4.50
|   |--- class: 0
|--- digit_count >  4.50
|   |--- class: 1

max_depth=2: 14 of 14 right
|--- digit_count <= 4.50
|   |--- class: 0
|--- digit_count >  4.50
|   |--- length <= 170.00
|   |   |--- class: 1
|   |--- length >  170.00
|   |   |--- class: 0

~~~
</details>

> **Read it** (class 0 = ham, 1 = spam): one question -- *"five or more digits? Then spam"* -- gets 12 of
> 14 right. The two misses are the long, digit-heavy hams (7 and 10 digits; 190 and 210 characters). A
> second question, `length <= 170`, fixes both -- and notice *where* it is asked: only on the many-digits
> side. The few-digits side is already all ham, so the tree leaves it alone, which is why the depth-2 tree
> has three leaves, not four.

Draw it. `plot_tree` turns the depth-2 tree into a diagram. (Reading this page rather than running the
notebook? Both of Step 1's pictures appear side by side in the concept note, under
[Trees Cut the Plane into Rectangles](l14_concept_feature_selection_trees.qmd#trees-cut-the-plane-into-rectangles).)

```python
fig, ax = plt.subplots(figsize=(8, 5))
plot_tree(toy_tree, feature_names=toy_features,
          class_names=["ham", "spam"], impurity=False,
          filled=True, fontsize=11, ax=ax)
plt.show()
```

> **Read it:** each box is a node, showing its question (if it asks one), how many training messages reach
> it (`samples`), how many of those are ham and spam (`value = [ham, spam]`), and the class it would
> predict. The left arrow is the "True" answer, the right arrow "False". The color names the majority
> class -- orange for ham, blue for spam -- and the paler the box, the more mixed it is.

Now the same tree as a map. Ask it to classify every point of a fine grid covering the plane, and paint
each grid cell with its answer. `np.linspace` makes 300 evenly spaced values per axis; `np.meshgrid`
pairs every x with every y (90,000 grid points); `.ravel()` flattens them into the two columns `predict`
expects; and `.reshape(xx.shape)` folds the answers back into the grid for `pcolormesh` to paint:

```python
HAM, SPAM = "#e58139", "#399de5"   # plot_tree's colors

# Ask the tree about every cell of a fine grid
xx, yy = np.meshgrid(np.linspace(-1, 15, 300),
                     np.linspace(0, 230, 300))
grid = pd.DataFrame({"digit_count": xx.ravel(),
                     "length": yy.ravel()})
zz = toy_tree.predict(grid).reshape(xx.shape)

fig, ax = plt.subplots(figsize=(7, 5))
# Paint each grid cell with the class the tree predicts
ax.pcolormesh(xx, yy, zz, alpha=0.25, shading="auto",
              cmap=ListedColormap([HAM, SPAM]))
for label, color, marker in [("ham", HAM, "o"),
                             ("spam", SPAM, "^")]:
    pts = toy[toy["label"] == label]
    ax.scatter(pts["digit_count"], pts["length"],
               c=color, marker=marker, s=80,
               edgecolor="black", label=label)
# The tree's two thresholds (from export_text above)
ax.axvline(4.5, color="black", linestyle="--")
ax.plot([4.5, 15], [170, 170],
        color="black", linestyle="--")
ax.set_xlabel("digit_count")
ax.set_ylabel("length (characters)")
ax.legend(loc="lower right")
plt.show()
```

> **Read it:** each leaf of the tree is one shaded rectangle. The left strip (few digits) is the ham leaf;
> the lower-right block is the spam leaf; the upper-right block is the leaf for long, digit-heavy ham.
> Every split is a straight cut parallel to an axis -- `digit_count <= 4.5` is the vertical dashed line,
> `length <= 170` the horizontal one -- and the horizontal cut *stops* at the vertical one, because that
> question is asked only inside the "many digits" branch. The border between orange and blue is the tree's
> **decision boundary**: a new message gets the class of the rectangle it lands in.

Last question: how good is each feature *on its own*? Give each one a single question (`max_depth=1`):

```python
# Each feature ALONE: the best single question about it
for feature in toy_features:
    stump = DecisionTreeClassifier(max_depth=1,
                                   random_state=42)
    stump.fit(X_toy[[feature]], y_toy)
    acc = stump.score(X_toy[[feature]], y_toy)
    print(f"{feature:<12} alone: {acc:.1%}")
```

<details><summary>Expected Output</summary>

~~~text
digit_count  alone: 85.7%
length       alone: 64.3%
~~~
</details>

> **Read it:** judged by its best single question, `length` scores 64.3% -- exactly the always-ham
> baseline; no single cut on length helps at all. Yet inside the tree, once the digit question has been
> answered, `length` separates the remaining seven messages perfectly. That is the idea behind L13's blind
> spot: what a feature is worth can depend on the answers to other questions, and a tree asks each question
> *given* the answers above it. It is a milder case than L13's exclusive-or, where *both* features look
> useless alone: here `digit_count` is strong by itself, so it earns the first question and opens the door
> for `length`. And on the real messages `length` is strong alone too -- L13's chi-square ranked it second
> -- so the toy isolates the effect rather than copying the data.

## Step 2: An Honest Yardstick -- Baseline and Train/Test Split (Worked)

Back to the 5,574 real messages. Notice that Step 1 graded the toy tree on the very messages it learned
from -- fine for watching a tree work, but not an honest score. Before scoring anything real, two pieces
of discipline. First, the **baseline**, as in Step 1: 86.6% of the real messages are ham, so "predict ham
for everything" is the score to beat. Second, the **train/test split**: every learned step from here on --
selection included -- is fit on 70% of the data and graded on the 30% it never saw.

```python
print(f"Majority-class baseline: predict 'ham' for everything -> {(y == 0).mean():.1%} accuracy")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y)

print("Training messages:", X_train.shape[0])
print("Test messages:    ", X_test.shape[0])
print(f"Spam share -- train: {y_train.mean():.1%}, test: {y_test.mean():.1%}")
```

<details><summary>Expected Output</summary>

~~~text
Majority-class baseline: predict 'ham' for everything -> 86.6% accuracy
Training messages: 3901
Test messages:     1673
Spam share -- train: 13.4%, test: 13.4%
~~~
</details>

> **Read it:** `stratify=y` keeps the spam share identical (13.4%) on both sides -- without it, chance could
> hand the test set a different class mix and quietly distort every accuracy. From here on, nothing learns
> from `X_test`.

## Step 3: Filter, Then a Tree You Can Read Aloud (Worked)

Apply L13's chi-square filter -- **fit on the training split only** (the discipline L13 named) -- keep the
top 30, and hand them to a decision tree with a strict honesty budget: at most three questions per message
(`max_depth=3`).

```python
selector = SelectKBest(chi2, k=30).fit(X_train, y_train)   # fit on TRAIN only
kept = feature_names[selector.get_support()]
X_train_sel = selector.transform(X_train)
X_test_sel = selector.transform(X_test)

tree_sel = DecisionTreeClassifier(max_depth=3, random_state=42)
tree_sel.fit(X_train_sel, y_train)

print(export_text(tree_sel, feature_names=list(kept)))
print(f"Test accuracy: {tree_sel.score(X_test_sel, y_test):.1%}")
```

<details><summary>Expected Output</summary>

~~~text
|--- digit_count <= 4.50
|   |--- www <= 0.50
|   |   |--- mobile <= 0.50
|   |   |   |--- class: 0
|   |   |--- mobile >  0.50
|   |   |   |--- class: 1
|   |--- www >  0.50
|   |   |--- mobile <= 0.50
|   |   |   |--- class: 1
|   |   |--- mobile >  0.50
|   |   |   |--- class: 0
|--- digit_count >  4.50
|   |--- digit_count <= 9.50
|   |   |--- length <= 176.00
|   |   |   |--- class: 1
|   |   |--- length >  176.00
|   |   |   |--- class: 0
|   |--- digit_count >  9.50
|   |   |--- length <= 182.00
|   |   |   |--- class: 1
|   |   |--- length >  182.00
|   |   |   |--- class: 0

Test accuracy: 97.8%
~~~
</details>

> **Read it** (class 0 = ham, 1 = spam): *"Fewer than five digits? Then it's ham -- unless it mentions 'www'
> or 'mobile'. Five or more digits? Then it's spam -- unless it's suspiciously long."* The root split -- the
> single most valuable question among all 30 features -- is `digit_count`, the column you invented with one
> line of `str.count` in L11/L12. And this readable rule scores 97.8% on messages it never saw, against an
> 86.6% baseline. It is Step 1's toy at full size: the same root question, `digit_count <= 4.5`, and on the
> many-digits side `length` again separates the shorter spam from the longer ham.

## Step 4: The Showdown -- 30 Features vs 2,003 (Worked)

What did selection cost? Fit the identical tree on the full matrix and compare.

```python
tree_all = DecisionTreeClassifier(max_depth=3, random_state=42)
tree_all.fit(X_train, y_train)

results = [
    (f"Depth-3 tree, all {X_train.shape[1]:,} features", tree_all.score(X_test, y_test)),
    ("Depth-3 tree, top 30 features", tree_sel.score(X_test_sel, y_test)),
    ("Majority-class baseline", (y_test == 0).mean()),
]
for name, acc in results:
    print(f"{name:<36} {acc:.1%}")
```

<details><summary>Expected Output</summary>

~~~text
Depth-3 tree, all 2,003 features     98.0%
Depth-3 tree, top 30 features        97.8%
Majority-class baseline              86.6%
~~~
</details>

> **Read it:** dropping 1,973 features cost 0.2 percentage points. Thirty columns do the work of two
> thousand, because the signal was concentrated in a handful of features all along.

## Step 5: The Random Forest -- Accuracy and a Second Opinion (Worked + Completion)

A forest grows 200 randomized trees and lets them vote. Collect its accuracy, and its feature-importance
ranking -- a second, independent opinion on which features matter.

```python
forest = RandomForestClassifier(n_estimators=200, random_state=42)
forest.fit(X_train, y_train)
print(f"Random forest, all features: {forest.score(X_test, y_test):.1%}")
```

<details><summary>Expected Output</summary>

~~~text
Random forest, all features: 99.0%
~~~
</details>

> **Read it:** 1.2 points above the top-30 tree and 12.4 above the baseline -- the vote of 200 trees is the
> most accurate model in this lab. Hold the "is that worth it?" question for Exercise 4.

```python
importances = pd.Series(forest.feature_importances_, index=feature_names)
top12 = importances.sort_values(ascending=False).head(12)

fig, ax = plt.subplots(figsize=(7, 5))
top12.sort_values().plot.barh(ax=ax, color="#4c72b0")
ax.set_xlabel("importance (share of the forest's split value)")
ax.set_title("Top 12 features by random-forest importance")
plt.tight_layout()
plt.show()
```

> **Read it:** compare this leaderboard with L13's chi-square ranking: `digit_count` and `length` on top,
> then free, txt, claim, www. The filter and the forest share no machinery -- one is a statistical test, the
> other an average over 200 trees -- yet they crown the same features. When two unrelated methods agree on
> what matters, that agreement is evidence.

The forest can also tell us how much of the matrix it *ignores*:

```python
# COMPLETION: count the features whose importance is below 0.0001 (they contribute
# essentially nothing to the forest's votes). Fill the ____ and uncomment both lines.
# n_ignored = (importances < ____).sum()
# print(f"{n_ignored} of {len(importances)} features have importance below 0.0001")
```

<details><summary>Expected Output (after completing <code>0.0001</code>)</summary>

~~~text
1286 of 2003 features have importance below 0.0001
~~~
*(Nearly two thirds of the matrix is dead weight even to the model that uses all of it -- the forest quietly
performs its own feature selection.)*
</details>

## Your Turn

### Exercise 1 -- Read the regions

Three new messages arrive, described as (digit_count, length): (2, 150), (8, 100) and (12, 200). First
predict each one **by hand** -- walk the depth-2 tree from Step 1, or find its rectangle on the map. Then
check your answers with `toy_tree.predict`.

```python
# Your Turn 1: put the three messages in a DataFrame with
# columns digit_count and length; call toy_tree.predict.
```

<details><summary>Answer (one possible answer -- match the values, not the print format)</summary>

~~~text
[0 1 0]
~~~
*(Ham, spam, ham. Two digits lands in the few-digits strip: ham, however long the message. Eight digits
and 100 characters lands in the spam block. Twelve digits but 200 characters lands in the upper-right
block -- ham, like the two long hams in the toy data.)*
</details>

### Exercise 2 -- Turn the knobs

How sensitive is the result to our choices? Re-run Step 3's select-then-tree pipeline twice: once with
fewer features (`k=10`, depth 3), once with a deeper tree (`max_depth=5`, k=30). Print both test
accuracies and write one sentence comparing them with Step 3's 97.8%.

```python
# Your Turn 2: only the SelectKBest(k=...) and
# DecisionTreeClassifier(max_depth=...) arguments change.
# Use fresh variable names.
```

<details><summary>Answer (one possible answer -- match the values, not the print format)</summary>

~~~text
k= 10, depth=3: 97.8%
k= 30, depth=5: 97.7%
~~~
*(A plateau: a third of the features, or two more questions per message, and the accuracy barely moves
from 97.8%. Once digit_count and a few spam words are in, the rest add almost nothing -- the signal really
is that concentrated.)*
</details>

### Exercise 3 -- Read the mistakes

The top-30 tree gets 97.8% right. Look at what it gets *wrong*: predict on the test set, count the
misclassified messages, and print the first three with their true labels (the first 90 characters is
plenty). One sentence: what fooled the tree?

```python
# Your Turn 3: tree_sel.predict(X_test_sel) gives the
# predictions; y_test.index[y_test.values != y_pred]
# gives the original row numbers of the mistakes.
```

<details><summary>Answer (one possible answer -- match the values, not the print format)</summary>

~~~text
Misclassified: 37 of 1673 test messages
  true=ham: 1Apple/Day=No Doctor. 1Tulsi Leaf/Day=No Cancer. 1Lemon/Day=No Fat. 1Cup Milk/day=No Bone
  true=spam: Your weekly Cool-Mob tones are ready to download !This weeks new Tones include: 1) Crazy F
  true=ham: .Please charge my mobile when you get up in morning.
~~~
*(The tree fails exactly where its features mislead: a chain-letter ham stuffed with digits reads as spam; a
ham containing "mobile" trips the www/mobile branch; a spam ad happens to dodge the thresholds. A model's
mistakes are a map of its features' blind spots.)*
</details>

### Exercise 4 -- Written: argue the case

A teammate looks at Step 5 and says: *"The forest gets 99.0%, the tree only 97.8% -- always ship the
forest."* Argue for or against in 3-4 sentences, using at least one piece of evidence from this lab.
Consider what this unit is named after: extracting *information* from data.

> **Hint:** what can you hand to a boss, an auditor, or a curious user from the depth-3 tree that the
> 200-tree forest cannot produce? And when would that argument flip?

## Summary

| Move | Key command | What you learned |
|------|-------------|------------------|
| A tree you can see | `plot_tree`, `predict` on a grid | Each split is an axis-parallel cut; `length` alone = baseline, decisive after the digit question |
| Honest yardstick | `train_test_split(stratify=y)`, baseline | 86.6% is the score to beat, not 0% |
| Filter then tree | `SelectKBest(k=30)`, `DecisionTreeClassifier(max_depth=3)` | A 3-question rule scores 97.8% |
| The showdown | same tree, both matrices | 30 features do the work of 2,003 (98.0% vs 97.8%) |
| Random forest | `RandomForestClassifier`, `.feature_importances_` | 99.0% accuracy, but the explanation is gone |
| What it ignores | `(importances < 0.0001).sum()` | 1,286 of 2,003 features are dead weight |

Unit IV closes here: you defined features from raw data (L11/L12) and extracted the ones that carry
information (L13/L14). Next, **L15** -- the data-mining mini-project -- runs the whole loop end to end on a
fresh dataset.
