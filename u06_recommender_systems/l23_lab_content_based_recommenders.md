---
title: "Lab: The Other Recommenders, and How to Score Them"
unit: "VI"
lesson: "23"
type: lab
tags: [recommender-systems, content-based, demographic, knowledge-based, hybrid, evaluation, rmse]
difficulty: intermediate
duration: "45 mins"
---

**Goal:** pick up where L22 stopped. You will build the three recommenders that
need no rating history from the target user -- **content-based** (genre
features, starting with Ana from L22's toy), **demographic** (cohorts) and
**knowledge-based** (rules) -- then answer L22's open question by **measuring**
L22's models -- its baseline, an item-item predictor built from its similarity
`S`, and its matrix factorization -- with a train/test split, RMSE and MAE: an
**accuracy ladder**. Pairs with the concept note
[Content-, Knowledge- & Demographic-Based Recommenders](l23_concept_content_based_recommenders.qmd).

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/devomh/cdat4001-2026/blob/main/u06_recommender_systems/l23_lab_content_based_recommenders.ipynb)

> This page is the read-only view. To run the lab, open the notebook
> (`l23_lab_content_based_recommenders.ipynb`) -- in Colab via the badge above, or
> locally. Needs network access the first time (it downloads MovieLens 100k).
> Every text output below comes from actually running the cells.

## Prerequisites & Setup

The same **MovieLens 100k** as L22 (GroupLens, U. Minnesota). Its license forbids
redistribution, so the notebook downloads it at run time and checks its md5 -- we do
not bundle it. This lesson uses the columns L22 ignored: the 19 **genre flags** in
`u.item` (content-based) and the **age/gender** in `u.user` (demographic).

```python
# Setup cell 1 of 2: install only (Colab resets on open; run this first)
%pip install -q pandas numpy matplotlib scikit-learn
```

```python
# Setup cell 2 of 2: imports + download + load. Safe to re-run (skips the download once present).
import os, hashlib, urllib.request, zipfile
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

URL = "https://files.grouplens.org/datasets/movielens/ml-100k.zip"
MD5 = "0e33842e24a9c977be4e0107933c0723"
os.makedirs("data", exist_ok=True)
if not os.path.exists("data/ml-100k/u.data"):
    urllib.request.urlretrieve(URL, "data/ml-100k.zip")
    assert hashlib.md5(open("data/ml-100k.zip", "rb").read()).hexdigest() == MD5, "checksum mismatch"
    with zipfile.ZipFile("data/ml-100k.zip") as z:
        z.extractall("data")

ratings = pd.read_csv("data/ml-100k/u.data", sep="\t", names=["user", "item", "rating", "ts"])
users = pd.read_csv("data/ml-100k/u.user", sep="|", names=["user", "age", "gender", "occupation", "zip"])
GENRES = ["unknown", "Action", "Adventure", "Animation", "Children", "Comedy", "Crime",
          "Documentary", "Drama", "Fantasy", "Film-Noir", "Horror", "Musical", "Mystery",
          "Romance", "Sci-Fi", "Thriller", "War", "Western"]
items = pd.read_csv("data/ml-100k/u.item", sep="|", encoding="latin-1",
                    names=["item", "title", "release", "video", "imdb"] + GENRES)
title = items.set_index("item")["title"]   # movie id -> title, used everywhere below
counts = ratings.groupby("item").rating.count()
print("MovieLens 100k -- GroupLens / U. Minnesota (Harper & Konstan 2015); runtime download, not redistributed.")
print("ratings:", ratings.shape, "| users:", users.shape, "| items:", items.shape)
```

<details><summary>Expected Output</summary>

~~~text
MovieLens 100k -- GroupLens / U. Minnesota (Harper & Konstan 2015); runtime download, not redistributed.
ratings: (100000, 4) | users: (943, 5) | items: (1682, 24)
~~~
</details>

## Step 1: Content-Based Filtering (Worked)

Collaborative filtering never asked what a movie *is*. **Content-based filtering**
does: each movie is a 0/1 vector over the 19 genres, and a user's **content
profile** adds up the genre vectors of the movies they rated, each weighted by how
far that rating sat from the user's own average. Genres they like come out
positive, genres they dislike negative. Then every unseen movie is scored by the
**cosine** (L18) between its genre vector and that profile.

### Ana again

Start with L22's toy. Ana rated Terminator 5, Top Gun 4, Toy Story 1 and Lion King
2 (her average is 3). Here are the real MovieLens genre rows of those movies, of
the two she has not seen (Speed and Aladdin), and of **Die Hard** -- think of it
as a movie added to the catalog today, so in the toy **nobody has rated it**.

```python
toy_titles = {"Terminator": "Terminator, The (1984)", "TopGun": "Top Gun (1986)",
              "ToyStory": "Toy Story (1995)", "LionKing": "Lion King, The (1994)",
              "Speed": "Speed (1994)", "Aladdin": "Aladdin (1992)", "DieHard": "Die Hard (1988)"}
toy_G = items.set_index("title").loc[list(toy_titles.values()), GENRES]
toy_G.index = list(toy_titles)
toy_G = toy_G.loc[:, toy_G.any()]                  # keep only the genres these movies use
print(toy_G.to_string())

ana = pd.Series({"Terminator": 5, "TopGun": 4, "ToyStory": 1, "LionKing": 2})
ana_profile = (ana - ana.mean()) @ toy_G.loc[ana.index]   # centered ratings weight the genre rows
print()
print(ana_profile.astype(int).to_frame("Ana's profile").T.to_string())

def cosine(a, b):
    return (a @ b) / (np.linalg.norm(a) * np.linalg.norm(b))

for movie in ["Speed", "Aladdin", "DieHard"]:
    print(f"cos(Ana, {movie}) = {cosine(ana_profile.values, toy_G.loc[movie].values):+.3f}")
```

<details><summary>Expected Output</summary>

~~~text
            Action  Animation  Children  Comedy  Musical  Romance  Sci-Fi  Thriller
Terminator       1          0         0       0        0        0       1         1
TopGun           1          0         0       0        0        1       0         0
ToyStory         0          1         1       1        0        0       0         0
LionKing         0          1         1       0        1        0       0         0
Speed            1          0         0       0        0        1       0         1
Aladdin          0          1         1       1        1        0       0         0
DieHard          1          0         0       0        0        0       0         1

               Action  Animation  Children  Comedy  Musical  Romance  Sci-Fi  Thriller
Ana's profile       3         -3        -3      -2       -1        1       2         2
cos(Ana, Speed) = +0.541
cos(Ana, Aladdin) = -0.703
cos(Ana, DieHard) = +0.552
~~~
</details>

> **Read it:** Ana's two loved movies push Action, Sci-Fi and Thriller up; her two
> disliked ones push Animation, Children, Comedy and Musical down. The profile has
> negative entries because we weighted by **centered** ratings -- the same move
> that took L22's taste arrows out of the positive corner. Speed (an action movie)
> points along her profile and Aladdin points away: the same verdict L22's
> collaborative filtering reached, but **without looking at any other user**. And
> Die Hard, which nobody has rated, still gets a score at once. Collaborative
> filtering has no ratings for it to compare; content-based only needs its genres.

### Now a real user

The same recipe for user 196 on the full 1682 x 19 genre matrix.

```python
G = items.set_index("item").reindex(range(1, 1683))[GENRES].values.astype(float)   # 1682 x 19 genre matrix
gnorm = np.sqrt((G**2).sum(axis=1)); gnorm[gnorm == 0] = 1e-9

def content_scores(uid):
    seen = ratings[ratings.user == uid]
    profile = (seen.rating - seen.rating.mean()).values @ G[seen.item.values - 1]   # 19-dim taste profile
    pnorm = np.sqrt((profile**2).sum()) or 1e-9
    sims = (G @ profile) / (gnorm * pnorm)
    sims[seen.item.values - 1] = -np.inf                                            # drop seen
    return np.round(sims, 6), profile                                               # round: exact ties

sims_196, profile_196 = content_scores(196)
top_g = np.argsort(-profile_196, kind="stable")[:3]
print("user 196's strongest genres:", [(GENRES[k], round(float(profile_196[k]), 1)) for k in top_g])
```

<details><summary>Expected Output</summary>

~~~text
user 196's strongest genres: [('Drama', 5.2), ('Children', 0.8), ('Documentary', 0.4)]
~~~
</details>

### Naive approach: rank by the cosine alone   

```python
best = sims_196.max()
print(f"{int((sims_196 == best).sum())} unseen movies tie at the top score {best:.3f}")
print("first five of them (by item id):")
for j in np.argsort(-sims_196, kind="stable")[:5]:
    print(f"{sims_196[j]:.3f}  {title.loc[j + 1]}  ({int(counts.get(j + 1, 0))} ratings)")
```

<details><summary>Expected Output</summary>

~~~text
372 unseen movies tie at the top score 0.555
first five of them (by item id):
0.555  Shanghai Triad (Yao a yao yao dao waipo qiao) (1995)  (26 ratings)
0.555  Dead Man Walking (1995)  (299 ratings)
0.555  Mr. Holland's Opus (1995)  (293 ratings)
0.555  White Balloon, The (1995)  (10 ratings)
0.555  Antonia's Line (1995)  (69 ratings)
~~~
</details>

> **Read it:** genres are a **coarse** feature. User 196's profile is dominated
> by Drama, so *every* movie tagged only Drama has the same genre vector and the
> same cosine -- the best one possible for this profile. The cosine cannot choose
> among them; "first by item id" is an accident of the file, and it hands out
> movies with 10 or 26 ratings as readily as famous ones.

### Better: break the tie with a stated quality signal   

Keep the cosine as the first key, and among equal cosines prefer movies many
people rated -- and say so. Here we also require at least 50 ratings.

```python
def content_recommend(uid, n=5, min_count=50):
    sims, _ = content_scores(uid)
    cand = [j for j in range(len(sims)) if counts.get(j + 1, 0) >= min_count and np.isfinite(sims[j])]
    cand.sort(key=lambda j: (-sims[j], -counts.get(j + 1, 0)))                      # cosine, then most rated
    return [(title.loc[j + 1], sims[j]) for j in cand[:n]]

print("top-5 content-based recommendations for user 196 (>= 50 ratings, ties -> most rated):")
for t, s in content_recommend(196):
    print(f"{s:.3f}  {t}")
```

<details><summary>Expected Output</summary>

~~~text
top-5 content-based recommendations for user 196 (>= 50 ratings, ties -> most rated):
0.555  Dead Man Walking (1995)
0.555  Mr. Holland's Opus (1995)
0.555  Shawshank Redemption, The (1994)
0.555  One Flew Over the Cuckoo's Nest (1975)
0.555  Dead Poets Society (1989)
~~~
</details>

> **Interpretation:** every pick is a **Drama** -- user 196's dominant genre and
> nothing else. That is **over-specialization**: content-based keeps serving the
> same flavor. Compare L22's guarded user-user list for the same user: Shawshank,
> My Life as a Dog, Remains of the Day, The Usual Suspects and Das Boot -- four of
> five are Dramas too, and Shawshank is on both lists. Two independent signals
> agree about user 196. The difference is *how* they got there: collaborative
> filtering went by **who** liked a movie (so a Crime/Thriller like The Usual
> Suspects gets in), content-based by **what** the movie is. One caution: the
> quality filter we just added (`min_count=50`) is exactly what would hide a
> brand-new movie. Ana's Die Hard got a score; this list could never show it.

## Step 2: Demographic Filtering (Worked)

When you know nothing about a user's ratings, you can still use *who they are*.
**Demographic filtering** recommends what people in the same group liked. MovieLens
ships real demographics in `u.user`, so we can ask what women aged 25-34 rate
highest -- and, to see whether the cohort changes anything, what *everyone* rates
highest under the same rule.

```python
dem = ratings.merge(users[["user", "gender", "age"]], on="user")
dem["age_group"] = pd.cut(dem.age, [0, 25, 35, 50, 100], right=False,     # [25, 35) = 25-34
                          labels=["<25", "25-34", "35-49", "50+"])

def top_rated(df, n=5, min_count=20):
    return (df.groupby("item").rating.agg(["count", "mean"])
            .query("count >= @min_count").sort_values("mean", ascending=False, kind="stable")
            .head(n).join(title))

cohort = dem[(dem.gender == "F") & (dem.age_group == "25-34")]
print("Top movies among women aged 25-34:")
print(top_rated(cohort)[["title", "mean", "count"]].round(2).to_string(index=False))
print("\nTop movies among all users (same rule):")
print(top_rated(dem)[["title", "mean", "count"]].round(2).to_string(index=False))
```

<details><summary>Expected Output</summary>

~~~text
Top movies among women aged 25-34:
                           title  mean  count
      Usual Suspects, The (1995)  4.71     21
         Schindler's List (1993)  4.61     23
Silence of the Lambs, The (1991)  4.46     37
   Much Ado About Nothing (1993)  4.40     20
               Casablanca (1942)  4.40     20

Top movies among all users (same rule):
                                                 title  mean  count
                                 Close Shave, A (1995)  4.49    112
                               Schindler's List (1993)  4.47    298
                            Wrong Trousers, The (1993)  4.47    118
                                     Casablanca (1942)  4.46    243
Wallace & Gromit: The Best of Aardman Animation (1996)  4.45     67
~~~
</details>

> **Interpretation:** the all-users list is L22's Step 1 crowd-pleaser list; the
> cohort list differs from it. No ratings from the target user were needed -- only
> their age and gender -- which is how demographic filtering *can* greet a
> brand-new user. Whether the cohort list actually **predicts** a new user's
> ratings better than the all-users list is a separate question, and one you can
> measure: Exercise 2.

## Step 3: Knowledge-Based Filtering (Worked + Completion)

A **knowledge-based** recommender works from stated **rules**, not history. We parse
the release year from each title and filter the catalog: a highly-rated Comedy from
the 1990s, with enough ratings to trust. Notice the rule has two kinds of
constraint. Genre and decade are what the user **states**; they come from the
movie's own attributes. The mean and count are a **quality filter borrowed from
ratings**.

```python
items["year"] = items.title.str.extract(r"\((\d{4})\)")[0].astype(float)
stats = ratings.groupby("item").rating.agg(["count", "mean"])
catalog = items.set_index("item").join(stats)

stated = (catalog.Comedy == 1) & catalog.year.between(1990, 1999)          # what the user asked for
quality = (catalog["mean"] >= 4.0) & (catalog["count"] >= 50)              # borrowed from ratings
print(f"1990s Comedies: {int(stated.sum())} | also pass the quality filter: {int((stated & quality).sum())}")

rule = catalog[stated & quality].astype({"year": int})
print("Highly-rated 1990s Comedies (mean>=4.0, >=50 ratings):")
print(rule.sort_values("mean", ascending=False, kind="stable")[["title", "year", "mean", "count"]]
      .head(5).round(2).to_string(index=False))
```

<details><summary>Expected Output</summary>

~~~text
1990s Comedies: 416 | also pass the quality filter: 7
Highly-rated 1990s Comedies (mean>=4.0, >=50 ratings):
                        title  year  mean  count
        Close Shave, A (1995)  1995  4.49    112
   Wrong Trousers, The (1993)  1993  4.47    118
    As Good As It Gets (1997)  1997  4.20    112
      Grand Day Out, A (1992)  1992  4.11     66
Much Ado About Nothing (1993)  1993  4.06    176
~~~
</details>

> **Read it:** the stated constraints alone would accept a Comedy added to the
> catalog today -- genre and year are known from day one. The quality filter is
> what drops it: a movie with no ratings has no mean and no count. Knowledge-based
> systems in the wild (a laptop finder, a travel site) mostly filter on item
> attributes such as price, size or dates, which is why they work on new items.

**Your completion problem:** change one constraint to find highly-rated 1990s
**Dramas** instead of Comedies. Uncomment, replace the `____` with the genre column
name, and run.

```python
# stated_d = (catalog.____ == 1) & catalog.year.between(1990, 1999)
# rule_d = catalog[stated_d & quality].astype({"year": int})
# print(rule_d.sort_values("mean", ascending=False, kind="stable")[["title", "year", "mean", "count"]]
#       .head(5).round(2).to_string(index=False))
```

<details><summary>Expected Output (when completed correctly)</summary>

~~~text
                           title  year  mean  count
         Schindler's List (1993)  1993  4.47    298
Shawshank Redemption, The (1994)  1994  4.45    283
Silence of the Lambs, The (1991)  1991  4.29    390
           Secrets & Lies (1996)  1996  4.27    162
        Good Will Hunting (1997)  1997  4.26    198
~~~
</details>

## Step 4: Evaluation -- the Accuracy Ladder (Worked)

Now L22's open question: are these any good? **Hide** some ratings (a train/test
split, as in L14), **predict** them, and **measure** the error with **RMSE**
(punishes big misses) and **MAE** (average miss). Predictions are clipped to the
1-5 scale first, since no rating can fall outside it.

```python
train, test = train_test_split(ratings, test_size=0.2, random_state=42)

def rmse_mae(pred, true):
    pred = np.clip(pred, 1, 5)                 # ratings live on a 1-5 scale
    err = pred - true
    return np.sqrt((err**2).mean()), np.abs(err).mean()

true = test.rating.values
gmean = train.rating.mean()                                    # mu: the global mean
item_mean = train.groupby("item").rating.mean()
rmse_g, mae_g = rmse_mae(np.full(len(test), gmean), true)
rmse_i, mae_i = rmse_mae(test.item.map(item_mean).fillna(gmean).values, true)
print(f"global mean:          RMSE={rmse_g:.3f}  MAE={mae_g:.3f}")
print(f"item mean:            RMSE={rmse_i:.3f}  MAE={mae_i:.3f}")
```

<details><summary>Expected Output</summary>

~~~text
global mean:          RMSE=1.124  MAE=0.942
item mean:            RMSE=1.021  MAE=0.812
~~~
</details>

The item mean is the number behind L22's Step 1 popularity list: "how well-liked is
this movie?". The next rung is L22's **baseline**, $\mu + b_u + b_i$: also ask how
generous *this user* is. Here we estimate the two biases with plain averages; matrix
factorization below learns its own `bu` and `bi`.

```python
item_bias = (train.rating - gmean).groupby(train.item).mean()                          # b_i
user_bias = (train.rating - gmean - train.item.map(item_bias)).groupby(train.user).mean()   # b_u
pred_base = gmean + test.item.map(item_bias).fillna(0).values + test.user.map(user_bias).fillna(0).values
rmse_b, mae_b = rmse_mae(pred_base, true)
print(f"baseline mu+b_u+b_i:  RMSE={rmse_b:.3f}  MAE={mae_b:.3f}")
```

<details><summary>Expected Output</summary>

~~~text
baseline mu+b_u+b_i:  RMSE=0.946  MAE=0.745
~~~
</details>

Then **item-item collaborative filtering**. L22 built the item-item similarity `S`
(its Step 3) and a prediction formula for users (its Step 2: the user's mean plus
similarity-weighted centered ratings). Turn that formula sideways and you have an
item-based predictor: user $u$'s mean, plus $u$'s own centered ratings of the
other movies, weighted by how similar each is to the movie being predicted. Same
code as L22, built on the training ratings only.

```python
Rv = (train.pivot(index="user", columns="item", values="rating")
      .reindex(index=range(1, 944), columns=range(1, 1683)).values)
rated = ~np.isnan(Rv)
user_mean = np.nanmean(Rv, axis=1)                    # every user has training ratings
Rc = np.where(rated, Rv - user_mean[:, None], 0.0)    # mean-centered; unrated -> 0
item_norm = np.sqrt((Rc**2).sum(axis=0)); item_norm[item_norm == 0] = 1e-9
S = (Rc.T @ Rc) / np.outer(item_norm, item_norm); np.fill_diagonal(S, 0.0)
num = Rc @ S                                          # similarity-weighted centered ratings
den = rated.astype(float) @ np.abs(S)
pred_all = user_mean[:, None] + np.divide(num, den, out=np.zeros_like(num), where=den > 0)
pred_cf = pred_all[test.user.values - 1, test.item.values - 1]
rmse_cf, mae_cf = rmse_mae(pred_cf, true)
print(f"item-item CF:         RMSE={rmse_cf:.3f}  MAE={mae_cf:.3f}")
```

<details><summary>Expected Output</summary>

~~~text
item-item CF:         RMSE=0.945  MAE=0.743
~~~
</details>

Finally **matrix factorization**: L22's Step 4 model, line for line, now trained on
the training split only (about 15 seconds on a laptop, a little longer on Colab).

```python
user_ids = {u: k for k, u in enumerate(sorted(ratings.user.unique()))}
item_ids = {i: k for k, i in enumerate(sorted(ratings.item.unique()))}
data_arr = np.array([[user_ids[u], item_ids[i], r]
                     for u, i, r in train[["user", "item", "rating"]].values])

rng = np.random.default_rng(42)
K, lr, reg, epochs = 20, 0.01, 0.05, 20
P = rng.normal(0, 0.1, (len(user_ids), K))      # user latent factors
Q = rng.normal(0, 0.1, (len(item_ids), K))      # movie latent factors
bu = np.zeros(len(user_ids)); bi = np.zeros(len(item_ids))

for epoch in range(epochs):
    rng.shuffle(data_arr)
    for u, i, r in data_arr:
        u, i = int(u), int(i)
        # prediction = baseline (gmean + bu + bi) + factor match P[u] @ Q[i]
        err = r - (gmean + bu[u] + bi[i] + P[u] @ Q[i])
        bu[u] += lr * (err - reg * bu[u]); bi[i] += lr * (err - reg * bi[i])
        Pu = P[u].copy()
        P[u] += lr * (err * Q[i] - reg * P[u])
        Q[i] += lr * (err * Pu - reg * Q[i])

ui = test.user.map(user_ids).values; ii = test.item.map(item_ids).values
pred_mf = gmean + bu[ui] + bi[ii] + (P[ui] * Q[ii]).sum(axis=1)
rmse_mf, mae_mf = rmse_mae(pred_mf, true)
print(f"matrix factorization: RMSE={rmse_mf:.3f}  MAE={mae_mf:.3f}")
```

<details><summary>Expected Output</summary>

~~~text
matrix factorization: RMSE=0.922  MAE=0.725
~~~
</details>

```python
labels = ["global\nmean", "item\nmean", "baseline\nmu+b_u+b_i", "item-item\nCF", "matrix\nfactorization"]
vals = [rmse_g, rmse_i, rmse_b, rmse_cf, rmse_mf]
colors = ["#b0b0b0", "#b0b0b0", "#6b6b6b", "#4878a8", "#3a8a3a"]
plt.figure(figsize=(8, 4))
plt.bar(labels, vals, color=colors, width=0.62)
plt.ylim(0.85, 1.16); plt.ylabel("test RMSE (lower is better)")
plt.title("The baseline does most of the work")
for i, v in enumerate(vals):
    plt.text(i, v + 0.004, f"{v:.3f}", ha="center")
plt.show()
```

> **Interpretation:** read the ladder by the size of each step. Knowing **which
> movie** is the biggest gain (global mean -> item mean); knowing how generous
> **this user** is comes next (item mean -> baseline). Item-item CF then lands
> level with the baseline -- on other random splits it comes out a hair above or
> below it -- so on this data the neighbors add little that the two biases had not
> already captured. Matrix factorization's latent factors buy the last real
> step, about 0.02. That is L22's finding again, now measured on ratings the models never saw:
> the baseline does most of the work, and the factors personalize at the margin.
> (This scores *rating prediction*. How well the **top** of each list is ordered --
> ranking, e.g. precision@k -- is a related but separate question, taken up in
> L24.)

## Your Turn (Exercises)

### Exercise 1 -- A weighted hybrid

Blend two families for user 196: the content-based cosine from Step 1
(`sims_196`, with `-inf` on movies 196 has rated) and the **matrix-factorization
predicted rating** from Step 4 (`gmean + bu[u] + bi + P[u] @ Q.T`). Min-max scale
each to 0-1 over the movies 196 has *not* rated, average them with weight 0.5,
and print the top 5. Then try weights 0.3 and 0.7 for the content part. Does the
weight change the list? Why?

> **Hint:** movie ids run 1-1682 with no gaps, so position `j` is movie `j + 1` in
> both `sims_196` and the MF scores (`item_ids[j + 1] == j`). The user's row is
> `u = user_ids[196]`, as in L22's `recommend_mf`. Use
> `unseen = np.isfinite(sims_196)` as the mask, and sort with
> `np.argsort(-np.round(blend, 6), kind="stable")`.

```python
# TODO: your code here
```

<details><summary>Expected Output (one possible answer -- match the values, not the print format)</summary>

~~~text
content weight 0.5:
  1.000  Shawshank Redemption, The (1994)
  0.987  It's a Wonderful Life (1946)
  0.974  Good Will Hunting (1997)
  0.969  Pather Panchali (1955)
  0.967  To Kill a Mockingbird (1962)
~~~
*(At 0.3 and 0.7 the same five movies come back in the same order; only the
blended scores move. Why: among the movies 196 has not rated, the content score
gives every pure Drama the same top value, so the content part decides which
**shelf** a movie is on and the MF part orders the movies on that shelf. MF alone
would have listed Shawshank, Schindler's List, Wallace & Gromit (Animation), It's
a Wonderful Life and Paradise Lost (Documentary); the content part swaps in
Dramas -- Schindler's List is also tagged War, a genre 196 rated below average.
So this blend narrows the list rather than diversifying it. From a content
weight of about 0.25 upward the five do not change; below that, MF's strong
non-Drama picks climb back in -- Schindler's List at 0.05-0.2, Wallace & Gromit
too at 0.05.)*
</details>

### Exercise 2 -- The new-user test

Step 2 showed that a cohort list *differs* from the all-users list. Does it
**predict** a new user's ratings better? Hold out 20% of the users entirely,
`old_users, new_users = train_test_split(users.user, test_size=0.2, random_state=42)`,
and use only the old users' ratings (from `dem`, Step 2) to build two predictors
for every rating a new user gave:

- **item mean:** the movie's mean rating among old users (the old users' global
  mean for a movie none of them rated) -- the ladder's item-mean rung;
- **demographic:** the movie's mean rating among old users of the same gender and
  age group, if that cohort rated it at least 20 times -- otherwise the item-mean
  prediction.

Score both with `rmse_mae`. Which wins, and what does that say about demographic
filtering?

> **Hint:** `old.groupby(["gender", "age_group", "item"], observed=True).rating.agg(["count", "mean"])`
> gives each cohort's count and mean per movie (`observed=True` skips empty
> cohort combinations); keep the rows with count >= 20. Look each new rating up
> by its three keys, e.g. with
> `.reindex(pd.MultiIndex.from_frame(new[["gender", "age_group", "item"]]))`,
> and fill the gaps with the item-mean prediction.

```python
# TODO: your code here
```

<details><summary>Expected Output (one possible answer -- match the values, not the print format)</summary>

~~~text
new users: 189 | their ratings: 19693
item mean:   RMSE=1.050  MAE=0.831
demographic: RMSE=1.051  MAE=0.831
~~~
*(A tie: knowing a new user's gender and age group does not predict their
ratings better than knowing how well-liked each movie is. Fewer than half of the
new users' ratings even get a cohort mean -- the rest fall back to the item mean --
and where they do, the cohort's mean for a movie differs from everyone's mean
for it by only about 0.14 stars on average. Demographic filtering **can** run on a new user, which is its real
contribution; on MovieLens it does not predict better than the item mean. That is
why many platforms greet a newcomer with popular items plus a few questions --
a knowledge-based start -- and switch to collaborative filtering as ratings
arrive.)*
</details>

### Exercise 3 -- Written: pick the family for each cold start

For each scenario, name the recommender family you would use and why (1 sentence
each): (a) a brand-new user who has rated nothing; (b) a brand-new movie added
today with no ratings; (c) a shopper who states "I want a quiet hatchback under
$20k."

**Your answer:**

<details><summary>One possible answer</summary>

**(a)** A new user has no ratings, so collaborative filtering has no neighbors:
start with **popular** or **demographic** picks, or ask a few questions
(**knowledge-based**), and switch to collaborative filtering once they rate.
**(b)** A new movie has no ratings but it has genres, so **content-based**
filtering can place it at once for users whose profiles match (Ana's Die Hard).
**(c)** The shopper has stated constraints on the item's own attributes (quiet,
hatchback, price), which is exactly what **knowledge-based** filtering uses --
no rating history needed.
</details>

## Summary

- **Content-based** recommends by item features: a user's profile is the
  centered-rating-weighted sum of genre vectors, scored by cosine. Ana's profile
  picked Speed over Aladdin with no other users, and scored a movie nobody had
  rated. Genres are coarse, so scores tie; a stated quality signal breaks the tie.
  It **over-specializes** (all one flavor).
- **Demographic** (cohorts) and **knowledge-based** (rules) need **no rating
  history** from the target user, so they can run where collaborative filtering
  cannot. Running is not the same as predicting well: on MovieLens the cohort
  averages do no better than plain item means (Exercise 2). A knowledge rule's
  stated constraints work on a new item; a quality filter borrowed from ratings
  does not.
- **Hybrid** systems combine families: a content + MF blend narrows by genre and
  orders by predicted rating.
- **Evaluate** with a held-out split. The **ladder** (test RMSE): global mean 1.124
  -> item mean 1.021 -> baseline 0.946 -> item-item CF 0.945 -> matrix
  factorization 0.922. The baseline does most of the work, CF ties it, the latent
  factors add the last small step.
- Next, **L24** is the mini-project: build, evaluate on ratings *and* on rankings,
  and decide what to ship.
