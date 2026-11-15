---
title: "Lab: Building a Movie Recommender with Collaborative Filtering"
unit: "VI"
lesson: "22"
type: lab
tags: [recommender-systems, collaborative-filtering, cosine-similarity, matrix-factorization, movielens]
difficulty: intermediate
duration: "55 mins"
---

**Goal:** build a movie recommender from nothing but a table of ratings -- no
recommender library. You will start with the concept note's five-user toy (worked
by hand), then build the real user-item matrix, find similar users and similar
movies with cosine similarity, guard against flukes, generate top-N
recommendations, and finish with a small matrix-factorization model whose latent
factors are the same idea you met in L18 (`P @ Q.T`) and L20 (PCA). Pairs with the
concept note
[Recommender Systems & Collaborative Filtering](l22_concept_collaborative_filtering.qmd).

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/devomh/cdat4001-2026/blob/main/u06_recommender_systems/l22_lab_collaborative_filtering.ipynb)

> This page is the read-only view. To run the lab, open the notebook
> (`l22_lab_collaborative_filtering.ipynb`) -- in Colab via the badge above, or
> locally. Needs network access the first time (it downloads MovieLens 100k).
> Every text output below comes from actually running the cells.

## Prerequisites & Setup

We use **MovieLens 100k** (GroupLens, University of Minnesota): 100,000 ratings
(1-5 stars) from 943 users on 1,682 movies, plus user demographics and movie
genres. Its license **forbids redistribution**, so -- unlike every other lab in
this course -- we do **not** bundle the data; the notebook downloads it from
GroupLens at run time and checks its md5. Please use it for learning only and
cite Harper & Konstan (2015).

```python
# Setup cell 1 of 2: install only (Colab resets on open; run this first)
%pip install -q pandas numpy matplotlib
```

```python
# Setup cell 2 of 2: imports + download + load. Safe to re-run (skips the download once present).
import os, hashlib, urllib.request, zipfile
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# MovieLens 100k -- downloaded at run time; NOT redistributed (license forbids it).
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

print("MovieLens 100k -- GroupLens / U. Minnesota (Harper & Konstan 2015); runtime download, not redistributed.")
print("ratings:", ratings.shape, "| users:", users.shape, "| items:", items.shape)
```

<details><summary>Expected Output</summary>

~~~text
MovieLens 100k -- GroupLens / U. Minnesota (Harper & Konstan 2015); runtime download, not redistributed.
ratings: (100000, 4) | users: (943, 5) | items: (1682, 24)
~~~
</details>

## Step 1: The Ratings Matrix (Worked)

A recommender starts from one object: the **user-item matrix** $R$, where $R_{ui}$
is the rating user $u$ gave item $i$. Almost every cell is empty -- nobody rates
more than a sliver of all movies. That emptiness (**sparsity**) is the whole
problem.

```python
R = ratings.pivot(index="user", columns="item", values="rating")
R = R.reindex(index=range(1, 944), columns=range(1, 1683))   # full 943 x 1682 grid
n_users, n_items = R.shape
filled = ratings.shape[0]
sparsity = 100 * (1 - filled / (n_users * n_items))
print(f"ratings matrix: {n_users} users x {n_items} movies")
print(f"filled cells: {filled}  ({100 - sparsity:.1f}% filled, {sparsity:.1f}% empty)")
```

<details><summary>Expected Output</summary>

~~~text
ratings matrix: 943 users x 1682 movies
filled cells: 100000  (6.3% filled, 93.7% empty)
~~~
</details>

Seeing that emptiness helps. The cell below draws a 60x80 corner of the matrix --
blue where a rating exists, white where it is missing.

```python
plt.figure(figsize=(7, 4))
plt.imshow(~np.isnan(R.values[:60, :80].astype(float)), aspect="auto", cmap="Blues")
plt.xlabel("movies (first 80)"); plt.ylabel("users (first 60)")
plt.title("Ratings matrix: a 60x80 corner (blue = rated)")
plt.show()
```

Almost all white: those few blue cells are everything a recommender has to work
with. The denser columns are the popular movies nearly everyone rates; the denser
rows are the most active users.

### Naive approach: just recommend the most popular   

The simplest "recommender" ignores *you* entirely and shows everyone the same
crowd-pleasers: the highest-rated movies (with enough ratings to be trustworthy).

```python
stats = ratings.groupby("item").rating.agg(["count", "mean"])
popular = (stats[stats["count"] >= 50]
           .sort_values("mean", ascending=False).head(5)
           .join(items.set_index("item")["title"]))
print(popular[["title", "mean", "count"]].round(2).to_string(index=False))
```

<details><summary>Expected Output</summary>

~~~text
                                                 title  mean  count
                                 Close Shave, A (1995)  4.49    112
                               Schindler's List (1993)  4.47    298
                            Wrong Trousers, The (1993)  4.47    118
                                     Casablanca (1942)  4.46    243
Wallace & Gromit: The Best of Aardman Animation (1996)  4.45     67
~~~
</details>

Fine list -- but it is the *same* for every user. Personalization needs the
ratings of people like you. That is collaborative filtering.

## Step 2: User-User Collaborative Filtering (Worked)

**Collaborative filtering** predicts your taste from the crowd: find users who
rated movies the way you did, then recommend what *they* liked that you have not
seen. Start where the concept note did: the **five-user toy**, small enough to
check by hand. Ana has rated four movies; the other four users have rated all six.
Ana has not seen *Speed* or *Aladdin* -- we will predict both.

```python
# The concept note's toy: 5 users x 6 movies (NaN = not seen)
toy = pd.DataFrame(
    [[5, 4, 1, 2, np.nan, np.nan],
     [5, 5, 3, 3, 5, 3],
     [3, 3, 1, 1, 3, 1],
     [1, 2, 5, 4, 1, 5],
     [4, 3, 4, 3, 3, 4]],
    index=["Ana", "Beto", "Cami", "Dani", "Eli"],
    columns=["Terminator", "TopGun", "ToyStory", "LionKing",
             "Speed", "Aladdin"])
shared = ["Terminator", "TopGun", "ToyStory", "LionKing"]

def cosine(a, b):
    return (a @ b) / (np.linalg.norm(a) * np.linalg.norm(b))

ana = toy.loc["Ana", shared].to_numpy()
rows = []
for name in ["Beto", "Cami", "Dani", "Eli"]:
    v = toy.loc[name, shared].to_numpy()
    c = cosine(ana - ana.mean(), v - v.mean())   # center, then cosine
    rows.append({"user": name,
                 "raw_cos": cosine(ana, v),
                 "centered_cos": c,
                 "pearson_r": np.corrcoef(ana, v)[0, 1],
                 "angle_deg": np.degrees(np.arccos(np.clip(c, -1, 1)))})
sims = pd.DataFrame(rows).set_index("user").round(3) + 0.0
print(sims.to_string())
```

<details><summary>Expected Output</summary>

~~~text
      raw_cos  centered_cos  pearson_r  angle_deg
user                                             
Beto    0.966         0.949      0.949     18.435
Cami    0.989         0.949      0.949     18.435
Dani    0.565        -1.000     -1.000    180.000
Eli     0.876         0.000      0.000     90.000
~~~
</details>

> **Read it:** the raw cosine says everyone resembles Ana (0.57 to 0.99) --
> all-positive ratings live in L18's positive corner, never more than 90 degrees
> apart. Centering moves the arrows out of that corner: generous Beto and stingy
> Cami become the *same* arrow (18.4 degrees from Ana), Eli is unrelated (90
> degrees) and Dani is Ana's opposite (180 degrees). On complete vectors like
> these, the centered cosine **is** Pearson's $r$ -- the two columns agree to
> the last digit, as L19 promised.

Now predict. Take the $k = 2$ most similar users, and add their
similarity-weighted **centered** ratings to Ana's own average:

$$\hat{r}_{\text{Ana},\,m} = \bar{r}_{\text{Ana}} + \frac{\sum_v s_v\,(r_{v,m} - \bar{r}_v)}{\sum_v |s_v|}$$

```python
k_nearest = sims["centered_cos"].sort_values(ascending=False, kind="stable").index[:2]
print("k = 2 neighbors:", list(k_nearest))
for movie in ["Speed", "Aladdin"]:
    s = sims.loc[k_nearest, "centered_cos"]
    centered = toy.loc[k_nearest, movie] - toy.loc[k_nearest].mean(axis=1)
    pred = ana.mean() + (s * centered).sum() / s.abs().sum()
    print(f"{movie}: {ana.mean():.2f} + {(s * centered).sum():.3f} / {s.abs().sum():.3f} = {pred:.2f}")
```

<details><summary>Expected Output</summary>

~~~text
k = 2 neighbors: ['Beto', 'Cami']
Speed: 3.00 + 1.898 / 1.898 = 4.00
Aladdin: 3.00 + -1.898 / 1.898 = 2.00
~~~
</details>

Both neighbors rated *Speed* one star above their own average and *Aladdin* one
below, so Ana gets 3 + 1 = 4.00 and 3 - 1 = 2.00: an action fan should see the
action movie.

**Real users are not this tidy.** In MovieLens a typical pair of users shares
only about 10 of the 1,682 movies, so the lab does three things more robustly
than the toy:

1. It compares over **all** 1,682 movies, not just the shared ones, counting
   every unrated movie as 0 (that is, "at the user's own average", which is
   taken over the movies that user rated). On complete vectors that is exactly $r$; on sparse ones it is a shrunk cousin
   of $r$ (often called *adjusted cosine*): two users who share only a few movies
   get a small score instead of a lucky $\pm 1$.
2. It uses 30 neighbors instead of 2.
3. It refuses to recommend a movie that too few of those neighbors rated (the
   guard below).

The cell below shows point 1 on the toy itself: Ana's two unseen movies now count
as zeros, and every similarity shrinks.

```python
toy_c = toy.sub(toy.mean(axis=1), axis=0).fillna(0.0)   # center, unseen -> 0
for name in ["Beto", "Cami", "Dani", "Eli"]:
    s = cosine(toy_c.loc["Ana"].to_numpy(), toy_c.loc[name].to_numpy())
    print(f"{name}: {s + 0.0:+.3f}")
```

<details><summary>Expected Output</summary>

~~~text
Beto: +0.775
Cami: +0.775
Dani: -0.745
Eli: +0.000
~~~
</details>

Same ranking, smaller numbers: the toy's 0.949 becomes 0.775 because Ana's
arrow now carries two zeros the others do not share. On the real matrix this is
what keeps thin overlaps from looking like soulmates.

Now the real thing: the same centering and cosine for all 943 users at once.

```python
Rv = R.values.astype(float)
rated = ~np.isnan(Rv)                       # boolean: which cells have a rating
user_mean = np.nanmean(Rv, axis=1)          # each user's average (over rated movies)
Rc = np.where(rated, Rv - user_mean[:, None], 0.0)   # mean-centered; unrated -> 0
norm = np.sqrt((Rc**2).sum(axis=1)); norm[norm == 0] = 1e-9
sim = (Rc @ Rc.T) / np.outer(norm, norm)    # 943 x 943 cosine similarities (L18 matrix product)
np.fill_diagonal(sim, 0.0)                  # a user is not their own neighbor
print("user-user similarity matrix:", sim.shape)
```

<details><summary>Expected Output</summary>

~~~text
user-user similarity matrix: (943, 943)
~~~
</details>

To recommend for a user: take their $k$ most similar neighbors and score each
unseen movie with the same formula as the toy. The function also counts the
**support** of each movie -- how many of the $k$ neighbors actually rated it.

```python
def recommend_user_based(uid, k=30, n=5, min_support=1):
    u = uid - 1                                   # user ids are 1-based
    neighbors = np.argsort(-sim[u])[:k]           # k most similar users
    w = sim[u, neighbors]
    num = w @ Rc[neighbors]                       # weighted (centered) ratings
    den = np.abs(w) @ rated[neighbors].astype(float)
    support = rated[neighbors].sum(axis=0)        # neighbors who rated each movie
    scores = user_mean[u] + np.divide(
        num, den, out=np.full(n_items, -np.inf), where=den > 0)
    scores[support < min_support] = -np.inf       # the guard
    scores[rated[u]] = -np.inf                    # drop movies u already rated
    top = np.argsort(-scores, kind="stable")[:n]  # stable: ties break by item id
    return [(title.loc[i + 1], scores[i], int(support[i])) for i in top]

print("top-5 for user 196, no guard:")
for t, score, sup in recommend_user_based(196):
    print(f"{score:.2f}  ({sup} of 30)  {t}")
```

<details><summary>Expected Output</summary>

~~~text
top-5 for user 196, no guard:
5.47  (1 of 30)  Andre (1994)
5.35  (1 of 30)  Old Yeller (1957)
5.35  (1 of 30)  Apartment, The (1960)
5.12  (1 of 30)  Bean (1997)
5.12  (1 of 30)  Story of Xinghua, The (1993)
~~~
</details>

> **Read it:** every one of these rests on a **single** neighbor out of 30 who
> happened to love it -- *Story of Xinghua* has only 5 ratings in the whole
> dataset. With one voter, the weighted average is just that voter's centered
> rating (how far above their own average they rated it), added on top of user
> 196's mean -- so the list is five individual opinions, not a consensus. (Scores above 5 are a
> clue: these are *ranking* scores -- a mean plus a bump -- not star predictions.)

### Better: require support from several neighbors   

The fix is one line: only score a movie that at least `min_support` of the 30
neighbors rated.

```python
print("top-5 for user 196, min_support=3:")
for t, score, sup in recommend_user_based(196, min_support=3):
    print(f"{score:.2f}  ({sup} of 30)  {t}")
```

<details><summary>Expected Output</summary>

~~~text
top-5 for user 196, min_support=3:
4.94  (5 of 30)  Shawshank Redemption, The (1994)
4.79  (3 of 30)  My Life as a Dog (Mitt liv som hund) (1985)
4.78  (3 of 30)  Remains of the Day, The (1993)
4.74  (6 of 30)  Usual Suspects, The (1995)
4.70  (7 of 30)  Boot, Das (1981)
~~~
</details>

> **Interpretation:** now each pick is backed by 3 to 7 like-minded users, and the
> list is still personal: none of these five is on Step 1's crowd-pleaser list.
> The guard is a trade-off:
> raise it and the list drifts toward movies *everyone* rated (you will see this
> in Exercise 1).

## Step 3: Item-Item Collaborative Filtering (Worked + Completion)

The other flavor flips the question: instead of "users like you," ask "movies like
the ones you liked." We compute cosine similarity between **movie columns** of the
same centered matrix (note the transpose -- L18's rule for going across columns),
then, for any movie, list its nearest neighbors.

```python
item_norm = np.sqrt((Rc**2).sum(axis=0)); item_norm[item_norm == 0] = 1e-9
S = (Rc.T @ Rc) / np.outer(item_norm, item_norm)   # 1682 x 1682 item-item cosine
np.fill_diagonal(S, 0.0)
print("item-item similarity matrix:", S.shape)
```

<details><summary>Expected Output</summary>

~~~text
item-item similarity matrix: (1682, 1682)
~~~
</details>

For a well-rated movie like Star Wars, no guard is needed: a rarely rated movie
shares only a few of Star Wars' many nonzero cells, so its cosine with Star Wars
stays small -- the zero-filling of Step 2 does the guarding. Query a *rarely*
rated movie, though, and its nearest neighbors are other rare movies rated by
the same few people, at cosines near 1: there the guard would be needed again.

```python
def similar_movies(movie_id, n=5):
    i = movie_id - 1
    top = np.argsort(-S[i], kind="stable")[:n]   # most similar movies first
    return [(title.loc[j + 1], S[i, j]) for j in top]

print(f"Movies most similar to {title.loc[50]}:")
for t, s in similar_movies(50):
    print(f"{s:.3f}  {t}")
```

<details><summary>Expected Output</summary>

~~~text
Movies most similar to Star Wars (1977):
0.665  Return of the Jedi (1983)
0.640  Empire Strikes Back, The (1980)
0.564  Raiders of the Lost Ark (1981)
0.386  Godfather, The (1972)
0.372  Princess Bride, The (1987)
~~~
</details>

> **Interpretation:** with no idea what any movie *is*, item-item cosine puts the
> two other Star Wars films at the very top -- the rating pattern alone recovers
> the trilogy. This is the engine behind "because you watched...".

**Your completion problem:** find the 5 movies most similar to **Pulp Fiction**
(item id 56). Uncomment, fill the `____`, and run -- you should get a tight cluster
of acclaimed crime dramas.

```python
# for t, s in similar_movies(____, n=5):
#     print(f"{s:.3f}  {t}")
```

<details><summary>Expected Output (when completed correctly)</summary>

~~~text
0.346  Usual Suspects, The (1995)
0.322  GoodFellas (1990)
0.304  Reservoir Dogs (1992)
0.302  Silence of the Lambs, The (1991)
0.285  One Flew Over the Cuckoo's Nest (1975)
~~~
</details>

## Step 4: Matrix Factorization -- the L18 Matrix Product, Made Real

Comparing whole rating vectors is shallow. **Matrix factorization** instead learns
a short **latent vector** per user and per movie -- a few hidden "taste" axes -- so
that the predicted ratings are a single matrix product $\hat{R} = P\,Q^\top$. First
see the shape of the idea on a tiny hand-built example: 4 users, 5 movies, 2
factors (`action`, `romance`).

```python
P_toy = np.array([[1.0, 0.1],    # Ana  -- action fan
                  [0.1, 1.0],    # Beto -- romance fan
                  [0.9, 0.8],    # Cami -- likes both
                  [0.2, 0.2]])   # Deni -- light viewer
Q_toy = np.array([[1.0, 0.0],    # Die Hard       -- action
                  [0.0, 1.0],    # The Notebook   -- romance
                  [0.8, 0.7],    # Mr & Mrs Smith -- both
                  [0.1, 0.1],    # a documentary  -- neither
                  [0.4, 0.9]])   # a rom-com      -- mostly romance
R_hat_toy = P_toy @ Q_toy.T      # (4 x 2) @ (2 x 5) = 4 users x 5 movies
toy_mf = pd.DataFrame(np.round(R_hat_toy, 2),
                      index=["Ana", "Beto", "Cami", "Deni"],
                      columns=["DieHard", "Notebook", "Smith", "Doc", "RomCom"])
print(toy_mf.to_string())
```

<details><summary>Expected Output</summary>

~~~text
      DieHard  Notebook  Smith   Doc  RomCom
Ana       1.0       0.1   0.87  0.11    0.49
Beto      0.1       1.0   0.78  0.11    0.94
Cami      0.9       0.8   1.28  0.17    1.08
Deni      0.2       0.2   0.30  0.04    0.26
~~~
</details>

The action fan scores highest on Die Hard, the romance fan on the Notebook and the
rom-com, and the user who likes both tops out on the movie that has both factors --
all from one matrix multiply. Here we *named* the two factors; a learned model will
not.

Now the real thing. Real predictions have a **baseline** that has nothing to do
with taste: the global average rating (`gmean`), how generous this user is
(`bu`, the user bias) and how well-liked this movie is (`bi`, the item bias). The
model is

$$\hat{r}_{ui} = \mu + b_u + b_i + p_u^\top q_i$$

and we **learn** all of it from the observed ratings by **stochastic gradient
descent** (nudge every term toward reducing the error on one rating at a time).
This trains on all 100k ratings -- about 15 seconds on a laptop, give it a little
longer on Colab.

```python
user_ids = {u: k for k, u in enumerate(sorted(ratings.user.unique()))}
item_ids = {i: k for k, i in enumerate(sorted(ratings.item.unique()))}
inv_item = {k: i for i, k in item_ids.items()}
data_arr = np.array([[user_ids[u], item_ids[i], r]
                     for u, i, r in ratings[["user", "item", "rating"]].values])

rng = np.random.default_rng(42)
K, lr, reg, epochs = 20, 0.01, 0.05, 20
P = rng.normal(0, 0.1, (len(user_ids), K))      # user latent factors
Q = rng.normal(0, 0.1, (len(item_ids), K))      # movie latent factors
bu = np.zeros(len(user_ids)); bi = np.zeros(len(item_ids)); gmean = ratings.rating.mean()

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
print(f"trained MF: P {P.shape}, Q {Q.shape}, K={K} latent factors")
```

<details><summary>Expected Output</summary>

~~~text
trained MF: P (943, 20), Q (1682, 20), K=20 latent factors
~~~
</details>

```python
def recommend_mf(uid, n=5):
    u = user_ids[uid]
    scores = gmean + bu[u] + bi + P[u] @ Q.T        # predict every movie at once (L18 matrix-vector)
    seen = ratings[ratings.user == uid].item.map(item_ids).values
    scores[seen] = -np.inf                          # drop already-rated movies
    top = np.argsort(-scores, kind="stable")[:n]       # stable: equal scores break by item id, not sort internals
    return [(title.loc[inv_item[j]], scores[j]) for j in top]

print("top-5 matrix-factorization recommendations for user 196:")
for t, score in recommend_mf(196):
    print(f"{score:.2f}  {t}")
print("user 196 latent vector (first 5 of 20 factors):", np.round(P[user_ids[196]][:5], 2))
```

<details><summary>Expected Output</summary>

~~~text
top-5 matrix-factorization recommendations for user 196:
4.68  Schindler's List (1993)
4.56  Pather Panchali (1955)
4.55  To Kill a Mockingbird (1962)
4.52  Shawshank Redemption, The (1994)
4.47  Casablanca (1942)
user 196 latent vector (first 5 of 20 factors): [-0.02  0.07 -0.17 -0.03 -0.01]
~~~
</details>

> **Interpretation:** these look like Step 1's crowd-pleasers, and that is no
> accident: the item bias `bi` (how well-liked a movie is overall) carries most
> of what lifts each score above the global average, and the factors re-order the list at the margin -- Exercise 4
> takes one prediction apart. Each row of `P` is a user's coordinates in a hidden
> taste space -- the same kind of low-dimensional embedding PCA gave you in L20,
> now learned from ratings -- but the 20 axes come **unnamed**: unlike the toy's
> "action" and "romance", most learned factors are mixtures that no single label
> fits. And 20 numbers per user and movie store far less than a 1682 x 1682
> similarity matrix. **Do these predictions match ratings the model has never
> seen?** We have not measured that -- the model was fit on every rating. That is
> the measurement L23 makes.

## Your Turn (Exercises)

### Exercise 1 -- Another user, and the guard

Pick any user id from 1 to 943. Loop `min_support` over 1, 3 and 5 for that
user and print each top-5 user-based list with its support counts.
In two sentences: what happens to the list as the guard rises, and why does a
high guard drift toward Step 1's crowd-pleasers?

```python
# TODO: your code here
```

<details><summary>Expected Output (for user 1 -- your user will differ)</summary>

~~~text
min_support=1
5.56  (1 of 30)  Flirt (1995)
5.51  (1 of 30)  Once Upon a Time... When We Were Colored (1995)
5.51  (1 of 30)  High Noon (1952)
5.51  (1 of 30)  Thirty-Two Short Films About Glenn Gould (1993)
5.51  (1 of 30)  Great Day in Harlem, A (1994)
min_support=3
5.11  (4 of 30)  Die xue shuang xiong (Killer, The) (1989)
5.10  (4 of 30)  Big Sleep, The (1946)
5.08  (3 of 30)  Once Upon a Time in the West (1969)
4.97  (12 of 30)  Close Shave, A (1995)
4.95  (4 of 30)  Secrets & Lies (1996)
min_support=5
4.97  (12 of 30)  Close Shave, A (1995)
4.87  (15 of 30)  Casablanca (1942)
4.86  (10 of 30)  L.A. Confidential (1997)
4.86  (6 of 30)  Duck Soup (1933)
4.85  (9 of 30)  Maltese Falcon, The (1941)
~~~
*(With no guard, all five picks rest on one neighbor each -- "A Great Day in
Harlem" has a single rating in the whole dataset. At 3, the picks are backed by
3-12 neighbors and mostly personal (a Hong Kong action film, film noir, a
Western), though Step 1's number one, A Close Shave, already slips in. At 5,
Casablanca joins it and the list reads like a classics shelf: only movies that
MANY people rated can collect five votes from a 30-user sample, and those are
the popular ones. The guard buys trust at the price of some
personalization.)*
</details>

### Exercise 2 -- "Because you liked...": item-based top-N for a user

Step 3 listed movies similar to *one* movie. Turn it into recommendations for a
**user**: score every movie by the **sum** of its item-item similarities `S` to
the movies user 196 rated **5**, drop the movies 196 has already rated, and
print the top 5 with their scores.

> **Hint:** `loved = np.where(Rv[195] == 5)[0]` gives the column indexes of the
> 5-star movies (user 196 is row 195). `S[:, loved].sum(axis=1)` sums each
> movie's similarity to all of them at once. Reuse `rated[195]` to drop seen
> movies.

```python
# TODO: your code here
```

<details><summary>Expected Output (one possible answer -- match the values, not the print format)</summary>

~~~text
0.95  Amadeus (1984)
0.91  To Kill a Mockingbird (1962)
0.91  Glory (1989)
0.90  Gandhi (1982)
0.89  Shawshank Redemption, The (1994)
~~~
*(Five acclaimed dramas, each close to several of the movies 196 loved. Why a
plain **sum**? It rewards a movie that is similar to many loved movies. A
similarity-**weighted average** of the user's ratings -- the Step 2 formula
turned sideways -- divides by the total similarity, which cancels the
shrinkage: a movie with one rater and a tiny similarity to a few of the user's
movies can still top the list. Try it and you will meet the Step 2 flukes again
(Coldblooded, 4.69, rated by one person); the weighted version needs Step 2's
support guard.)*
</details>

### Exercise 3 -- Written: the cold-start problem

A brand-new user has just signed up and rated **nothing**. Explain in 2-3
sentences why `recommend_user_based` cannot help them yet, and which approach
from the concept note's family table you would use instead, and why.

**Your answer:**

<details><summary>One possible answer</summary>

A new user has an empty rating row: there is no average to center on and no
movie to compare, so every similarity with them is 0, they have no real
neighbors, and no movie can collect any support. That is the
cold-start problem. A **demographic** approach can still recommend from what
users of the same age group or occupation liked, and a **knowledge-based** one
can simply ask what they want; both need no rating history (L23 builds them).
</details>

### Exercise 4 -- Take a prediction apart

For user 196's top matrix-factorization pick, print its four parts -- `gmean`,
`bu[u]`, `bi[i]` and `P[u] @ Q[i]` -- and their sum. Then rank user 196's
unseen movies by the **baseline alone** (`gmean + bu[u] + bi`) and compare its
top 5 with `recommend_mf(196)`. In two sentences: how much do the latent factors
change, and what is the baseline really recommending?

```python
# TODO: your code here
```

<details><summary>Expected Output (one possible answer -- match the values, not the print format)</summary>

~~~text
Schindler's List (1993): 3.530 + -0.017 + 0.984 + 0.178 = 4.675
baseline-only top-5:
4.55  Pather Panchali (1955)
4.50  Schindler's List (1993)
4.49  Casablanca (1942)
4.46  Close Shave, A (1995)
4.45  Shawshank Redemption, The (1994)
~~~
*(The item bias (+0.984) is more than five times the factor term (+0.178),
and the baseline alone already picks 4 of the MF top 5. Within one user, `bu`
adds the same constant to every movie, so it never changes the order: the
baseline ranking is simply a ranking by `bi`, a shrunk version of Step 1's
highest-rated list, the same for every user apart from the movies they have
already rated. The factors are what make it personal -- they
lifted To Kill a Mockingbird into user 196's list -- but on this data they
nudge the order rather than rewrite it.)*
</details>

## Summary

- A recommender starts from the sparse **user-item matrix** (here 93.7% empty).
- **Collaborative filtering** turns the crowd into personalization. On the toy,
  the centered cosine between two users **is** L19's Pearson $r$ -- the angle between taste arrows;
  on the sparse matrix the lab uses its practical cousin (center on each user's
  mean, unrated = 0), which damps thin overlaps -- between users (Step 2) or
  movies (Step 3), with no library.
- A **support guard** (`min_support`) keeps one enthusiastic neighbor from
  producing a "recommendation"; raising it trades personalization for trust.
- **Matrix factorization** learns a baseline (global mean + user bias + item
  bias) plus the taste match $P\,Q^\top$ of latent user/movie vectors -- L18's
  matrix product and L20's PCA idea, applied to recommendation. The baseline
  does most of the work; the unnamed factors personalize at the margin.
- We **generated** recommendations but did not yet **measure** them. Next, **L23**
  adds content-, knowledge-, and demographic-based recommenders and shows how to
  evaluate every approach.
