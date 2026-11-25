---
title: "Mini-Project: Which Recommender Would You Ship?"
unit: "VI"
lesson: "24"
type: lab
tags: [recommender-systems, capstone, collaborative-filtering, evaluation, precision-at-k, goodbooks-10k]
difficulty: intermediate
duration: "75 mins"
---

**The brief.** You are the (only) data scientist at a small reading app. The product team
wants a "Recommended for you" shelf, and your manager asks a sharp question: *which
recommender should we ship, and how do you know it is good?* This time the data is new --
**book** ratings, not the movies of L22/L23 -- so nothing you remember about MovieLens can
carry you. You get a small **toolkit** that loads the data, trains two recommenders and
scores them. You write the three functions that decide the verdict, read the evidence,
make the calls, and write the one paragraph your manager actually wants. Builds on
[L22 (collaborative filtering)](l22_concept_collaborative_filtering.qmd) and
[L23 (other recommender families + how to evaluate)](l23_concept_content_based_recommenders.qmd).

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/devomh/cdat4001-2026/blob/main/u06_recommender_systems/l24_lab_recommender_mini_project.ipynb)

> This page is the read-only view. To run the lab, open the notebook
> (`l24_lab_recommender_mini_project.ipynb`) -- in Colab via the badge above, or
> locally. Needs network access the first time (it downloads about 27 MB).
> Every text output below comes from actually running the cells. This is a capstone:
> the help fades as you go -- worked, then fill-in, then described tasks, then your decisions.

**What you will do -- and hand in:**

1. **Set up** an honest test: read the data card, hold out 20% of the ratings, and say what
   counts as a "good" recommendation.
2. **Write** `top_k_unseen` and `precision_at_k`, then **score** a popularity recommender
   and item-item collaborative filtering against a random ranker and the best possible score.
3. **Diagnose** why one model's lists go wrong, write `apply_guard`, and **choose** a guard
   threshold from a sweep.
4. **Decide** what to ship, in a short paragraph with your final table, a plan for brand-new
   readers, and an honest account of why popularity is so hard to beat.

Hand in the notebook with your three functions, your `MY_GUARD`, and your three written
answers (Steps 1, 5 and 6).

## Prerequisites & Setup

We use **goodbooks-10k** (Zygmunt Zajac, 2017): about six million 1-5 star ratings that
53,424 Goodreads readers gave to the 10,000 most-rated books. It is licensed **CC BY-SA
4.0**, so you may share it with attribution. The notebook downloads two files from the
dataset's GitHub repository, pinned to one release, and checks their md5 sums.

```python
# Setup cell 1 of 2: install only (Colab resets on open; run this first)
%pip install -q pandas numpy matplotlib scikit-learn
```

```python
# Setup cell 2 of 2: imports + download. Safe to re-run.
import os, hashlib, urllib.request, unicodedata, zipfile
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# goodbooks-10k (Zajac 2017, CC BY-SA 4.0), pinned to release v1.0
FILES = {
    "ratings.zip": ("https://github.com/zygmuntz/goodbooks-10k/releases/download/v1.0/ratings.zip",
                    "f224b34afc6033663f7f40ccbdb194bf"),
    "books.csv": ("https://raw.githubusercontent.com/zygmuntz/goodbooks-10k/"
                  "6dd165b555a7b47b2dd36743a425776e641ff50c/books.csv",
                  "0854aebb4f0221e653ebaf5d18eeb144"),
}
os.makedirs("data/goodbooks", exist_ok=True)
for name, (url, md5) in FILES.items():
    path = f"data/goodbooks/{name}"
    if not os.path.exists(path):
        urllib.request.urlretrieve(url, path)
    assert hashlib.md5(open(path, "rb").read()).hexdigest() == md5, \
        f"{name}: checksum mismatch -- delete data/goodbooks/{name} and re-run this cell"
print("goodbooks-10k files ready:", ", ".join(sorted(FILES)))
```

<details><summary>Expected Output</summary>

~~~text
goodbooks-10k files ready: books.csv, ratings.zip
~~~
</details>

## The Toolkit

The next four cells are the toolkit. **Run them; you do not need to read every line.** Each
starts with a `# @title` and a docstring that says what it does. Everything is a table with
one row per reader and one column per book, and every model offers the same two methods:

| Piece | What it gives you |
|---|---|
| `BookData` | the sample, the train/test split, `data.card()` |
| `Popularity`, `ItemItemCF` | `.fit(data)` trains; `.scores()` returns a readers x books table of scores |
| `Evaluator` | `ev.precision(...)`, `ev.rmse(...)`, `ev.references(...)`, `ev.sweep(...)`, `ev.board(...)` |
| plot helpers | `plot_long_tail`, `plot_pick_sources`, `plot_guard_sweep`, `plot_board` |

The evaluator ships with its own fast versions of the three functions you will write. When
your version passes its check, the evaluator switches to yours, so the final numbers are
computed by your code. Write yours **before** you look at the toolkit's.

```python
# @title Toolkit 1/4 -- data: load, sample, split
class BookData:
    """A sample of goodbooks-10k with a train/test split.

    Keeps the n_books most-rated books, then samples n_users readers who rated at
    least min_user of them. Readers and books are numbered 0, 1, 2, ... (rows and
    columns of every table). R holds the training ratings (NaN = not rated).
    """
    def __init__(self, n_books=4000, n_users=6000, min_user=30, seed=42):
        with zipfile.ZipFile("data/goodbooks/ratings.zip") as z:
            ratings = pd.read_csv(z.open("ratings.csv"))
        books = pd.read_csv("data/goodbooks/books.csv", usecols=["book_id", "title"])
        # 1) the n_books most-rated books (ties broken by book id)
        per_book = ratings.groupby("book_id").size().sort_values(ascending=False, kind="stable")
        self.all_counts = per_book.to_numpy()          # all 10,000 books, most-rated first
        ratings = ratings[ratings.book_id.isin(per_book.index[:n_books])]
        # 2) readers with at least min_user ratings among them -- a fixed random sample
        per_user = ratings.groupby("user_id").size()
        eligible = pd.Series(per_user[per_user >= min_user].index)
        chosen = eligible.sample(n=n_users, random_state=seed)
        r = ratings[ratings.user_id.isin(chosen)].copy()
        # 3) number readers and books 0, 1, 2, ...
        self.user_ids, self.book_ids = np.sort(r.user_id.unique()), np.sort(r.book_id.unique())
        r["u"] = np.searchsorted(self.user_ids, r.user_id)
        r["i"] = np.searchsorted(self.book_ids, r.book_id)
        self.ratings = r.reset_index(drop=True)
        self.n_users, self.n_books = len(self.user_ids), len(self.book_ids)
        to_ascii = lambda t: unicodedata.normalize("NFKD", t).encode("ascii", "ignore").decode()
        self.titles = books.set_index("book_id").title.map(to_ascii).loc[self.book_ids].to_numpy()
        # 4) hold out 20% of the ratings
        self.train, self.test = train_test_split(self.ratings, test_size=0.2, random_state=42)
        self.R = np.full((self.n_users, self.n_books), np.nan, dtype=np.float32)
        self.R[self.train.u, self.train.i] = self.train.rating
        self.seen = ~np.isnan(self.R)                  # True = rated in the training split
        self.counts = self.seen.sum(axis=0)            # training ratings per book
        liked = self.test[self.test.rating >= 4]
        self.relevant = liked.groupby("u").i.apply(set)   # each reader's liked held-out books

    def card(self):
        """Print a short description of the sample (and a fingerprint to compare with)."""
        r = self.ratings
        stars = r.rating.value_counts(normalize=True).sort_index()
        print(f"ratings: {len(r):,} | readers: {self.n_users:,} | books: {self.n_books:,}")
        print(f"filled: {len(r) / (self.n_users * self.n_books):.1%} of the readers x books table")
        print(f"ratings per reader: median {r.groupby('u').size().median():.0f} | "
              f"per book: median {r.groupby('i').size().median():.0f}, "
              f"fewest {r.groupby('i').size().min()}")
        print("stars:", " | ".join(f"{s}: {p:.0%}" for s, p in stars.items()))
        print(f"train: {len(self.train):,} ratings | test: {len(self.test):,} ratings")
        print(f"fingerprint: {len(r)} ratings, {int(r.rating.sum())} stars in total")
```

```python
# @title Toolkit 2/4 -- models: every model has fit(data) and scores()
class Popularity:
    """Recommend the most-rated books to everyone (no personalization).

    scores() gives every book its number of training ratings: a ranking score,
    not a predicted star rating, so this model has no RMSE.
    """
    predicts_ratings = False

    def fit(self, data):
        self.shape, self.counts = (data.n_users, data.n_books), data.counts.astype(float)
        return self

    def scores(self):
        return np.broadcast_to(self.counts, self.shape)   # the same row for every reader

class ItemItemCF:
    """L22's item-item collaborative filtering, trained on the training split.

    Cosine similarity between centered book columns; a reader's predicted rating
    for a book = their mean + the similarity-weighted average of their own
    centered ratings (the version L22's Exercise 2 warned about).
    """
    predicts_ratings = True

    def fit(self, data):
        rated = data.seen
        user_mean = np.nanmean(data.R, axis=1)
        Rc = np.where(rated, data.R - user_mean[:, None], 0).astype(np.float32)
        item_norm = np.sqrt((Rc**2).sum(axis=0))
        item_norm[item_norm == 0] = 1e-9
        S = (Rc.T @ Rc) / np.outer(item_norm, item_norm)   # books x books cosine
        np.fill_diagonal(S, 0.0)
        num = Rc @ S
        den = rated.astype(np.float32) @ np.abs(S)
        self.pred = user_mean[:, None] + np.divide(num, den, out=np.zeros_like(num), where=den > 0)
        return self

    def scores(self):
        return self.pred
```

```python
# @title Toolkit 3/4 -- evaluator: RMSE, precision@k, guard, reference values
def builtin_top_k_unseen(scores, seen, k):
    """Toolkit version, works on one row or a whole table at once."""
    masked = np.where(seen, -np.inf, scores)
    return np.argsort(-masked, axis=-1, kind="stable")[..., :k]

def builtin_apply_guard(scores, counts, min_ratings):
    return np.where(counts >= min_ratings, scores, -np.inf)

class Evaluator:
    """Scores models on the held-out test ratings.

    Ranking: for every reader with at least one liked held-out book, take the k
    best-scored books they have not rated and measure precision@k; average over
    readers. Rating prediction: RMSE on all held-out ratings.
    """
    def __init__(self, data):
        self.data, self.mine = data, {}               # mine = your functions that passed
        self.readers = np.array(sorted(data.relevant.index))
        print(f"readers with at least one liked held-out book: {len(self.readers):,} "
              f"of {data.n_users:,}")

    # ---- the three steps of a ranking evaluation (yours replace the toolkit's) ----
    def _lists(self, model, k, guard):
        d, S = self.data, model.scores()
        if not self.mine:                              # fast path: the whole table at once
            S = builtin_apply_guard(S, d.counts, guard) if guard else S
            return builtin_top_k_unseen(S[self.readers], d.seen[self.readers], k)
        top_k = self.mine.get("top_k_unseen", builtin_top_k_unseen)
        guard_fn = self.mine.get("apply_guard", builtin_apply_guard)
        rows = []
        for u in self.readers:                         # your functions: one reader at a time
            s = np.array(S[u])                         # a copy, so the model stays untouched
            s = guard_fn(s, d.counts, guard) if guard else s
            rows.append(np.asarray(top_k(s, d.seen[u].copy(), k)))
        return np.array(rows)

    def precision(self, model, k=10, guard=0):
        """Mean precision@k over readers (guard = minimum training ratings, 0 = none)."""
        lists, rel = self._lists(model, k, guard), self.data.relevant
        if "precision_at_k" in self.mine:
            p = [self.mine["precision_at_k"](lst, rel[u], k) for u, lst in zip(self.readers, lists)]
        else:
            p = [len(rel[u].intersection(lst.tolist())) / k for u, lst in zip(self.readers, lists)]
        return float(np.mean(p))

    def references(self, k=10):
        """Expected precision@k of a random ranker, and the best possible precision@k."""
        d = self.data
        n_rel = np.array([len(d.relevant[u]) for u in self.readers])
        n_unseen = d.n_books - d.seen[self.readers].sum(axis=1)
        return {"random": float(np.mean(n_rel / n_unseen)),
                "best possible": float(np.mean(np.minimum(n_rel, k) / k))}

    def rmse(self, model):
        """RMSE on all held-out ratings (predictions clipped to 1-5); None for a ranker."""
        if not model.predicts_ratings:
            return None
        t = self.data.test
        p = np.clip(model.scores()[t.u.to_numpy(), t.i.to_numpy()], 1, 5)
        return float(np.sqrt(np.mean((p - t.rating.to_numpy())**2)))

    def rmse_references(self):
        """RMSE of predicting the global mean, and of predicting each book's mean."""
        tr, t = self.data.train, self.data.test
        book_mean = tr.groupby("i").rating.mean().reindex(range(self.data.n_books))
        pred = book_mean.fillna(tr.rating.mean()).to_numpy()[t.i.to_numpy()]
        return {"global mean": float(np.sqrt(np.mean((tr.rating.mean() - t.rating)**2))),
                "book mean": float(np.sqrt(np.mean((pred - t.rating.to_numpy())**2)))}

    def top_k(self, model, reader, k=5, guard=0):
        """Print one reader's top-k list: training-rating count, title, and whether
        the reader liked it in the held-out set."""
        d = self.data
        s = np.array(model.scores()[reader])           # a copy, so the model stays untouched
        s = self.mine.get("apply_guard", builtin_apply_guard)(s, d.counts, guard) if guard else s
        liked = d.relevant.get(reader, set())
        for i in self.mine.get("top_k_unseen", builtin_top_k_unseen)(s, d.seen[reader].copy(), k):
            mark = "  <- liked it (held out)" if i in liked else ""
            print(f"{d.counts[i]:5d} ratings  {d.titles[i]}{mark}")

    def pick_counts(self, model, k=10, guard=0):
        """Training-rating count of every book placed in a top-k list."""
        return self.data.counts[self._lists(model, k, guard)].ravel()

    def sweep(self, model, guards, k=10):
        """precision@k for a list of guard thresholds."""
        return pd.DataFrame({"guard (min ratings)": guards,
                             "books left": [int((self.data.counts >= g).sum()) for g in guards],
                             f"precision@{k}": [round(self.precision(model, k, g), 3) for g in guards]})

    def board(self, models, names, guard=0, k=10):
        """The final table: RMSE and precision@k for each model (guard applied to all)."""
        rows = []
        for m, name in zip(models, names):
            e = self.rmse(m)
            rows.append({"model": name, "RMSE (lower=better)": "n/a (ranker)" if e is None else f"{e:.3f}",
                         f"precision@{k} (higher=better)": round(self.precision(m, k, guard), 3)})
        return pd.DataFrame(rows).set_index("model")

    def exposure(self, top=200):
        """How much of the test set the most-rated books hold, and how well they are liked."""
        d, t = self.data, self.data.test
        head = np.argsort(-d.counts, kind="stable")[:top]
        on_head = t.i.isin(head)
        print(f"the {top} most-rated books are {top / d.n_books:.0%} of the catalog")
        print(f"share of held-out ratings on them: {on_head.mean():.0%}")
        print(f"rated 4 or 5 -- on them: {(t.rating[on_head] >= 4).mean():.0%} | "
              f"on all other books: {(t.rating[~on_head] >= 4).mean():.0%}")

    # ---- checks: compare your function with the toolkit's ----
    def check(self, name, fn):
        """Run your function on a toy case and on real readers; use it if it passes."""
        d = self.data
        def frozen(a):                     # a read-only copy: your function must build a
            a = np.array(a)                # new array, not change the one it was given
            a.setflags(write=False)
            return a
        same_order = True                  # did ties come out in the toolkit's order?
        try:
            if name == "top_k_unseen":
                toy = [int(i) for i in fn(frozen([0.5, 2.0, 1.0, 3.0, 2.5]),
                                          frozen([False, False, False, True, False]), 3)]
                ok_toy = toy == [4, 1, 2]
                s = frozen(d.counts - 10000.0)   # real readers; every score is negative
                ok_real = True
                for u in self.readers[:200]:
                    mine = np.asarray(fn(s, frozen(d.seen[u]), 10)).astype(int)
                    ref = builtin_top_k_unseen(s, d.seen[u], 10)
                    ok_real &= (len(mine) == 10 and not d.seen[u][mine].any()
                                and np.array_equal(s[mine], s[ref]))   # ties may swap places
                    same_order &= np.array_equal(mine, ref)
            elif name == "precision_at_k":
                toy = fn(frozen([4, 1, 2]), {1, 7}, 3)
                ok_toy = abs(toy - 1 / 3) < 1e-9
                ok_real = abs(fn(frozen(np.arange(10)), {0, 5, 99}, 10) - 0.2) < 1e-9
            elif name == "apply_guard":
                toy = [float(x) for x in fn(frozen([5.0, 4.0, 3.0]), frozen([100, 10, 50]), 50)]
                ok_toy = toy == [5.0, -np.inf, 3.0]
                s = frozen(d.counts.astype(float))
                ok_real = np.array_equal(fn(s, frozen(d.counts), 50),
                                         builtin_apply_guard(s, d.counts, 50))
        except NotImplementedError:
            print(f"{name}: not finished yet -- the evaluator keeps using the toolkit's version")
            return
        except ValueError as err:
            if "read-only" not in str(err):
                raise
            print(f"{name}: your function changes the array it was given -- "
                  "build a new array instead (np.where does)")
            return
        print(f"{name} on the toy case: {toy}")
        if ok_toy and ok_real:
            self.mine[name] = fn
            print(f"{name}: PASS -- from now on the evaluator uses your function")
            if not same_order:
                print("  note: your function puts tied books in a different order than the "
                      "toolkit's,\n  so later numbers may differ from this page in the third "
                      "decimal; kind=\"stable\" matches it exactly")
        elif ok_toy:
            where = {"top_k_unseen": " (where do seen books have to end up?)",
                     "apply_guard": " (where do guarded books have to end up?)"}.get(name, "")
            print(f"{name}: the toy case is right, but real readers are not -- "
                  f"re-read the hint{where}")
        else:
            print(f"{name}: does not match yet -- compare with the expected output and try again")
```

```python
# @title Toolkit 4/4 -- plot helpers
COLORS = {"popularity": "#2a78d6", "item-item CF": "#eb6834", "random": "#8a8a85"}

def plot_long_tail(data):
    """Ratings per book for all 10,000 books (with the sample's cut), and the star mix."""
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.8), gridspec_kw={"width_ratios": [2.2, 1]})
    c = data.all_counts
    ax[0].plot(np.arange(1, len(c) + 1), c, color=COLORS["popularity"], lw=2)
    ax[0].set_yscale("log")
    ax[0].axvline(data.n_books, color="#55554f", ls="--", lw=1)
    ax[0].text(data.n_books + 150, c[0] / 3, f"our sample:\nthe top {data.n_books:,}", fontsize=9)
    ax[0].set_xlabel("book rank (most-rated first)")
    ax[0].set_ylabel("ratings per book (log scale)")
    ax[0].set_title("A long tail: a few books collect most of the ratings")
    stars = data.ratings.rating.value_counts(normalize=True).sort_index() * 100
    ax[1].bar(stars.index, stars.values, width=0.7,
              color=["#b7d3f6"] * 3 + [COLORS["popularity"]] * 2)
    for s, p in stars.items():
        ax[1].text(s, p + 1, f"{p:.0f}%", ha="center", fontsize=9)
    ax[1].set_ylim(0, stars.max() * 1.15)
    ax[1].set_xlabel("stars")
    ax[1].set_ylabel("% of the sample's ratings")
    ax[1].set_title("Most ratings are 4 or 5 stars")
    plt.tight_layout()
    plt.show()

def plot_pick_sources(ev, models, names, k=10):
    """Box plot: how many training ratings the books in each model's top-k lists have."""
    data = [ev.pick_counts(m, k) for m in models]
    fig, ax = plt.subplots(figsize=(9, 1.2 + 0.9 * len(models)))
    bp = ax.boxplot(data, orientation="horizontal", tick_labels=names, whis=(5, 95),
                    showfliers=False, patch_artist=True, widths=0.55)
    for box, n in zip(bp["boxes"], names):
        box.set_facecolor(COLORS.get(n, "#8a8a85"))
        box.set_edgecolor("white")
    for med in bp["medians"]:
        med.set_color("black")
    ax.set_xscale("log")
    ax.axvline(50, color="#55554f", ls="--", lw=1)
    for row, d in enumerate(data, start=1):
        ax.text(1.01, row, f"{(d < 50).mean():.0%} under 50", transform=ax.get_yaxis_transform(),
                va="center", fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel(f"training ratings of each book in a top-{k} list "
                  "(log scale; box = middle half, whiskers 5-95%)")
    ax.set_title(f"Where each model's top-{k} picks come from (dashed line: 50 ratings)")
    plt.tight_layout()
    plt.show()

def plot_guard_sweep(sweep, name, popularity, random, k=10):
    """precision@k against the guard threshold, with popularity and random as lines."""
    col = f"precision@{k}"
    x = np.arange(len(sweep))
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(x, sweep[col], "-o", color=COLORS.get(name, "#333333"), lw=2, ms=6, label=name)
    ax.axhline(popularity, color=COLORS["popularity"], ls="--", lw=1.5, label="popularity (no guard)")
    ax.axhline(random, color=COLORS["random"], ls=":", lw=1.5, label="random")
    ax.set_xticks(x, [f">= {g}\n{b:,} books" for g, b in
                      zip(sweep["guard (min ratings)"], sweep["books left"])])
    ax.set_ylim(0, max(popularity, sweep[col].max()) * 1.15)
    ax.set_xlabel("guard: minimum training ratings a book needs to be recommended")
    ax.set_ylabel(col)
    ax.set_title(f"The guard decision for {name}")
    ax.legend(loc="lower right", frameon=False)
    plt.tight_layout()
    plt.show()

def plot_board(board, guard=0, k=10):
    """Bar chart of the final precision@k column."""
    col = f"precision@{k} (higher=better)"
    fig, ax = plt.subplots(figsize=(6, 3))
    bars = ax.barh(board.index, board[col], color=[COLORS.get(n, "#8a8a85") for n in board.index])
    ax.bar_label(bars, fmt="%.3f", padding=3)
    ax.invert_yaxis()
    ax.set_xlabel(f"precision@{k}")
    ax.set_title(f"Final ranking quality (guard >= {guard})" if guard else
                 "Final ranking quality (no guard)")
    ax.set_xlim(0, board[col].max() * 1.2)
    plt.tight_layout()
    plt.show()

print("toolkit ready")
```

<details><summary>Expected Output</summary>

~~~text
toolkit ready
~~~
</details>

## Step 1: Meet the Data (Worked)

The toolkit keeps the **4,000 most-rated books** and a fixed random sample of **6,000
readers** who rated at least 30 of them (the full table, and the collaborative-filtering
computations on it, would not fit in Colab's memory). Load it and read the card.

```python
data = BookData(n_books=4000, n_users=6000, min_user=30, seed=42)
data.card()
```

<details><summary>Expected Output</summary>

~~~text
ratings: 558,464 | readers: 6,000 | books: 4,000
filled: 2.3% of the readers x books table
ratings per reader: median 92 | per book: median 72, fewest 5
stars: 1: 2% | 2: 6% | 3: 23% | 4: 36% | 5: 34%
train: 446,771 ratings | test: 111,693 ratings
fingerprint: 558464 ratings, 2192766 stars in total
~~~
</details>

> **Reading the card:** 558,464 ratings from 6,000 readers on 4,000 books. The table is only
> 2.3% filled -- sparser than MovieLens in L22 (6.3%) -- yet a typical reader rated 92 of these
> books. The books are uneven even inside the sample: the median book has 72 ratings, the
> least-rated only 5. And about 70% of the stars are 4 or 5: readers mostly rate books they
> enjoyed.
> The fingerprint lets you compare: if your counts differ, your sample differs from this page's.

```python
plot_long_tail(data)
```

> **Reading the figure:** on the left, all 10,000 books ordered by number of ratings (log
> scale). The most-rated book has 22,806 ratings; by rank 4,000 it is down to 311, and the curve
> keeps falling -- a **long tail**: a few books collect most of the ratings. Our sample keeps
> the left part, and the same shape repeats inside it. On the right, the star mix: 4s and 5s
> dominate, so "rated it 4 or 5" -- our definition of a good recommendation, below -- covers
> most of the books a reader rates at all. Both facts come back in Step 6.

**The evaluation plan.** These are **explicit** ratings. We built the sample's split above:
80% of the ratings train the models, 20% are **held out**. A held-out book counts as
**relevant** to a reader if they rated it **4 or 5** -- a book they liked. A good
recommender should put those books near the top of that reader's list, among all the books
the reader has *not* rated in training.

```python
ev = Evaluator(data)
```

<details><summary>Expected Output</summary>

~~~text
readers with at least one liked held-out book: 5,987 of 6,000
~~~
</details>

> **Reading it:** 13 readers have no liked book in the held-out set, so there is nothing a list
> could hit; the evaluator skips them and averages over the other 5,987.

**Your turn (written, 2-3 sentences).** Why must "relevant" come only from the **held-out**
ratings, never from the training ratings? And what does the share of 4- and 5-star
ratings, about 70%, mean for this definition of a good recommendation?

**Your answer:**

<details><summary>One possible answer</summary>

The models learn from the training ratings, so a book the reader rated there is already
known: a list that "found" it would be graded on data the model has seen, and the score would
say nothing about books the reader has not met yet. That is also why the lists may only contain
books the reader has not rated in training. And because about 70% of all ratings are 4 or 5,
most books a reader rated (about 7 in 10) count as liked: "relevant" here mostly means "a book
this reader went on to read". Keep that in mind when you read the scores.
</details>

## Step 2: Popularity and the Top-k List (Guided Completion)

The simplest recommender: show everyone the most-rated books.

```python
pop = Popularity().fit(data)
print("the 5 most-rated books in the training split (the same for every reader):")
for i in np.argsort(-data.counts, kind="stable")[:5]:
    print(f"{data.counts[i]:5d} ratings  {data.titles[i]}")
```

<details><summary>Expected Output</summary>

~~~text
the 5 most-rated books in the training split (the same for every reader):
 2118 ratings  The Hunger Games (The Hunger Games, #1)
 1985 ratings  Harry Potter and the Sorcerer's Stone (Harry Potter, #1)
 1756 ratings  To Kill a Mockingbird
 1589 ratings  Twilight (Twilight, #1)
 1538 ratings  The Great Gatsby
~~~
</details>

A reader should never be offered a book they have already rated. **Your first function**
takes one reader's row of scores and their `seen` row (True = already rated) and returns the
column numbers of the `k` best **unseen** books, best first.

> **Hint:** `np.where(condition, a, b)` keeps `b` where the condition is False. Sending seen
> books to `-np.inf` puts them last. `np.argsort(-scores, kind="stable")` lists positions from
> highest to lowest score (ties keep their order).

```python
def top_k_unseen(scores, seen, k):
    """Column numbers of the k highest-scored books with seen == False, best first."""
    # 1) uncomment and complete: push the already-seen books to the bottom
    # scores = np.where(seen, ____, scores)
    # 2) uncomment and complete: the positions of the k largest scores, best first
    # return np.argsort(-scores, kind="stable")[:____]
    raise NotImplementedError   # delete this line when you are done
```

Now check it. On the toy case (scores `[0.5, 2.0, 1.0, 3.0, 2.5]`, book 3 already seen,
`k=3`) the answer is `[4, 1, 2]`; the check also compares your function with the toolkit's on
200 real readers.

```python
ev.check("top_k_unseen", top_k_unseen)
```

<details><summary>Expected Output</summary>

~~~text
top_k_unseen on the toy case: [4, 1, 2]
top_k_unseen: PASS -- from now on the evaluator uses your function
~~~
</details>

Until you finish the function, the check prints this instead (the notebook still runs, using
the toolkit's version):

~~~text
top_k_unseen: not finished yet -- the evaluator keeps using the toolkit's version
~~~

What does popularity offer one reader? The list is the same for everyone, minus the books
each reader has already rated:

```python
ev.top_k(pop, reader=26, k=5)
```

<details><summary>Expected Output</summary>

~~~text
 1756 ratings  To Kill a Mockingbird  <- liked it (held out)
 1589 ratings  Twilight (Twilight, #1)
 1517 ratings  Catching Fire (The Hunger Games, #2)
 1498 ratings  Harry Potter and the Chamber of Secrets (Harry Potter, #2)
 1494 ratings  Harry Potter and the Prisoner of Azkaban (Harry Potter, #3)
~~~
</details>

> **Reading the list:** The Hunger Games, the first Harry Potter and The Great Gatsby are
> missing because reader 26 rated them in training, so the next bestsellers move up. The mark
> `<- liked it (held out)` comes from the test set the model never saw: To Kill a Mockingbird is
> one of reader 26's liked held-out books, so this list scores **one hit in five**. Step 3 turns
> that count into a metric.

## Step 3: Measuring a Ranking -- precision@k (Guided Completion)

Readers never see a predicted star; they see a **ranked list**. **precision@k** asks: of the
`k` books we put in front of a reader, what fraction did they actually like (a held-out book
they rated 4 or 5)? The evaluator averages it over all readers. L23 named this metric; now
you write it.

```python
def precision_at_k(recommended, relevant, k):
    """Fraction of the first k recommended books that are in the set `relevant`."""
    # Uncomment and complete one line. `recommended` is an array of column numbers,
    # `relevant` is a Python set of column numbers.
    # return len(set(recommended[:k]) & ____) / ____
    raise NotImplementedError   # delete this line when you are done
```

The toy case: recommended `[4, 1, 2]`, relevant `{1, 7}`, `k=3` -- one hit in three.

```python
ev.check("precision_at_k", precision_at_k)
```

<details><summary>Expected Output</summary>

~~~text
precision_at_k on the toy case: 0.3333333333333333
precision_at_k: PASS -- from now on the evaluator uses your function
~~~
</details>

Until it is finished, the check prints this instead:

~~~text
precision_at_k: not finished yet -- the evaluator keeps using the toolkit's version
~~~

A number on its own means little, so the evaluator gives two **reference values**: what a
**random** ranker would score, and the **best possible** score (a perfect ranker can only
hit as many books as the reader actually liked).

```python
p_pop = ev.precision(pop, k=10)
ref = ev.references(k=10)
print(f"popularity precision@10:     {p_pop:.3f}")
print(f"random ranker (expected):    {ref['random']:.3f}")
print(f"best possible precision@10:  {ref['best possible']:.3f}")
```

<details><summary>Expected Output</summary>

~~~text
popularity precision@10:     0.081
random ranker (expected):    0.003
best possible precision@10:  0.902
~~~
</details>

> **Reading the numbers:** popularity's top 10 holds 0.81 liked books on average (precision
> 0.081). That is about 25 times a random ranker (0.003, roughly one liked book per 300 shown)
> -- and far below the best possible 0.902, because most readers liked many held-out books, so a
> perfect ranker could fill nearly every slot. Remember 0.081: it is the bar a personalized
> model has to clear.

## Step 4: Item-Item Collaborative Filtering (Described Task)

Now a personalized model: L22's item-item collaborative filtering, trained on the training
split. Fitting it takes a few seconds.

```python
cf = ItemItemCF().fit(data)
print("item-item CF trained")
```

<details><summary>Expected Output</summary>

~~~text
item-item CF trained
~~~
</details>

**Your task (no template this time):** CF predicts star ratings, so it has an RMSE. First,
print CF's RMSE next to the two RMSE reference values from `ev.rmse_references()`; then print
CF's precision@10 next to popularity's `p_pop`. Use `ev.rmse(cf)` and `ev.precision(cf, k=10)`.

```python
# TODO: your code here
```

<details><summary>Expected Output (match the values, not the print format)</summary>

~~~text
RMSE   global mean: 0.994 | book mean: 0.961 | item-item CF: 0.838
precision@10   popularity: 0.081 | item-item CF: 0.020
~~~
</details>

> **Reading the numbers:** as a *rating predictor* CF is good: its typical error is 0.838
> stars, well below predicting each book's mean (0.961) or one global mean (0.994). As a
> *ranker* it is poor: 0.020, a quarter of popularity's 0.081 (though still about 6 times
> random). Same model, two verdicts -- L23's "rating error is not ranking quality", measured.
> Why is the list so poor?

Reader 26 again, this time with CF's list:

```python
ev.top_k(cf, reader=26, k=5)
```

<details><summary>Expected Output</summary>

~~~text
   31 ratings  Rush (Breathless, #1)
   43 ratings  Step on a Crack (Michael Bennett, #1)
   38 ratings  Charade (Games, #1)
   48 ratings  Beautiful Oblivion (The Maddox Brothers, #1)
   45 ratings  Beauty from Pain (Beauty, #1)
~~~
</details>

**Before reading on:** compare this list with reader 26's popularity list in Step 2. What do
CF's five picks have in common? Note your guess, then test it on every reader: for each book
placed in a top-10 list, how many training ratings does it have?

```python
plot_pick_sources(ev, [pop, cf], ["popularity", "item-item CF"])
```

> **Reading the figure:** popularity's picks have about 1,000 training ratings or more (its
> box sits far right). CF's middle pick has about 37, and 75% of its picks have fewer than 50 --
> the far end of the long tail. This is the flood L22's Exercise 2 warned about: a weighted
> average divides by the total similarity, so a book that only a few people rated, faintly
> similar to a few of the reader's books, can get an extreme predicted rating. A book few people
> read is rarely in anyone's held-out set, so these picks almost never hit. And RMSE does not
> notice: it is scored only on books readers actually rated, and they rarely rated these, so
> CF's extreme predictions for them almost never reach the RMSE.

## Step 5: Decide -- Guard the Ranking

L22 met this problem (Step 3 and Exercise 2) and named its fix: a **guard** that lets a book
into a list only if enough people rated it. **Your third function** takes one reader's scores, the training-rating
count of every book, and a threshold, and sends every book under the threshold to `-np.inf`.

```python
def apply_guard(scores, counts, min_ratings):
    """Scores with every book that has fewer than min_ratings training ratings set to -inf."""
    # TODO: one line (np.where again)
    raise NotImplementedError   # delete this line when you are done
```

The toy case: scores `[5.0, 4.0, 3.0]`, counts `[100, 10, 50]`, `min_ratings=50` -- a book
with exactly 50 ratings stays in.

```python
ev.check("apply_guard", apply_guard)
```

<details><summary>Expected Output</summary>

~~~text
apply_guard on the toy case: [5.0, -inf, 3.0]
apply_guard: PASS -- from now on the evaluator uses your function
~~~
</details>

Until it is finished, the check prints this instead:

~~~text
apply_guard: not finished yet -- the evaluator keeps using the toolkit's version
~~~

A stricter guard removes more of the tail -- and more of the catalog. Sweep it:

```python
sweep = ev.sweep(cf, guards=[0, 25, 50, 100, 200, 500, 700, 1000])
print(sweep.to_string(index=False))
plot_guard_sweep(sweep, "item-item CF", popularity=p_pop, random=ref["random"])
```

<details><summary>Expected Output</summary>

~~~text
 guard (min ratings)  books left  precision@10
                   0        4000         0.020
                  25        3902         0.021
                  50        2325         0.035
                 100        1107         0.053
                 200         484         0.066
                 500         128         0.081
                 700          65         0.088
                1000          35         0.087
~~~
</details>

> **Reading the sweep:** a guard of 50 nearly doubles CF's precision (0.020 -> 0.035) and still
> leaves 2,325 books; 100 lifts it to 0.053 with 1,107 books. At 500 CF catches popularity (0.081
> for both), and at 700 or 1,000 it edges past it (0.088 and 0.087) -- with only 65 or 35 books
> left. That is a real result: putting the bestsellers in a different order for each reader
> beats one order for everyone. But it is a small gain bought with almost the whole catalog:
> CF can no longer recommend anything a reader would not have found on the bestseller table.
> Every step up buys precision and costs catalog -- the books a reader could discover.

For example, reader 26's CF list with a guard of 100 ratings:

```python
ev.top_k(cf, reader=26, k=5, guard=100)
```

<details><summary>Expected Output</summary>

~~~text
  146 ratings  Under the Banner of Heaven: A Story of Violent Faith
  215 ratings  Storm Front (The Dresden Files, #1)
  113 ratings  Holidays on Ice
  275 ratings  I Know Why the Caged Bird Sings
  119 ratings  Requiem (Delirium, #3)
~~~
</details>

> **Reading the list:** real, widely read books (113 to 275 ratings) instead of the 31-48-rating
> flood. None is a held-out hit for reader 26 -- but one reader's five books prove little; the
> sweep's average over 5,987 readers is the evidence.

**Your decision.** Pick the guard you would ship for CF. There is no single right answer:
a higher threshold buys precision but shrinks the catalog CF may recommend from; past about
500, CF is only re-ordering a few dozen bestsellers. Set `MY_GUARD` and write one or two
sentences on why.

```python
MY_GUARD = None   # TODO: your threshold, e.g. one of the values in the sweep
```

**Your guard and why:**

## Step 6: Final Results and Your Decision

The final table applies your guard to both models (popularity's lists are bestsellers, so
the guard leaves them unchanged).

```python
if MY_GUARD is None:
    print("MY_GUARD is not set yet -- showing the unguarded table. Set it in Step 5 and re-run.")
board = ev.board([pop, cf], ["popularity", "item-item CF"], guard=MY_GUARD or 0)
print(board.to_string())
plot_board(board, guard=MY_GUARD or 0)
```

<details><summary>Expected Output (before you set MY_GUARD)</summary>

~~~text
MY_GUARD is not set yet -- showing the unguarded table. Set it in Step 5 and re-run.
             RMSE (lower=better)  precision@10 (higher=better)
model                                                         
popularity          n/a (ranker)                         0.081
item-item CF               0.838                         0.020
~~~
</details>

With `MY_GUARD = 100` (one possible choice) the table reads:

~~~text
             RMSE (lower=better)  precision@10 (higher=better)
model                                                         
popularity          n/a (ranker)                         0.081
item-item CF               0.838                         0.053
~~~

The guard changes only the lists, never the predicted stars, so CF's RMSE stays 0.838.

**Why is popularity so hard to beat?** The easy story is "everyone likes the famous books".
Test it:

```python
ev.exposure(top=200)
```

<details><summary>Expected Output</summary>

~~~text
the 200 most-rated books are 5% of the catalog
share of held-out ratings on them: 31%
rated 4 or 5 -- on them: 71% | on all other books: 68%
~~~
</details>

> **Reading the numbers:** the 200 most-rated books are 5% of the catalog but hold 31% of all
> held-out ratings -- yet readers rate them 4 or 5 only a little more often than other books (71%
> vs 68%). So "everyone likes the famous books" is not why popularity is so hard to beat. It
> is hard to beat because almost every reader has **read** some of them: the held-out set can
> only contain books a reader chose to read and rate, and the famous books are the ones most
> readers chose. An offline test rewards
> predicting what people *will read*, which a bestseller list does well. It cannot show whether
> a personal list would have led a reader to a book they would never have found -- that takes
> an online test with real readers (an A/B test), which is beyond this course.

### Your decision (written, 4-6 sentences)

Write the paragraph your manager asked for: **which recommender you would ship and why**,
**which metric you would track and why it matches the product**, **your guard**, and **your
plan for brand-new readers** (the cold-start problem from L22/L23). Use the numbers from your
table, and say honestly why popularity scores the way it does.

**Your answer:**

<details><summary>One possible answer</summary>

I would keep popularity as the default "Recommended for you" shelf for now: on precision@10,
the metric that matches a ranked shelf, it scores 0.081, while CF with a guard of 100 ratings
reaches 0.053 -- and CF's good RMSE (0.838) says nothing about its lists. Popularity's lead is
mostly exposure (the test set holds the books people chose to read), and it shows every reader
the same list, so I would also launch guarded CF as a second row ("Because you liked...") or
to a test group of readers. I would keep the guard at 100 so CF still draws on 1,107 books; a
guard of 700 beats popularity only by re-ordering 65 bestsellers. Brand-new readers get the
popularity shelf, since CF has no ratings to use, and move to CF once they have rated a handful
of books. An A/B test with real readers, not this offline number, should decide whether CF
replaces the default.
</details>

## Summary

- You ran the full loop on a new dataset: **build** (popularity, item-item CF) ->
  **evaluate** (RMSE and precision@k, against random and best-possible reference values) ->
  **diagnose** -> **guard** -> **decide**.
- **precision@k** scores the list a reader sees; **RMSE** scores predicted stars. CF won on
  stars (0.838 vs 0.961 for the book mean) and lost on lists (0.020 vs popularity's 0.081) --
  the same model, two verdicts.
- CF's lists flooded with books from the **long tail** (75% under 50 ratings). A **guard**
  trades catalog for precision (0.035 at >= 50, 0.053 at >= 100); CF matches popularity
  at >= 500 and edges past it only when it is left with a few dozen bestsellers (0.088 with
  65 books).
- Popularity is hard to beat offline mostly through **exposure**: the test set holds what
  readers chose to read. Keep it as the baseline to beat and the cold-start shelf; judge a personalized
  model on a ranking metric, and confirm it with real readers.
- A follow-up assignment uses the same toolkit on two more recommenders: the baseline
  (mean + reader bias + book bias) and matrix factorization.
- Unit VI is complete. Next, **Unit VII** swaps the ratings table for a *graph*: social
  networks, neighborhoods, and communities.
