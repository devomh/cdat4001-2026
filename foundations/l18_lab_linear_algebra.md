---
title: "Lab: Linear Algebra in NumPy"
unit: "Foundations"
lesson: "18"
type: lab
tags: [linear-algebra, numpy, vectors, matrices, dot-product, cosine-similarity, geometry, matrix-multiplication]
difficulty: introductory
duration: "75 mins"
---

**Goal:** drill the five operations the quantitative half of this course runs
on -- dot product, matrix-vector, matrix-matrix, transpose, and projection --
on a movie-ratings world small enough to check by hand, and *see* them:
vectors as arrows, the dot product as a shadow, cosine as an angle. Pairs
with the concept note
[Linear Algebra for Data Science](l18_concept_linear_algebra.qmd).

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/devomh/cdat4001-2026/blob/main/foundations/l18_lab_linear_algebra.ipynb)

> This page is the read-only view. To run the lab, open the notebook
> (`l18_lab_linear_algebra.ipynb`) -- in Colab via the badge above, or
> locally. Every text output below comes from actually running the cells.
> Figures, sliders, and the rotatable 3-D plots appear only when you run
> the notebook.

## Prerequisites & Setup

Run the setup cells first. No dataset and no randomness -- the matrices are
hand-built and every result is fully deterministic, so no seed is needed.
We need NumPy (the `ndarray`, shapes, and vectorized arithmetic from L02),
Matplotlib for 2-D pictures, Plotly for 3-D pictures you can rotate, and
ipywidgets for sliders.

```python
# Setup cell 1 of 2: install only (Colab resets on open; run this first)
%pip install -q numpy matplotlib plotly ipywidgets
```

```python
# Setup cell 2 of 2: imports + build the ratings world
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from ipywidgets import interact

np.set_printoptions(precision=3, suppress=True)   # tidy array printing

movies = ["Action1", "Action2", "Romance1", "Romance2", "Comedy"]
users = ["Ana", "Beto", "Carla", "Dani"]

#               Action1 Action2 Romance1 Romance2 Comedy
R = np.array([[5, 4, 1, 1, 3],    # Ana   -- action fan
              [4, 5, 2, 1, 2],    # Beto  -- action fan
              [1, 1, 5, 4, 3],    # Carla -- romance fan
              [2, 1, 4, 5, 3]])   # Dani  -- romance fan

print("Ratings matrix R loaded -- shape", R.shape)
```

<details><summary>Expected Output</summary>

~~~text
Ratings matrix R loaded -- shape (4, 5)
~~~
</details>

### Plotting helpers

Run this cell once. It defines five small drawing functions that the steps
below call: `plot_vectors`, `show_angle`, `show_shadow`, `plot_unit`, and
`plot_vectors_3d`. You do not need to read their code to use them --
each one's first line says what it draws.

```python
# Plotting helpers -- run once; reading the code is optional.
ACTION, ROMANCE, EVA = "crimson", "steelblue", "darkorange"


def _arrow(ax, v, color, label):
    ax.annotate("", xy=(v[0], v[1]), xytext=(0, 0),
                arrowprops=dict(arrowstyle="-|>", color=color,
                                lw=2, shrinkA=0, shrinkB=0))
    if label:
        ax.text(v[0] * 1.06, v[1] * 1.06, label, color=color,
                fontweight="bold")


def _frame(ax, vecs, xlabel="x", ylabel="y"):
    lim = 1.25 * max(np.abs(np.asarray(vecs, float)).max(), 1e-9)
    ax.axhline(0, color="gray", lw=0.6)
    ax.axvline(0, color="gray", lw=0.6)
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect("equal")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)


def plot_vectors(vecs, labels, colors=None,
                 xlabel="x", ylabel="y", title=None):
    """Draw 2-D vectors as arrows from the origin."""
    colors = colors or [f"C{i}" for i in range(len(vecs))]
    fig, ax = plt.subplots(figsize=(5, 5))
    for v, name, c in zip(vecs, labels, colors):
        _arrow(ax, v, c, name)
    _frame(ax, vecs, xlabel, ylabel)
    if title:
        ax.set_title(title)
    plt.show()


def show_angle(a, b, names=("a", "b")):
    """Draw a and b with the angle between them; print the numbers."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    cos = a @ b / (na * nb)
    deg = np.degrees(np.arccos(np.clip(cos, -1, 1)))
    print(f"{names[0]} . {names[1]}  = {a @ b:.3f}")
    print(f"lengths      = {na:.3f}, {nb:.3f}")
    print(f"cos(angle)   = {cos:.3f}")
    print(f"angle        = {deg:.1f} degrees")
    fig, ax = plt.subplots(figsize=(5, 5))
    _arrow(ax, a, ACTION, names[0])
    _arrow(ax, b, ROMANCE, names[1])
    t1 = np.arctan2(a[1], a[0])
    t2 = t1 + np.radians(deg) * np.sign(a[0] * b[1] - a[1] * b[0])
    t = np.linspace(t1, t2, 50)
    r = 0.3 * min(na, nb)
    ax.plot(r * np.cos(t), r * np.sin(t), color="black", lw=1)
    _frame(ax, [a, b])
    ax.set_title(f"angle = {deg:.1f} degrees, cos = {cos:.3f}")
    plt.show()


def show_shadow(b, a, names=("b", "a")):
    """Draw b's shadow on a's direction; print shadow and a . b."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    na = np.linalg.norm(a)
    shadow = a @ b / na
    foot = shadow * a / na
    print(f"shadow of {names[0]} on {names[1]} = {shadow:.3f}")
    print(f"{names[1]} . {names[0]} = shadow x |{names[1]}| = "
          f"{shadow:.3f} x {na:.3f} = {a @ b:.3f}")
    fig, ax = plt.subplots(figsize=(5, 5))
    lim = 1.25 * max(np.abs(np.r_[a, b]).max(), 1e-9)
    u = a / na
    ax.plot([-lim * u[0], lim * u[0]], [-lim * u[1], lim * u[1]],
            color=ACTION, lw=0.8, ls=":")
    color = "seagreen" if shadow >= 0 else "darkorchid"
    ax.plot([0, foot[0]], [0, foot[1]], color=color, lw=7, alpha=0.7)
    ax.plot([b[0], foot[0]], [b[1], foot[1]], color="gray", ls="--")
    _arrow(ax, a, ACTION, names[1])
    _arrow(ax, b, ROMANCE, names[0])
    _frame(ax, [a, b])
    ax.set_title(f"{names[1]} . {names[0]} = {a @ b:.3f}")
    plt.show()


def plot_unit(vecs, labels, colors=None):
    """Shrink each 2-D vector to length 1 and draw the unit circle."""
    unit = [np.asarray(v, float) / np.linalg.norm(v) for v in vecs]
    colors = colors or [f"C{i}" for i in range(len(vecs))]
    fig, ax = plt.subplots(figsize=(5, 5))
    t = np.linspace(0, 2 * np.pi, 200)
    ax.plot(np.cos(t), np.sin(t), color="gray", lw=1)
    for v, name, c in zip(unit, labels, colors):
        _arrow(ax, v, c, name)
    _frame(ax, [[1, 1]])
    ax.set_title("every arrow shrunk to length 1")
    plt.show()


def plot_vectors_3d(vecs, labels, colors=None,
                    axis_names=("x", "y", "z"), title=None,
                    diagonal=False):
    """Draw 3-D vectors as arrows you can rotate (drag the plot)."""
    colors = colors or ["crimson", "steelblue", "seagreen",
                        "darkorange", "purple"]
    fig = go.Figure()
    if diagonal:
        # dashed line through the origin where all 3 values are equal
        m = np.abs(np.asarray(vecs, float)).max()
        fig.add_trace(go.Scatter3d(
            x=[-m, m], y=[-m, m], z=[-m, m], mode="lines",
            line=dict(color="gray", width=3, dash="dash"),
            name="diagonal"))
    for v, name, c in zip(vecs, labels, colors):
        fig.add_trace(go.Scatter3d(
            x=[0, v[0]], y=[0, v[1]], z=[0, v[2]],
            mode="lines+markers+text", text=["", name],
            line=dict(color=c, width=6),
            marker=dict(size=[0, 4], color=c), name=name))
    fig.update_layout(
        height=480, showlegend=False, title=title,
        margin=dict(l=0, r=0, t=40 if title else 10, b=0),
        scene=dict(xaxis_title=axis_names[0],
                   yaxis_title=axis_names[1],
                   zaxis_title=axis_names[2], aspectmode="cube"))
    fig.show()


print("helpers ready")
```

<details><summary>Expected Output</summary>

~~~text
helpers ready
~~~
</details>

## Step 1: Data as Vectors and Matrices (Worked)

The whole table is a **matrix**; one user's ratings are a **vector** (a row);
one movie's ratings across users are a column.

```python
print("R shape:", R.shape, "(users x movies)")
print("Ana's row (a vector):", R[0])
print("Romance1 column:", R[:, 2])
print("Ana's rating of Comedy R[0, 4]:", R[0, 4])
```

<details><summary>Expected Output</summary>

~~~text
R shape: (4, 5) (users x movies)
Ana's row (a vector): [5 4 1 1 3]
Romance1 column: [1 2 5 4]
Ana's rating of Comedy R[0, 4]: 3
~~~
</details>

> **Read it:** `R.shape` is `(4, 5)` -- 4 users, 5 movies -- and saying that
> out loud is the habit that prevents most linear-algebra bugs. `R[0]` pulls
> one row (a length-5 vector); `R[:, 2]` pulls one column (a length-4
> vector); `R[0, 4]` is a single number.

## Step 2: The Dot Product (Worked)

The dot product multiplies two equal-length vectors element by element and
sums -- one number out. Three ways to write it, all identical:

```python
ana, beto, carla = R[0], R[1], R[2]
print("Ana . Beto manual:", int(np.sum(ana * beto)))
print("Ana . Beto np.dot:", int(np.dot(ana, beto)))
print("Ana . Beto  ana@beto:", int(ana @ beto))
print("Ana . Carla:", int(ana @ carla))
print("Ana . Ana:", int(ana @ ana))
```

<details><summary>Expected Output</summary>

~~~text
Ana . Beto manual: 49
Ana . Beto np.dot: 49
Ana . Beto  ana@beto: 49
Ana . Carla: 27
Ana . Ana: 52
~~~
</details>

> **Read it:** `ana * beto` is elementwise (a length-5 array); wrapping it in
> `np.sum` collapses it to one number, which is exactly what `np.dot` and `@`
> compute. Ana aligns far more with Beto (49) than with Carla (27) -- the dot
> product is reading taste similarity. The next step shows *why*.

## Step 3: Arrows, Angles, and Shadows (Worked + Explore)

Describe each user by two **taste** numbers -- how much they like action,
how much romance. Two numbers make a point on a plane, or an **arrow** from
the origin. Ana's taste is all action, Carla's all romance:

```python
ana_t = np.array([1.0, 0.2])     # [action, romance]
carla_t = np.array([0.2, 1.0])
plot_vectors([ana_t, carla_t], ["Ana", "Carla"],
             colors=[ACTION, ROMANCE],
             xlabel="action", ylabel="romance")
```

The dot product has a geometric formula:
`a . b = |a| |b| cos(angle)`, where `|a|` is the arrow's length
(`np.linalg.norm`). `show_angle` computes both sides:

```python
show_angle(ana_t, carla_t, names=("Ana", "Carla"))
```

<details><summary>Expected Output</summary>

~~~text
Ana . Carla  = 0.400
lengths      = 1.020, 1.020
cos(angle)   = 0.385
angle        = 67.4 degrees
~~~
</details>

> **Read it:** multiply-and-add gives `1.0*0.2 + 0.2*1.0 = 0.4`; the
> geometric formula gives `1.020 * 1.020 * 0.385 = 0.4` -- the same number,
> read two ways. The two tastes sit 67.4 degrees apart.

Now the **shadow** picture: drop a line from the tip of Carla's arrow
straight onto Ana's direction. The dot product is that shadow times Ana's
length:

```python
show_shadow(carla_t, ana_t, names=("Carla", "Ana"))
```

<details><summary>Expected Output</summary>

~~~text
shadow of Carla on Ana = 0.392
Ana . Carla = shadow x |Ana| = 0.392 x 1.020 = 0.400
~~~
</details>

> **Read it:** the shadow is how far Carla's taste reaches *in Ana's
> direction* -- not far (0.392). A big dot product means two arrows reach
> the same way; a small one means they mostly don't.

**Explore with a slider.** Run the next cell, then drag the sliders: turn
`b` around a fixed `a` and change its length. Watch the shadow (green when
positive, purple when negative) and the printed dot product.

```python
a_fixed = np.array([1.0, 0.2])


def turn_b(angle=67, length=1.0):
    start = np.arctan2(a_fixed[1], a_fixed[0])
    t = start + np.radians(angle)
    b = length * np.array([np.cos(t), np.sin(t)])
    show_shadow(b, a_fixed)


# interact() builds one slider per argument from (min, max, step).
# The ; stops Jupyter from also printing interact's return value.
interact(turn_b, angle=(0, 180, 1), length=(0.25, 1.5, 0.05));
```

No slider (or reading this page)? The same experiment at three fixed
angles, with `a = [1, 0]` and `b` of length 1, as in the concept's
sign figure:

```python
for angle in [40, 90, 140]:
    t = np.radians(angle)
    b = np.array([np.cos(t), np.sin(t)])
    a = np.array([1.0, 0.0])
    print(f"angle {angle:3d} degrees -> a . b = {a @ b:+.3f}")
```

<details><summary>Expected Output</summary>

~~~text
angle  40 degrees -> a . b = +0.766
angle  90 degrees -> a . b = +0.000
angle 140 degrees -> a . b = -0.766
~~~
</details>

> **Read it:** under 90 degrees the shadow points forward and the dot
> product is **positive**; at exactly 90 there is no shadow and it is
> **zero** (the vectors are **orthogonal** -- "unrelated"); past 90 the
> shadow points backward and it goes **negative**.

## Step 4: Cosine Similarity (Worked + Completion)

Raw dot products grow with length, so **cosine similarity** divides by both
lengths, leaving only the angle:

```python
def cosine(a, b):
    return a @ b / (np.linalg.norm(a) * np.linalg.norm(b))

print(f"cosine(Ana, Beto):  {cosine(ana, beto):.3f}")
print(f"cosine(Ana, Carla): {cosine(ana, carla):.3f}")
print(f"|Ana| = sqrt(Ana.Ana) = {np.linalg.norm(ana):.3f}")
for name, other in [("Beto", beto), ("Carla", carla)]:
    deg = np.degrees(np.arccos(cosine(ana, other)))
    print(f"Ana vs {name}: {deg:.1f} degrees apart")
```

<details><summary>Expected Output</summary>

~~~text
cosine(Ana, Beto):  0.961
cosine(Ana, Carla): 0.519
|Ana| = sqrt(Ana.Ana) = 7.211
Ana vs Beto: 16.1 degrees apart
Ana vs Carla: 58.7 degrees apart
~~~
</details>

> **Read it:** cosine lands in [-1, 1] and ignores length: two action fans
> sit 16.1 degrees apart (cosine 0.96), an action fan and a romance fan 58.7
> degrees apart (0.52). The rating rows have five numbers, so you cannot draw
> them -- but any two arrows share one flat plane, and in it they make an
> ordinary angle. This is the similarity engine behind the recommenders in
> Unit VI.

Dividing by the lengths is the same as shrinking every arrow to length 1
first. Here are the four 2-D tastes on the unit circle:

```python
U = np.array([[1.0, 0.2],    # Ana    [action-taste, romance-taste]
              [0.9, 0.1],    # Beto
              [0.2, 1.0],    # Carla
              [0.1, 0.9]])   # Dani
plot_unit(U, users, colors=[ACTION, ACTION, ROMANCE, ROMANCE])
```

> **Read it:** on the unit circle only direction is left. Ana and Beto
> nearly overlap; Carla and Dani nearly overlap; an action fan and a romance
> fan sit 67 to 77 degrees apart. (We meet `U` again in Step 7.)

Your turn -- how alike are Beto (action) and Carla (romance)?

```python
# Uncomment and fill the ____ : reuse the cosine() function on beto and carla.
# print(f"cosine(Beto, Carla): {cosine(beto, ____):.3f}")
```

<details><summary>Expected Output</summary>

~~~text
cosine(Beto, Carla): 0.569
~~~
*(Cross-taste, like Ana-vs-Carla: moderate, well below the 0.96 of two fans
of the same genre.)*
</details>

## Step 5: The Positive Corner, and Shifting the Origin (Worked)

Ana and Carla have *opposite* tastes, yet their cosine is +0.52. Why not
negative? Plot every user on two movies, Action1 and Romance1 -- and add
**Eva**, who rates every movie 5:

```python
eva = np.array([5, 5, 5, 5, 5])
two = [0, 2]                     # Action1, Romance1
plot_vectors([R[0, two], R[1, two], R[2, two], R[3, two], eva[two]],
             users + ["Eva"],
             colors=[ACTION, ACTION, ROMANCE, ROMANCE, EVA],
             xlabel="Action1", ylabel="Romance1",
             title="all ratings positive: one corner")

print("R @ eva:", R @ eva)
for name, row in zip(users, R):
    print(f"cosine({name}, Eva) = {cosine(row, eva):.3f}")
```

<details><summary>Expected Output</summary>

~~~text
R @ eva: [70 70 70 75]
cosine(Ana, Eva) = 0.868
cosine(Beto, Eva) = 0.885
cosine(Carla, Eva) = 0.868
cosine(Dani, Eva) = 0.905
~~~
</details>

> **Read it:** ratings are positive, so every arrow lives in the positive
> **corner**, and two arrows in that corner are never more than 90 degrees
> apart -- the cosine cannot go negative. Eva's arrow runs down the
> diagonal, close to everyone: her raw dot products (70-75) beat Ana-Beto's
> 49, and even her cosines stay high (0.87-0.91).

The fix is to **shift the origin**: subtract each user's own average
rating (**mean-centering**). Above-average becomes positive, below-average
negative, and the arrows can leave the corner:

```python
# keepdims=True keeps one average per row (a 4 x 1 column),
# so each row subtracts its own average.
R_c = R - R.mean(axis=1, keepdims=True)
print("Ana centered:", R_c[0])
print(f"Ana vs Carla: raw {cosine(R[0], R[2]):+.3f}"
      f" -> centered {cosine(R_c[0], R_c[2]):+.3f}")
print(f"Ana vs Beto:  raw {cosine(R[0], R[1]):+.3f}"
      f" -> centered {cosine(R_c[0], R_c[1]):+.3f}")

eva_c = eva - eva.mean()
print("Eva centered:", eva_c, "| length:", np.linalg.norm(eva_c))
```

<details><summary>Expected Output</summary>

~~~text
Ana centered: [ 2.2  1.2 -1.8 -1.8  0.2]
Ana vs Carla: raw +0.519 -> centered -0.953
Ana vs Beto:  raw +0.961 -> centered +0.834
Eva centered: [0. 0. 0. 0. 0.] | length: 0.0
~~~
</details>

> **Read it:** Ana and Carla swing from +0.519 to **-0.953** -- opposite
> tastes, finally opposite numbers. Ana and Beto stay alike (0.834). Eva
> becomes the **zero vector**: length 0, no direction, so her cosine is
> undefined (0 divided by 0) -- a user who rates everything the same tells
> you nothing about taste. This centered cosine is the **correlation** of
> L19, and the way the L22 recommenders compare users.

See it in 3-D with three movies as the axes. **Drag to rotate** each plot:

```python
three = [0, 1, 2]                # Action1, Action2, Romance1
names3 = ["Action1", "Action2", "Romance1"]
colors5 = [ACTION, ACTION, ROMANCE, ROMANCE, EVA]
raw3 = np.vstack([R[:, three], eva[three]]).astype(float)
cen3 = raw3 - raw3.mean(axis=1, keepdims=True)

plot_vectors_3d(raw3, users + ["Eva"], colors5, names3,
                title="raw ratings", diagonal=True)
plot_vectors_3d(cen3, users + ["Eva"], colors5, names3,
                title="mean-centered", diagonal=True)
print(f"3 movies, Ana vs Carla: raw {cosine(raw3[0], raw3[2]):+.3f}"
      f" -> centered {cosine(cen3[0], cen3[2]):+.3f}")
```

<details><summary>Expected Output</summary>

~~~text
3 movies, Ana vs Carla: raw +0.416 -> centered -0.971
~~~
</details>

> **Read it:** raw, all five arrows huddle in one corner around the dashed
> diagonal. Centered, rotate until you look along the dashed diagonal (it
> shrinks to a point): every arrow
> lies on one flat plane through the origin, the action fans point one way,
> the romance fans the other, and Eva has shrunk to a dot at the origin.

## Step 6: Matrix times Vector (Worked + Completion)

A dot product compares two vectors; a **matrix-vector product** runs one dot
product for every row at once. Give each movie an `[action, romance]` feature
pair (matrix `F`, shape `5 x 2`) and a user a taste vector `w`; then `F @ w`
scores all five movies in one multiply.

```python
F = np.array([[1.0, 0.0],    # Action1   [action, romance]
              [0.9, 0.1],    # Action2
              [0.0, 1.0],    # Romance1
              [0.1, 0.9],    # Romance2
              [0.5, 0.5]])   # Comedy
w_ana = np.array([1.0, 0.2])     # Ana: loves action, mild on romance

scores = F @ w_ana
print("F shape:", F.shape, "| w_ana shape:", w_ana.shape, "| scores shape:", scores.shape)
print("Ana's predicted taste-scores per movie:")
for m, s in zip(movies, scores):
    print(f"  {m:9s} {s:.2f}")
```

<details><summary>Expected Output</summary>

~~~text
F shape: (5, 2) | w_ana shape: (2,) | scores shape: (5,)
Ana's predicted taste-scores per movie:
  Action1   1.00
  Action2   0.92
  Romance1  0.20
  Romance2  0.28
  Comedy    0.60
~~~
</details>

> **Read it:** each score is one dot product (a movie's features with Ana's
> taste), and the shapes confirm it: `(5, 2) @ (2,) -> (5,)`, the shared `2`
> cancels. One operation scored all five movies, action on top -- exactly
> what a recommender does for one user.

Your turn -- score the movies for Carla, who loves romance: `w = [0.2, 1.0]`.

```python
# Uncomment and fill the ____ : build Carla's taste vector, then F @ w.
# w_carla = np.array([0.2, ____])
# carla_scores = F @ w_carla
# print("Carla's predicted taste-scores per movie:")
# for m, s in zip(movies, carla_scores):
#     print(f"  {m:9s} {s:.2f}")
```

<details><summary>Expected Output</summary>

~~~text
Carla's predicted taste-scores per movie:
  Action1   0.20
  Action2   0.28
  Romance1  1.00
  Romance2  0.92
  Comedy    0.60
~~~
*(Same matrix `F`, a different taste vector -- now the romance movies rise to
the top. The features are fixed; the user vector picks what matters.)*
</details>

**The taste dial.** Geometrically, each movie is a point in the
action-romance plane and the taste `w` is a direction; each score is the
movie's shadow on that direction. Turn the dial (0 degrees = pure action,
90 = pure romance) and watch the ranking change:

```python
def taste_dial(degrees=20):
    t = np.radians(degrees)
    w = np.array([np.cos(t), np.sin(t)])   # length-1 taste
    # round first, so tied scores keep the catalogue order
    order = np.argsort(-np.round(F @ w, 6), kind="stable")
    print(f"taste w = [{w[0]:.2f}, {w[1]:.2f}]  ranking:")
    for i in order:
        print(f"  {movies[i]:9s} {F[i] @ w:.2f}")


interact(taste_dial, degrees=(0, 90, 5));
```

No slider? Three dial positions:

```python
for degrees in [0, 45, 70]:
    taste_dial(degrees)
```

<details><summary>Expected Output</summary>

~~~text
taste w = [1.00, 0.00]  ranking:
  Action1   1.00
  Action2   0.90
  Comedy    0.50
  Romance2  0.10
  Romance1  0.00
taste w = [0.71, 0.71]  ranking:
  Action1   0.71
  Action2   0.71
  Romance1  0.71
  Romance2  0.71
  Comedy    0.71
taste w = [0.34, 0.94]  ranking:
  Romance1  0.94
  Romance2  0.88
  Comedy    0.64
  Action2   0.40
  Action1   0.34
~~~
</details>

> **Read it:** at 0 degrees the action movies cast the longest shadows; at
> 70 the romance movies do. At 45 degrees every movie ties at 0.71 -- each
> one's two features add to 1, so a perfectly balanced taste cannot tell them
> apart (tied movies stay in catalogue order). Turning one dial reorders
> the whole catalogue: that is matrix-times-vector, seen as shadows.

## Step 7: Matrix times Matrix (Worked)

Stack every user's taste into a matrix `U` (`4 x 2`, built in Step 4) and you
can score *all users against all movies at once*. We need the movie features
as `2 x 5`, which is the **transpose** `F.T` (rows become columns):

```python
V = F.T                          # (5, 2) -> (2, 5): the two factors as rows
print("U shape:", U.shape, "| V = F.T shape:", V.shape)

R_hat = U @ V                    # (4, 2) @ (2, 5) -> (4, 5)
print("R_hat = U @ V shape:", R_hat.shape, "(users x movies)")
print(R_hat)
print("Ana's R_hat row equals Step-6 scores?", np.allclose(R_hat[0], scores))
```

<details><summary>Expected Output</summary>

~~~text
U shape: (4, 2) | V = F.T shape: (2, 5)
R_hat = U @ V shape: (4, 5) (users x movies)
[[1.   0.92 0.2  0.28 0.6 ]
 [0.9  0.82 0.1  0.18 0.5 ]
 [0.2  0.28 1.   0.92 0.6 ]
 [0.1  0.18 0.9  0.82 0.5 ]]
Ana's R_hat row equals Step-6 scores? True
~~~
</details>

> **Read it:** `U @ V` is a full `4 x 5` table -- every user's predicted score
> for every movie -- and Ana's row is byte-for-byte the Step-6 result, because
> matrix-matrix multiplication is just the matrix-vector product run for every
> user at once. Predicting all ratings from two compact factor matrices is
> **matrix factorization**, the recommender method of Unit VI.

## Step 8: Transpose, Identity, Shapes, and Projection (Worked + Completion)

The **identity** matrix is the do-nothing matrix (`A @ I == A`); the
**inner-dimension rule** tells you a product's shape before you run it:

```python
I2 = np.eye(2)
print("Identity I2:")
print(I2)
print("F @ I2 == F ?", np.allclose(F @ I2, F))
print("Shape rule: (m,n)@(n,p)=(m,p). U(4,2)@V(2,5) inner 2==2 -> (4,5). OK")
```

<details><summary>Expected Output</summary>

~~~text
Identity I2:
[[1. 0.]
 [0. 1.]]
F @ I2 == F ? True
Shape rule: (m,n)@(n,p)=(m,p). U(4,2)@V(2,5) inner 2==2 -> (4,5). OK
~~~
</details>

**Projection** -- the operation all of L20 (PCA) rests on -- is the Step-3
shadow onto a **unit** direction (length 1, so the dot product *is* the
shadow). Project each user's 2-D taste row onto the "action-versus-romance"
axis and a single number says where they fall:

```python
d = np.array([1.0, -1.0]) / np.sqrt(2)   # unit axis: action minus romance
coords = U @ d                           # (4, 2) @ (2,) -> (4,)
print("1D projection of each user onto the action-vs-romance axis:")
for u, c in zip(users, coords):
    print(f"  {u:6s} {c:+.3f}")
```

<details><summary>Expected Output</summary>

~~~text
1D projection of each user onto the action-vs-romance axis:
  Ana    +0.566
  Beto   +0.566
  Carla  -0.566
  Dani   -0.566
~~~
</details>

> **Read it:** two action fans at `+0.566`, two romance fans at `-0.566` --
> a 2-D dataset collapsed to one axis that still separates the groups. That
> is the PCA idea (L20), where the axis is *found* for you rather than
> handed to you. (`show_shadow(U[0], d)` draws Ana's shadow on this axis.)

Your turn -- predict a shape, then check it. Is `U.T @ R_hat` legal, and what
shape is it? (`U` is `4 x 2`, `R_hat` is `4 x 5`.)

```python
# Uncomment and fill the ____ : transpose U so the inner dimensions match,
# then read the result shape. U.T is (2, 4); R_hat is (4, 5).
# result = U.____ @ R_hat
# print("U.T @ R_hat shape:", result.shape)
```

<details><summary>Expected Output</summary>

~~~text
U.T @ R_hat shape: (2, 5)
~~~
*(`U` itself (4x2) cannot multiply `R_hat` (4x5) -- inner 2 != 4. Transposing
to `U.T` (2x4) makes the inner dimensions 4==4, giving a (2, 5) result.)*
</details>

## Your Turn (Exercises)

Exercises 1-7 are the core; 8 and 9 are optional extensions.

### Exercise 1 -- Nearest neighbor by cosine

For each user, find their most similar *other* user by cosine similarity, and
print the pair with the score. One sentence: do the action fans and romance
fans pair up as expected?

> **Hint:** loop over users; for user `i`, compute `cosine(R[i], R[j])` for
> every `j != i` and keep the largest. Reuse the `cosine` function from
> Step 4.

```python
# TODO: your code here
```

<details><summary>Expected Output (one possible answer -- match the values, not the print format)</summary>

~~~text
Ana    -> Beto   (cos 0.961)
Beto   -> Ana    (cos 0.961)
Carla  -> Dani   (cos 0.972)
Dani   -> Carla  (cos 0.972)
~~~
*(The two action fans pick each other; the two romance fans pick each other.
This pairing-by-similarity is the seed of user-user collaborative filtering
in Unit VI.)*
</details>

### Exercise 2 -- Rotate and watch

Fix `a = [1, 0]`. For angles 0, 30, 60, ..., 180 degrees, build `b` of
**length 2** at that angle, and print the angle, `a @ b`, and
`cosine(a, b)`. Then plot the cosine against the angle. Where does the
cosine cross zero, and why is `a @ b` always exactly twice the cosine here?

> **Hint:** `t = np.radians(angle)`; `b = 2 * np.array([np.cos(t), np.sin(t)])`.
> Collect the cosines in a list and `plt.plot(angles, cosines, marker="o")`.

```python
# TODO: your code here
```

<details><summary>Expected Output (one possible answer -- match the values, not the print format)</summary>

~~~text
angle   a.b     cos
    0  +2.000  +1.000
   30  +1.732  +0.866
   60  +1.000  +0.500
   90  +0.000  +0.000
  120  -1.000  -0.500
  150  -1.732  -0.866
  180  -2.000  -1.000
~~~
*(The cosine crosses zero at 90 degrees -- perpendicular, no shadow -- and is
negative beyond it. `a @ b = |a| |b| cos = 1 x 2 x cos`, so the length of `b`
only rescales the dot product; the cosine does not see it.)*
</details>

### Exercise 3 -- Predict, then check

Invent 2-D taste vectors `[action, romance]` for yourself and a friend.
**Before computing**, draw them with `plot_vectors` and write down your
guess for the angle between them. Then run `show_angle` and compare. How far
off was your guess?

> **Hint:** `plot_vectors([me, friend], ["me", "friend"])`, then
> `show_angle(me, friend, names=("me", "friend"))`.

```python
# TODO: your code here
```

<details><summary>Expected Output (one possible answer -- your vectors and numbers will differ)</summary>

~~~text
me . friend  = 0.780
lengths      = 1.000, 0.949
cos(angle)   = 0.822
angle        = 34.7 degrees
~~~
*(With `me = [0.8, 0.6]` and `friend = [0.3, 0.9]`. Most guesses from the
picture land within 10 degrees -- the angle is something you can see.)*
</details>

### Exercise 4 -- Stretching doesn't matter

Multiply Ana's taste `ana_t` by 0.5, 2, and 10. For each version print its
dot product with `carla_t` and its cosine with `carla_t`. Which number
changes and which does not? If a user rated everything twice as strongly,
which of the two should a recommender use to find similar users?

```python
# TODO: your code here
```

<details><summary>Expected Output (one possible answer -- match the values, not the print format)</summary>

~~~text
scale  0.5: dot 0.200  cos 0.385
scale  2.0: dot 0.800  cos 0.385
scale 10.0: dot 4.000  cos 0.385
~~~
*(The dot product scales with the arrow's length; the cosine does not,
because stretching an arrow does not turn it. A recommender comparing tastes
wants the cosine -- intensity is not taste.)*
</details>

### Exercise 5 -- Unrelated is not opposite

Build a taste vector `v` that is **orthogonal** to Ana's `ana_t = [1.0, 0.2]`
(dot product 0), and also Ana's **opposite**, `-ana_t`. Print the cosine of
each with Ana, then score the five movies for both with `F @ v` and
`F @ -ana_t`. Look at the **signs** of the scores: in one sentence, what is
the difference between "orthogonal to Ana" and "opposite of Ana"?

> **Hint:** swapping the two numbers and flipping one sign, `[-0.2, 1.0]`,
> gives a dot product of `-0.2 + 0.2 = 0`.

```python
# TODO: your code here
```

<details><summary>Expected Output (one possible answer -- match the values, not the print format)</summary>

~~~text
cos(Ana, v)    = +0.000
cos(Ana, -Ana) = -1.000
F @ v:     [-0.2  -0.08  1.    0.88  0.4 ]
F @ -Ana:  [-1.   -0.92 -0.2  -0.28 -0.6 ]
~~~
*(`-Ana` points straight against Ana: it scores **every** movie negative,
most of all the action movies Ana loves -- it is Ana's taste with every
sign flipped. `v` is at 90 degrees: it casts no shadow on Ana's direction,
so it neither leans toward nor against Ana's taste as a whole. Its own
scores show a clear romance fan, mildly negative on action. On this
catalogue both happen to rank the movies in the same order -- every
movie's two features add to 1, so any romance-leaning taste does (Step 6)
-- which is why the signs, not the ranking, tell them apart.)*
</details>

### Exercise 6 -- Dropped dimensions in 3-D

Keep only three movies -- Action1, Romance1, Comedy (columns `[0, 2, 4]`)
-- and draw the four users with `plot_vectors_3d`. Rotate the plot. Then
compute the cosine of every pair of users using only these three movies, and
compare each with its full five-movie cosine. Which pair changes the most,
and why might dropping movies change who looks similar?

> **Hint:** `R3 = R[:, [0, 2, 4]]`; loop over pairs `i < j` and print
> `cosine(R[i], R[j])` next to `cosine(R3[i], R3[j])`.

```python
# TODO: your code here
```

<details><summary>Expected Output (one possible answer -- match the values, not the print format)</summary>

~~~text
Ana   - Beto  : 5 movies 0.961 | 3 movies 0.966
Ana   - Carla : 5 movies 0.519 | 3 movies 0.543
Ana   - Dani  : 5 movies 0.598 | 3 movies 0.722
Beto  - Carla : 5 movies 0.569 | 3 movies 0.690
Beto  - Dani  : 5 movies 0.610 | 3 movies 0.834
Carla - Dani  : 5 movies 0.972 | 3 movies 0.973
~~~
*(Beto-Dani jumps from 0.610 to 0.834. Dropping Action2 and Romance2 throws
away exactly the movies on which Beto and Dani disagree most, and keeps
Comedy, which every user rates 2 or 3. Fewer dimensions can hide
differences -- the cost PCA (L20) tries to keep small by choosing which
directions to keep.)*
</details>

### Exercise 7 -- What centering changes

Center `R` the way Step 5 did (`R_c`). For every pair of users, print the
raw cosine next to the centered cosine. Which pairs turn negative, and which
stay positive? Then rerun Exercise 1's nearest-neighbor search on `R_c`:
does anyone's nearest neighbor change? In two sentences, say what centering
changed and what it did not.

> **Hint:** loop over pairs `i < j` as in Exercise 6, printing
> `cosine(R[i], R[j])` and `cosine(R_c[i], R_c[j])`.

```python
# TODO: your code here
```

<details><summary>Expected Output (one possible answer -- match the values, not the print format)</summary>

~~~text
Ana   - Beto  : raw +0.961 | centered +0.834
Ana   - Carla : raw +0.519 | centered -0.953
Ana   - Dani  : raw +0.598 | centered -0.884
Beto  - Carla : raw +0.569 | centered -0.868
Beto  - Dani  : raw +0.610 | centered -0.962
Carla - Dani  : raw +0.972 | centered +0.884
Ana    -> Beto   (centered cos +0.834)
Beto   -> Ana    (centered cos +0.834)
Carla  -> Dani   (centered cos +0.884)
Dani   -> Carla  (centered cos +0.884)
~~~
*(All four action-fan/romance-fan pairs turn negative, and the two same-genre
pairs stay positive. Raw, every pair sat between 0.52 and 0.97 because all
the arrows shared the positive corner. Nobody's nearest neighbor changes:
centering does not reorder who is *most* alike here, but it makes "unlike"
show up as negative, so the numbers now separate agreement from
disagreement.)*
</details>

### Exercise 8 (optional) -- Distance versus angle

The **distance** between two arrow tips is `np.linalg.norm(a - b)` (L21
uses it to cluster). Compare Ana's taste with "double Ana" (`2 * ana_t`)
and with Carla (`carla_t`): print the distance and the cosine for each pair.
Which pair is closer by distance? By cosine? When would you want each?

```python
# TODO: your code here
```

<details><summary>Expected Output (one possible answer -- match the values, not the print format)</summary>

~~~text
Ana vs 2xAna: distance 1.020  cos 1.000
Ana vs Carla: distance 1.131  cos 0.385
~~~
*(By distance, "double Ana" is almost as far from Ana as Carla is; by
cosine, double Ana is identical and Carla is not. Distance mixes length and
direction; cosine keeps only direction. Use cosine for "same taste", and
distance when the size of the numbers matters too.)*
</details>

### Exercise 9 (optional) -- Written: the shape of a projection

In L20 you will project a data matrix `X` with shape `(n, d)` -- `n` rows
(samples), `d` columns (features) -- down to `k` dimensions for plotting. In
2-3 sentences: what shape must the projection matrix `W` have so that
`X @ W` is legal, and what shape does the result have? Use the
inner-dimension rule, and say in plain words what `k` controls.

> **Hint:** `(n, d) @ (?, ?) -> (n, k)`. Which inner dimensions must match?

## Summary

| Operation | NumPy | What it gave us |
|-----------|-------|-----------------|
| Read a shape | `R.shape` | `(4, 5)` = users x movies; the bug-prevention habit |
| Dot product | `a @ b` | one number: weighted sum / similarity (Ana.Beto = 49) |
| Dot product, geometrically | `norm(a) * norm(b) * cos` | the shadow of one arrow on the other, times its length |
| Cosine similarity | `a @ b / (norm(a)*norm(b))` | direction only, no length (0.96 = 16.1 deg vs 0.52 = 58.7 deg) |
| Angle in degrees | `np.degrees(np.arccos(cos))` | "how far apart" as an angle, in any dimension |
| Mean-centering | `R - R.mean(axis=1, keepdims=True)` | leaves the positive corner: Ana-Carla +0.52 -> -0.95 |
| Matrix x vector | `F @ w` | score every movie for one user (shadows on the taste direction) |
| Matrix x matrix | `U @ V` | score all users x all movies (the factorization idea) |
| Transpose / identity | `A.T`, `np.eye(n)` | line up shapes; the do-nothing matrix |
| Projection | `points @ d` | collapse to one axis -- the PCA preview |

Next lesson (L19): the centered cosine gets its full treatment -- center two
*columns* (variables) instead of two users and it becomes the **correlation
coefficient**, the measure of how two variables move together.
