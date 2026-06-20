---
title: "Lab: Linear Algebra in NumPy"
unit: "Foundations"
lesson: "18"
type: lab
tags: [linear-algebra, numpy, vectors, matrices, dot-product, matrix-multiplication]
difficulty: introductory
duration: "60 mins"
---

**Goal:** drill the five operations the quantitative half of this course runs
on -- dot product, matrix-vector, matrix-matrix, transpose, and projection --
on a movie-ratings world small enough to check by hand. Pairs with the
concept note
[Linear Algebra for Data Science](l18_concept_linear_algebra.qmd).

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/devomh/cdat4001-2026/blob/main/foundations/l18_lab_linear_algebra.ipynb)

> This page is the read-only view. To run the lab, open the notebook
> (`l18_lab_linear_algebra.ipynb`) -- in Colab via the badge above, or
> locally. Every text output below comes from actually running the cells.

## Prerequisites & Setup

Run the setup cells first. No dataset and no randomness -- the matrices are
hand-built and every result is fully deterministic, so no seed is needed.
We only need NumPy (the `ndarray`, shapes, and vectorized arithmetic from
L02).

```python
# Setup cell 1 of 2: install only (Colab resets on open; run this first)
%pip install -q numpy
```

```python
# Setup cell 2 of 2: imports + build the ratings world
import numpy as np
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

## Step 2: The Dot Product (Worked + Completion)

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
> product is reading taste similarity.

Raw dot products are inflated by magnitude, so **cosine similarity** divides
by the vectors' lengths, leaving only the angle:

```python
def cosine(a, b):
    return a @ b / (np.linalg.norm(a) * np.linalg.norm(b))

print(f"cosine(Ana, Beto):  {cosine(ana, beto):.3f}")
print(f"cosine(Ana, Carla): {cosine(ana, carla):.3f}")
print(f"|Ana| = sqrt(Ana.Ana) = {np.linalg.norm(ana):.3f}")
```

<details><summary>Expected Output</summary>

~~~text
cosine(Ana, Beto):  0.961
cosine(Ana, Carla): 0.519
|Ana| = sqrt(Ana.Ana) = 7.211
~~~
</details>

> **Read it:** cosine lands in [-1, 1] and ignores magnitude: two action
> fans sit at 0.96 (nearly the same direction), an action fan and a romance
> fan at 0.52. This is the similarity engine behind the recommenders in
> Unit VI.

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

## Step 3: Matrix times Vector (Worked + Completion)

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

## Step 4: Matrix times Matrix (Worked)

Stack every user's taste into a matrix `U` (`4 x 2`) and you can score *all
users against all movies at once*. We need the movie features as `2 x 5`,
which is the **transpose** `F.T` (rows become columns):

```python
U = np.array([[1.0, 0.2],    # Ana    [action-taste, romance-taste]
              [0.9, 0.1],    # Beto
              [0.2, 1.0],    # Carla
              [0.1, 0.9]])   # Dani
V = F.T                          # (5, 2) -> (2, 5): the two factors as rows
print("U shape:", U.shape, "| V = F.T shape:", V.shape)

R_hat = U @ V                    # (4, 2) @ (2, 5) -> (4, 5)
print("R_hat = U @ V shape:", R_hat.shape, "(users x movies)")
print(R_hat)
print("Ana's R_hat row equals Step-3 scores?", np.allclose(R_hat[0], scores))
```

<details><summary>Expected Output</summary>

~~~text
U shape: (4, 2) | V = F.T shape: (2, 5)
R_hat = U @ V shape: (4, 5) (users x movies)
[[1.   0.92 0.2  0.28 0.6 ]
 [0.9  0.82 0.1  0.18 0.5 ]
 [0.2  0.28 1.   0.92 0.6 ]
 [0.1  0.18 0.9  0.82 0.5 ]]
Ana's R_hat row equals Step-3 scores? True
~~~
</details>

> **Read it:** `U @ V` is a full `4 x 5` table -- every user's predicted score
> for every movie -- and Ana's row is byte-for-byte the Step-3 result, because
> matrix-matrix multiplication is just the matrix-vector product run for every
> user at once. Predicting all ratings from two compact factor matrices is
> **matrix factorization**, the recommender method of Unit VI.

## Step 5: Transpose, Identity, Shapes, and Projection (Worked + Completion)

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

**Projection** -- the operation all of L20 (PCA) rests on -- is just a dot
product onto a **unit** direction. Project each user's 2-D taste row onto the
"action-versus-romance" axis and a single number says where they fall:

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
> handed to you.

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

### Exercise 1 -- Nearest neighbor by cosine

For each user, find their most similar *other* user by cosine similarity, and
print the pair with the score. One sentence: do the action fans and romance
fans pair up as expected?

> **Hint:** loop over users; for user `i`, compute `cosine(R[i], R[j])` for
> every `j != i` and keep the largest. Reuse the `cosine` function from
> Step 2.

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

### Exercise 2 -- Score the catalogue for a new taste

Invent a taste vector for a viewer who leans toward romance but is not
exclusive -- say `w = [0.3, 0.7]` -- score all five movies with `F @ w`, and
print the single highest-scoring movie with `np.argmax`.

> **Hint:** `F @ w` gives the five scores; `movies[int(np.argmax(scores))]`
> names the top one.

```python
# TODO: your code here
```

<details><summary>Expected Output (one possible answer -- match the values, not the print format)</summary>

~~~text
scores: [0.3  0.34 0.7  0.66 0.5 ]
top movie: Romance1
~~~
*(Same fixed feature matrix `F`, a romance-leaning taste vector -- Romance1
tops the list at 0.7, with Romance2 just behind. The features describe the
movies; the taste vector decides what wins.)*
</details>

### Exercise 3 -- Written: the shape of a projection

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
| Cosine similarity | `a @ b / (norm(a)*norm(b))` | magnitude-free "how alike" (0.96 vs 0.52) |
| Matrix x vector | `F @ w` | score every movie for one user in one multiply |
| Matrix x matrix | `U @ V` | score all users x all movies (the factorization idea) |
| Transpose / identity | `A.T`, `np.eye(n)` | line up shapes; the do-nothing matrix |
| Projection | `points @ d` | collapse to one axis -- the PCA preview |

Next lesson (L19): the dot product grows up -- mean-center two columns and
its cosine becomes the **correlation coefficient**, the measure of how two
variables move together.
