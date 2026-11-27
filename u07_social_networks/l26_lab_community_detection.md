---
title: "Lab: Community Detection on the Karate Club"
unit: "VII"
lesson: "26"
type: lab
tags: [community-detection, modularity, louvain, graph-partitioning, networkx, nmi, karate-club]
difficulty: intermediate
duration: "70 mins"
---

**Goal:** answer Zachary's question -- can the friendship structure alone recover the karate club's two
real factions? You will score a grouping with **modularity** (and compute it by hand), let **Louvain**
and greedy modularity **find** communities, check the result against the real split with **NMI**, sort
the adjacency matrix by what the algorithm found, and contrast community detection with **graph
partitioning**. Same graph as L25 (the built-in karate club). Pairs with the concept note
[Community Detection in Networks](l26_concept_community_detection.qmd).

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/devomh/cdat4001-2026/blob/main/u07_social_networks/l26_lab_community_detection.ipynb)

> This page is the read-only view. To run the lab, open the notebook
> (`l26_lab_community_detection.ipynb`) -- in Colab via the badge above, or
> locally. No data download is needed: the graph is built into NetworkX. Every
> text output below comes from actually running the cells.

## Prerequisites & Setup

We reuse **Zachary's Karate Club** from L25. NetworkX ships the community-detection algorithms in
`networkx.algorithms.community`, and we use scikit-learn's `normalized_mutual_info_score` to compare a
detected grouping with the ground truth.

Two habits for the whole lab:

- **The friendship graph, not the weights.** As in L25, an edge means "these two are friends", and we
  pass `weight=None` to every call. NetworkX also stores, on each edge, the number of settings in which
  the two members interacted; several community functions would quietly use those counts if we let them.
- **Seeds.** The Louvain method and Kernighan-Lin start from a random choice, so we pass a fixed `seed`.
  Exercise 1 shows how much that choice matters.

```python
# Setup cell 1 of 2: install only (Colab resets on open; run this first)
%pip install -q networkx scikit-learn numpy matplotlib
```

```python
# Setup cell 2 of 2: imports + load the built-in graph. Safe to re-run.
import numpy as np
import networkx as nx
from networkx.algorithms import community as nxcom
from sklearn.metrics import normalized_mutual_info_score
from collections import Counter
import matplotlib.pyplot as plt

G = nx.karate_club_graph()
pos = nx.spring_layout(G, seed=42)   # same layout as the L25 lab
deg = dict(G.degree())
big_first = sorted(G.nodes(), key=lambda n: -deg[n])   # draw hubs first, small nodes on top
print("Karate Club:", G.number_of_nodes(), "nodes,", G.number_of_edges(), "edges.")
```

<details><summary>Expected Output</summary>

~~~text
Karate Club: 34 nodes, 78 edges.
~~~
</details>

## Step 1 -- The ground truth: two real factions

The club really did split in two. NetworkX stores which faction each member joined in the `club` node
attribute ("Mr. Hi" or "Officer"). We turn that into a 0/1 label per node -- our **ground truth** to test
the algorithms against.

```python
truth = {n: (0 if G.nodes[n]["club"] == "Mr. Hi" else 1) for n in G.nodes()}
print("Faction sizes (0 = Mr. Hi, 1 = Officer):", dict(Counter(truth.values())))

gt_sets = [{n for n in G.nodes() if truth[n] == 0},
           {n for n in G.nodes() if truth[n] == 1}]
print("Mr. Hi's group:", len(gt_sets[0]), "members | Officer's group:", len(gt_sets[1]), "members")
```

<details><summary>Expected Output</summary>

~~~text
Faction sizes (0 = Mr. Hi, 1 = Officer): {0: 17, 1: 17}
Mr. Hi's group: 17 members | Officer's group: 17 members
~~~
</details>

The split was even: 17 and 17. The algorithms below never see `truth`; we only use it to check them.

## Step 2 -- Scoring a grouping with modularity

**Modularity** $Q$ scores how community-like a grouping is: the friendships that fall *inside* the
groups, minus the share you would expect inside them by chance. Higher means stronger communities; near
0 means no better than chance. Let us score the real split, then a deliberately bad one.

```python
q_truth = nxcom.modularity(G, gt_sets, weight=None)
print("Modularity of the REAL two-faction split: Q =", round(q_truth, 4))

# A bad grouping that ignores structure: split by even vs odd node id.
bad = [{n for n in G.nodes() if n % 2 == 0}, {n for n in G.nodes() if n % 2 == 1}]
print("Modularity of an arbitrary even/odd split:  Q =", round(nxcom.modularity(G, bad, weight=None), 4))
```

<details><summary>Expected Output</summary>

~~~text
Modularity of the REAL two-faction split: Q = 0.3582
Modularity of an arbitrary even/odd split:  Q = -0.0001
~~~
</details>

The real split scores `0.358` -- solidly community-like, so the friendships really did cluster around the
two leaders. The even/odd split scores `-0.0001`: no better than chance (it even dips just below zero),
because it cuts straight across the real groups.

Where does `0.358` come from? For each group $c$, take $L_c$, the friendships inside it, and $d_c$,
the sum of its members' degrees. With $m$ friendships in total,

$$Q = \sum_{c} \left[ \frac{L_c}{m} - \left( \frac{d_c}{2m} \right)^2 \right].$$

The first term is the share of all friendships that fall inside group $c$. The second is the share
you would expect inside it by chance: a friendship has two ends, $2m$ ends in all, and the group holds
$d_c$ of them. You already have every number from L25's sorted matrix: 35 and 32 friendships inside
the factions, 78 in total.

```python
m = G.number_of_edges()
q_hand = 0.0
for name, group in zip(["Mr. Hi", "Officer"], gt_sets):
    L_c = G.subgraph(group).number_of_edges()   # friendships inside the group
    d_c = sum(deg[n] for n in group)            # degree sum of the group
    term = L_c / m - (d_c / (2 * m)) ** 2
    print(f"{name:7s}: L_c = {L_c}, d_c = {d_c} -> {L_c}/{m} - ({d_c}/{2 * m})^2 = {term:.4f}")
    q_hand += term
print(f"Q by hand = {q_hand:.4f} | nxcom.modularity = {q_truth:.4f}")
```

<details><summary>Expected Output</summary>

~~~text
Mr. Hi : L_c = 35, d_c = 81 -> 35/78 - (81/156)^2 = 0.1791
Officer: L_c = 32, d_c = 75 -> 32/78 - (75/156)^2 = 0.1791
Q by hand = 0.3582 | nxcom.modularity = 0.3582
~~~
</details>

Inside the two factions sit 67 of the 78 friendships (`0.859`); chance alone would put about half of
them there (`0.270 + 0.231 = 0.501`). The gap, `0.358`, is the modularity. Each faction contributes the
same `0.179` here -- a coincidence of this graph, not a rule.

**Try it (completion).** Score a third grouping: nodes 0-16 against nodes 17-33. Predict first: will
it land near the real split's 0.358, near 0, or in between? Fill the blanks, uncomment, run.

```python
# Completion: split by node id order (nodes 0-16 vs 17-33) and print its modularity.
# id_split = [set(range(___)), set(range(___, 34))]
# print("id-order split: Q =", round(nxcom.modularity(G, id_split, weight=None), 4))
```

<details><summary>Expected Output (when completed correctly)</summary>

~~~text
id-order split: Q = 0.2433
~~~

*(Between the two above: low and high ids loosely track the two factions, so this split captures some
of the structure but far less than the real one.)*
</details>

## Step 3 -- Letting an algorithm find the communities

Now the real task: hand the graph to an algorithm and let it find a high-scoring grouping -- *without*
the ground truth. **Louvain** searches for high modularity and chooses the number of communities itself.
It is a fast heuristic, and it starts from a random order, so we fix `seed=0`.

```python
louvain = nxcom.louvain_communities(G, seed=0, weight=None)
print("Louvain found", len(louvain), "communities; Q =", round(nxcom.modularity(G, louvain, weight=None), 4))
for c in sorted(louvain, key=len):
    hi = sum(1 for n in c if truth[n] == 0)
    print(f"  {len(c):2d} members: {hi:2d} from Mr. Hi's faction, {len(c) - hi:2d} from the Officer's")

greedy = list(nxcom.greedy_modularity_communities(G, weight=None))
print("Greedy modularity found", len(greedy), "communities; Q =",
      round(nxcom.modularity(G, greedy, weight=None), 4))
```

<details><summary>Expected Output</summary>

~~~text
Louvain found 4 communities; Q = 0.4198
   5 members:  5 from Mr. Hi's faction,  0 from the Officer's
   6 members:  0 from Mr. Hi's faction,  6 from the Officer's
  11 members: 11 from Mr. Hi's faction,  0 from the Officer's
  12 members:  1 from Mr. Hi's faction, 11 from the Officer's
Greedy modularity found 3 communities; Q = 0.3807
~~~
</details>

Both find **more than two** groups -- and Louvain's `Q = 0.420` is *higher* than the real split's
`0.358`. That is the key idea: maximizing modularity rewards **finer** structure, so the algorithm
splits each faction into tighter sub-groups. It is not wrong; it found a grouping that scores higher
than the social split. And look at the membership: every Louvain community comes from **one** faction,
except a single Mr. Hi member sitting with eleven of the Officer's. Let us see it.

```python
lou_label = {n: cid for cid, c in enumerate(louvain) for n in c}
plt.figure(figsize=(7, 6))
nx.draw_networkx_edges(G, pos, alpha=0.25, edge_color="#999999")
nx.draw_networkx_nodes(G, pos, nodelist=big_first, node_color=[lou_label[n] for n in big_first],
                       cmap=plt.get_cmap("tab10"), vmin=0, vmax=9,
                       node_size=[30 + 60 * deg[n] for n in big_first],
                       edgecolors="#333333", linewidths=0.5)
nx.draw_networkx_labels(G, pos, labels={0: "Mr. Hi", 33: "Officer", 8: "8"}, font_size=9)
plt.title(f"Louvain communities ({len(louvain)}), seed 0")
plt.axis("off")
plt.show()
```

Each colour is one community: Mr. Hi's side splits into two, the Officer's side into two. Node 8 is the
one Mr. Hi member drawn in the colour of the Officer's community.

**Try it (completion).** Print the number of greedy communities and their sizes. Fill the blanks,
uncomment, run.

```python
# Completion: print the number of greedy communities and their sizes.
# print("greedy communities:", len(___), "| sizes:", sorted(len(c) for c in ___))
```

<details><summary>Expected Output (when completed correctly)</summary>

~~~text
greedy communities: 3 | sizes: [8, 9, 17]
~~~
</details>

## Step 4 -- Did it get it right? Comparing to ground truth with NMI

We have a detected grouping and the real factions. **Normalized mutual information (NMI)** scores their
agreement from 0 (none) to 1 (identical). It builds on the mutual information from L13, which there
scored a feature against a label; here it compares two labelings of the same 34 members.

```python
nodes = sorted(G.nodes())
truth_labels = [truth[n] for n in nodes]
lou_labels = [lou_label[n] for n in nodes]
print("NMI(Louvain 4-community, real 2-faction split):",
      round(normalized_mutual_info_score(truth_labels, lou_labels), 4))
```

<details><summary>Expected Output</summary>

~~~text
NMI(Louvain 4-community, real 2-faction split): 0.5878
~~~
</details>

A moderate `0.59`: Louvain recovered the factions but then subdivided them, which costs agreement. So let
us ask for **exactly two** communities and compare again.

```python
two = list(nxcom.greedy_modularity_communities(G, weight=None, cutoff=2, best_n=2))
two = sorted(two, key=lambda c: 0 not in c)          # the group holding node 0 first
two_label = {n: cid for cid, c in enumerate(two) for n in c}
two_labels = [two_label[n] for n in nodes]
print("Forcing two communities -> sizes", sorted(len(c) for c in two))
print("NMI(two communities, real split):",
      round(normalized_mutual_info_score(truth_labels, two_labels), 4))

# Which members landed on the 'wrong' side of the real split?
maj = {cid: Counter(truth[n] for n in c).most_common(1)[0][0] for cid, c in enumerate(two)}
misplaced = sorted(n for n in nodes if maj[two_label[n]] != truth[n])
print("Members on the wrong side of the real split:", misplaced, "->", len(misplaced), "of 34")
```

<details><summary>Expected Output</summary>

~~~text
Forcing two communities -> sizes [17, 17]
NMI(two communities, real split): 0.6772
Members on the wrong side of the real split: [8, 9] -> 2 of 34
~~~
</details>

**This is Zachary's answer.** From the friendship graph alone -- no knowledge of the conflict -- the
two-community grouping recovers the real factions with only **2 of 34** members misplaced. The structure
nearly predicted the split.

L25 ended with a question. Sorting the adjacency matrix by the real factions showed two dense blocks --
but that used the answer. Can the algorithm's grouping produce the same picture without it? Sort the
matrix by the two **detected** groups:

```python
order = sorted(two[0]) + sorted(two[1])
k = len(two[0])
A_det = nx.to_numpy_array(G, nodelist=order, dtype=int, weight=None)
print("Friendships inside detected group 1:", int(A_det[:k, :k].sum()) // 2)
print("Friendships inside detected group 2:", int(A_det[k:, k:].sum()) // 2)
print("Friendships across the two groups   :", int(A_det[:k, k:].sum()))

plt.figure(figsize=(5.5, 5))
plt.imshow(A_det, cmap="Blues", interpolation="nearest")
plt.axhline(k - 0.5, color="red", lw=1)
plt.axvline(k - 0.5, color="red", lw=1)
plt.title("Adjacency matrix sorted by the detected groups")
plt.xlabel("position j (detected group 1 first)")
plt.ylabel("position i")
plt.show()
```

<details><summary>Expected Output</summary>

~~~text
Friendships inside detected group 1: 34
Friendships inside detected group 2: 34
Friendships across the two groups   : 10
~~~
</details>

Two dense blocks again, found from the friendships alone. Only **10** friendships cross between the
detected groups -- one *fewer* than the 11 that cross the real split (L25). By structure, the detected
grouping is slightly tidier than what actually happened; the two members it moved are the difference.
Here are the detected two next to the truth.

```python
fig, axes = plt.subplots(1, 2, figsize=(12, 5.4))
for ax, labels, title in [(axes[0], truth_labels, "Real factions (ground truth)"),
                          (axes[1], two_labels, "Detected: two communities")]:
    nx.draw_networkx_edges(G, pos, alpha=0.25, edge_color="#999999", ax=ax)
    nx.draw_networkx_nodes(G, pos, nodelist=big_first, node_color=[labels[n] for n in big_first],
                           cmap=plt.get_cmap("coolwarm"), vmin=0, vmax=1,
                           node_size=[30 + 60 * deg[n] for n in big_first],
                           edgecolors="#333333", linewidths=0.5, ax=ax)
    nx.draw_networkx_labels(G, pos, labels={0: "Mr. Hi", 33: "Officer", 8: "8", 9: "9"},
                            font_size=9, ax=ax)
    ax.set_title(title)
    ax.axis("off")
plt.show()
```

Only nodes 8 and 9 change colour between the two panels; every other member lands on the side they
actually joined.

**Try it (completion).** Measure how well the *Louvain* and *greedy* groupings agree with each other.
Predict first: closer to 0 or to 1? Fill the blank, uncomment, run.

```python
# Completion: build greedy labels and compute NMI between Louvain and greedy.
# gre_label = {n: cid for cid, c in enumerate(___) for n in c}
# print("NMI(Louvain, greedy):",
#       round(normalized_mutual_info_score(lou_labels, [gre_label[n] for n in nodes]), 4))
```

<details><summary>Expected Output (when completed correctly)</summary>

~~~text
NMI(Louvain, greedy): 0.6341
~~~

*(Two detection methods, run on the same graph, agree only moderately with each other: both split the
factions more finely than two, but not in the same places.)*
</details>

## Step 5 -- A different goal: graph partitioning

Community detection lets group sizes vary by density. **Graph partitioning** asks a different question:
split the nodes into a fixed number of **balanced** parts while cutting **as few edges as possible**. The
**Kernighan-Lin** heuristic does the two-way case.

```python
part0, part1 = nxcom.kernighan_lin_bisection(G, seed=42, weight=None)
cut = list(nx.edge_boundary(G, part0, part1))
print("Kernighan-Lin partition sizes:", sorted([len(part0), len(part1)]))
print("Edge-cut (edges crossing between the two parts):", len(cut))
print("Edge-cut of the REAL split:", len(list(nx.edge_boundary(G, gt_sets[0], gt_sets[1]))))
print("Same split as 'force two communities'?",
      {frozenset(part0), frozenset(part1)} == {frozenset(c) for c in two})
print()
print("Community detection sizes (Louvain):", sorted(len(c) for c in louvain), "<- vary by density")
print("Partition sizes (Kernighan-Lin):    ", sorted([len(part0), len(part1)]), "<- forced balanced")
```

<details><summary>Expected Output</summary>

~~~text
Kernighan-Lin partition sizes: [17, 17]
Edge-cut (edges crossing between the two parts): 10
Edge-cut of the REAL split: 11
Same split as 'force two communities'? True

Community detection sizes (Louvain): [5, 6, 11, 12] <- vary by density
Partition sizes (Kernighan-Lin):     [17, 17] <- forced balanced
~~~
</details>

Kernighan-Lin returns **balanced halves** (17 and 17) cutting only **10** edges -- and they are exactly
the two groups that forced detection found, so the real split (11 crossing edges) is *not* the one with
the fewest crossings. On this graph the two methods meet. In general they do not, because the *goal*
differs: partitioning wants balanced parts with the fewest crossing edges (think sharding a graph across
servers), while community detection wants the natural dense groups, whatever their sizes.

## Your Turn (Exercises)

### Exercise 1 -- Is Louvain stable? And the resolution knob

**(a)** Run Louvain with seeds 0 to 9 and 42. For each, print the number of groups, their sizes,
$Q$, and whether Mr. Hi (node 0) and the Officer (node 33) land in different groups. What changes with
the seed, and what never does?

**(b)** `louvain_communities` has a `resolution` parameter (default 1). Below 1 it favours fewer, larger
communities. Run it with `resolution=0.5` (seed 0). How many groups do you get, and is it the
Kernighan-Lin split from Step 5?

```python
# TODO: your code here
```

<details><summary>One possible answer (code, output, reading)</summary>

~~~python
for s in list(range(10)) + [42]:
    comms = nxcom.louvain_communities(G, seed=s, weight=None)
    label = {n: cid for cid, c in enumerate(comms) for n in c}
    print(f"seed {s:2d}: {len(comms)} groups, sizes {sorted(len(c) for c in comms)}, "
          f"Q = {nxcom.modularity(G, comms, weight=None):.4f}, apart: {label[0] != label[33]}")

coarse = nxcom.louvain_communities(G, seed=0, weight=None, resolution=0.5)
print("resolution 0.5:", len(coarse), "groups, sizes", sorted(len(c) for c in coarse))
print("Same as the Kernighan-Lin split?",
      {frozenset(c) for c in coarse} == {frozenset(part0), frozenset(part1)})
~~~
~~~text
seed  0: 4 groups, sizes [5, 6, 11, 12], Q = 0.4198, apart: True
seed  1: 4 groups, sizes [5, 6, 11, 12], Q = 0.4188, apart: True
seed  2: 4 groups, sizes [5, 6, 11, 12], Q = 0.4188, apart: True
seed  3: 4 groups, sizes [4, 5, 11, 14], Q = 0.4151, apart: True
seed  4: 4 groups, sizes [4, 5, 11, 14], Q = 0.4151, apart: True
seed  5: 4 groups, sizes [4, 5, 12, 13], Q = 0.4156, apart: True
seed  6: 4 groups, sizes [5, 6, 11, 12], Q = 0.4188, apart: True
seed  7: 4 groups, sizes [5, 6, 11, 12], Q = 0.4188, apart: True
seed  8: 4 groups, sizes [5, 6, 11, 12], Q = 0.4198, apart: True
seed  9: 4 groups, sizes [5, 6, 11, 12], Q = 0.4188, apart: True
seed 42: 3 groups, sizes [4, 13, 17], Q = 0.3854, apart: True
resolution 0.5: 2 groups, sizes [17, 17]
Same as the Kernighan-Lin split? True
~~~
*(The **seed changes the answer**: Louvain is a fast heuristic, not an exhaustive search, so different
random starts settle on different groupings -- here 3 or 4 groups and $Q$ between 0.385 and 0.420.
Seed 42 is an unlucky start: it stops at a three-group answer that scores lower than most. What never
changes is the big picture: every seed finds more than two groups, every $Q$ beats the real split's
0.358, and Mr. Hi and the Officer always end up apart. Trust conclusions that survive several seeds; for
a single answer, keep the run with the highest $Q$. Lowering the resolution to 0.5 asks for coarser
groups, and Louvain returns two: the same 17/17 split Kernighan-Lin and forced detection found.)*
</details>

### Exercise 2 -- Name the holdouts

The two-community detection misplaced 2 of the 34 members (`misplaced`). For each, print the faction
this member actually joined, their degree, their friends, and how many of those friends are in each
faction. Why might structure put them on the other side?

```python
# TODO: your code here
```

<details><summary>One possible answer (code, output, reading)</summary>

~~~python
for n in misplaced:
    sides = Counter(G.nodes[f]["club"] for f in G.neighbors(n))
    print(f"node {n}: joined {G.nodes[n]['club']}, degree {deg[n]}, friends {sorted(G.neighbors(n))}")
    print(f"  friends on Mr. Hi's side {sides['Mr. Hi']}, on the Officer's side {sides['Officer']}")
~~~
~~~text
node 8: joined Mr. Hi, degree 5, friends [0, 2, 30, 32, 33]
  friends on Mr. Hi's side 2, on the Officer's side 3
node 9: joined Officer, degree 2, friends [2, 33]
  friends on Mr. Hi's side 1, on the Officer's side 1
~~~
*(**Node 8** has more friends on the Officer's side (3 of 5, including the Officer), so the structure
places this member there -- but they joined Mr. Hi. Zachary's account offers a reason the graph cannot
show: this member (number 9 in Zachary's 1-34 numbering) was about three weeks from a black-belt test
when the club split, and joining the Officer's new club would have meant giving up their rank and
starting over in a new style. **Node 9** has only
two friends, one on each side: a tie that the friendships alone cannot break. One holdout had a reason
the data never recorded; the other had too little data to decide.)*
</details>

### Exercise 3 -- Written: partition or detect?

For each task, say whether you would use **community detection** or **graph partitioning**, and why:
**(a)** splitting a 50-million-node social graph across 8 servers so that as few friendships as possible cross
between servers; **(b)** discovering interest-based groups of users for a "groups you might like" feature.
Two to three sentences total.

<details><summary>A model answer</summary>

**(a) Graph partitioning:** the number of parts is fixed (8 servers), the parts must be balanced so no
server is overloaded, and the goal is the fewest crossing friendships -- exactly the balanced, minimum
edge-cut problem.

**(b) Community detection:** nobody knows in advance how many interest groups exist or how big they
are, and the goal is groups that are dense inside, so a method that maximizes modularity and picks the
number of groups itself fits.
</details>

## Summary

- A **community** is dense inside, sparse outside. **Modularity**
  $Q = \sum_c [L_c/m - (d_c/2m)^2]$ scores a grouping: the share of friendships inside the groups minus
  the share chance would put there. The real karate split scores 0.358 (67 of 78 friendships inside vs
  about half by chance); an even/odd split, about 0.
- **Community detection** (Louvain, greedy modularity) searches for high $Q$ and **picks the number
  of groups itself** -- here 4 (Louvain) or 3 (greedy), finer than the 2 real factions, because finer
  splits raise $Q$. Louvain is **randomized**: seeds change the details, not the big picture.
- **NMI** checks a grouping against ground truth (0 to 1). Forcing two communities recovers the real
  factions with only **2 of 34** members misplaced, and sorting the matrix by the detected groups gives
  L25's two blocks without the labels (10 crossing friendships, vs 11 for the real split).
- **Graph partitioning** (Kernighan-Lin) is a different goal: **balanced** parts with the minimum
  **edge-cut** (here 17/17, cut 10) -- for sharding and load-balancing, not for finding natural groups.
- **Unit VII complete:** L25 made the graph and measured local structure; L26 found the global structure
  and recovered a real social split from connections alone.
