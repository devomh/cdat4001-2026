---
title: "Lab: Social Networks as Graphs with NetworkX"
unit: "VII"
lesson: "25"
type: lab
tags: [graphs, networks, networkx, adjacency-matrix, centrality, ego-network, karate-club]
difficulty: intermediate
duration: "70 mins"
---

**Goal:** turn a set of relationships into a **graph** you can measure. You will build and inspect a
network in NetworkX, write it as an **adjacency matrix** and run the L18 matrix operations on it (row
sums give degree, `A @ A` counts shared friends), find the important nodes with three kinds of
**centrality**, zoom into a node's **ego network**, and finish by suggesting new friends and testing
what happens when a central member leaves. The running graph is Zachary's Karate Club --
34 members, 78 friendships -- which ships inside NetworkX. Pairs with the concept note
[Social Networks as Graphs](l25_concept_social_networks_graphs.qmd).

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/devomh/cdat4001-2026/blob/main/u07_social_networks/l25_lab_social_networks_graphs.ipynb)

> This page is the read-only view. To run the lab, open the notebook
> (`l25_lab_social_networks_graphs.ipynb`) -- in Colab via the badge above, or
> locally. No data download is needed: the graph is built into NetworkX. Every
> text output below comes from actually running the cells.

## Prerequisites & Setup

We use **Zachary's Karate Club** (Zachary, 1977): the friendships among 34 members of a university
karate club that later split into two factions. It is the "hello world" of network science and is
bundled with **NetworkX**, so there is nothing to download.

```python
# Setup cell 1 of 2: install only (Colab resets on open; run this first)
%pip install -q networkx pandas numpy matplotlib
```

```python
# Setup cell 2 of 2: imports + load the built-in graph. Safe to re-run.
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

np.set_printoptions(precision=3, suppress=True)

G = nx.karate_club_graph()
print("Loaded Zachary's Karate Club:",
      G.number_of_nodes(), "nodes,", G.number_of_edges(), "edges.")
```

<details><summary>Expected Output</summary>

~~~text
Loaded Zachary's Karate Club: 34 nodes, 78 edges.
~~~
</details>

## Step 1 -- Build and inspect a graph

A graph is nodes plus edges. NetworkX lets you ask it anything: is it directed? how dense? who are a
node's neighbours? Node `0` is the instructor ("Mr. Hi") and node `33` is the club president ("the
Officer") -- the two who later split the club. The `club` attribute records which faction each member
ended up in; we use it in Step 2 to reorder the matrix (and once more in Exercise 2). Finding the
factions *without* it is L26's job (community detection).

```python
print("Directed?", G.is_directed())
print("Nodes:", G.number_of_nodes(), "| Edges:", G.number_of_edges())
print("Density:", round(nx.density(G), 4))
print("Neighbours of node 0:", sorted(G.neighbors(0)))
print("Node 0 friends with node 1?", G.has_edge(0, 1),
      "| with node 9?", G.has_edge(0, 9))
print("Node 0 club:", G.nodes[0]["club"], "| node 33 club:", G.nodes[33]["club"])
```

<details><summary>Expected Output</summary>

~~~text
Directed? False
Nodes: 34 | Edges: 78
Density: 0.139
Neighbours of node 0: [1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 12, 13, 17, 19, 21, 31]
Node 0 friends with node 1? True | with node 9? False
Node 0 club: Mr. Hi | node 33 club: Officer
~~~
</details>

The graph is **undirected** (friendship is mutual) and **sparse**. Density is the number of edges
divided by the number of possible pairs, `n(n-1)/2` for an undirected graph: `78 / (34 x 33 / 2) =
78 / 561 = 0.139`, so only about 14% of all possible friendships exist. One subtlety: each built-in
edge carries a **weight** -- the number of settings in which the two members interacted -- so this is
technically a weighted graph. We take the **binary** view (tie present or not) in Step 2, and come back
to the weights in Exercise 1.

```python
print("Edge (0, 1) stored data:", G.get_edge_data(0, 1))
```

<details><summary>Expected Output</summary>

~~~text
Edge (0, 1) stored data: {'weight': 4}
~~~
</details>

A picture builds intuition. We lay the graph out with a force-directed (`spring`) layout, size each node
by its degree, and mark the two leaders. Nodes are drawn largest first, so a smaller node that lands
next to a big one is drawn on top of it instead of hidden behind it.

```python
deg = dict(G.degree())
pos = nx.spring_layout(G, seed=42)   # fixed seed -> reproducible layout
big_first = sorted(G.nodes(), key=lambda n: -deg[n])

plt.figure(figsize=(8, 6))
nx.draw_networkx_edges(G, pos, alpha=0.3, edge_color="#888888")
nx.draw_networkx_nodes(G, pos, nodelist=big_first,
                       node_size=[30 + 60 * deg[n] for n in big_first],
                       node_color="#9bb7d4", edgecolors="#444444", linewidths=0.5)
nx.draw_networkx_labels(G, pos, labels={0: "Mr. Hi", 33: "Officer"}, font_size=9)
plt.title("Zachary's Karate Club")
plt.axis("off")
plt.show()
```

The two biggest nodes, Mr. Hi and the Officer, sit in different parts of the drawing, and most members
gather around one of them. A layout like this places tightly linked nodes close together, which is
why node 32 (the third-largest hub, 12 friends) lands right next to the Officer: they share many
friends.

You can build a graph from scratch too -- just hand NetworkX an edge list. Here is a tiny 4-person
graph (the same one the concept page works through by hand).

```python
H = nx.Graph()
H.add_edges_from([("Ana", "Beto"), ("Ana", "Cami"),
                  ("Beto", "Cami"), ("Cami", "Dani")])
print("H:", H.number_of_nodes(), "nodes,", H.number_of_edges(), "edges")
print("H degrees:", dict(H.degree()))
```

<details><summary>Expected Output</summary>

~~~text
H: 4 nodes, 4 edges
H degrees: {'Ana': 2, 'Beto': 2, 'Cami': 3, 'Dani': 1}
~~~
</details>

Cami has degree 3 (tied to Ana, Beto, and Dani); Dani has degree 1 (only tied to Cami). Reading degree
off a graph is just counting edges.

**Try it (completion).** Predict the effect, then run it: attach a new person to `H` and re-read the
degrees.

```python
# Completion: attach a new node "Eli" to "Dani" only, then reprint H's degrees.
# Predict first: whose degree changes, and to what? Uncomment and complete:
# H.add_edge("Dani", "___")
# print("H degrees:", dict(H.degree()))
```

<details><summary>Expected Output (when completed correctly)</summary>

~~~text
H degrees: {'Ana': 2, 'Beto': 2, 'Cami': 3, 'Dani': 2, 'Eli': 1}
~~~
*(Adding one edge raises the degree of both ends: Dani goes from 1 to 2, and Eli enters with 1.)*
</details>

## Step 2 -- The adjacency matrix (the L18 bridge)

Number the nodes `0..33` and build the `34 x 34` table `A` with `A[i, j] = 1` when `i` and `j` are
friends. We pass `weight=None` to get the **binary** adjacency (ignore the interaction weights). Because
the graph is undirected, `A` is **symmetric**.

```python
A = nx.to_numpy_array(G, nodelist=sorted(G.nodes()), dtype=int, weight=None)
print("A shape:", A.shape)
print("Symmetric (A == A.T)?", np.array_equal(A, A.T))
print("A[0, 1] =", A[0, 1], "| A[0, 9] =", A[0, 9], "| A[0, 0] =", A[0, 0])
print("Total 1s in A:", int(A.sum()), "= 2 x edges =", 2 * G.number_of_edges())
```

<details><summary>Expected Output</summary>

~~~text
A shape: (34, 34)
Symmetric (A == A.T)? True
A[0, 1] = 1 | A[0, 9] = 0 | A[0, 0] = 0
Total 1s in A: 156 = 2 x edges = 156
~~~
</details>

Now the **L18 payoff**. Adding up a row of `A` counts that node's friendships -- so the **row sum is the
degree**. Summing every row at once is `A @ ones`: the matrix-times-vector product from L18.

```python
deg_from_matrix = A.sum(axis=1)               # row sums
deg_from_nx = np.array([d for _, d in sorted(G.degree())])
print("Row sums == NetworkX degrees?", np.array_equal(deg_from_matrix, deg_from_nx))
print("Degree of node 0 (row sum):", int(deg_from_matrix[0]))
print("Degree of node 33 (row sum):", int(deg_from_matrix[33]))
```

<details><summary>Expected Output</summary>

~~~text
Row sums == NetworkX degrees? True
Degree of node 0 (row sum): 16
Degree of node 33 (row sum): 17
~~~
</details>

And the matrix-times-matrix product `A @ A` counts **common neighbours**: entry `(i, j)` is how many
friends `i` and `j` share, and the diagonal `(i, i)` is node `i`'s degree again.

```python
A2 = A @ A
print("(A @ A)[0, 0] =", int(A2[0, 0]), "(= degree of node 0)")
print("(A @ A)[0, 33] =", int(A2[0, 33]), "(shared friends of the two leaders)")
print("Check with NetworkX:", len(list(nx.common_neighbors(G, 0, 33))))
```

<details><summary>Expected Output</summary>

~~~text
(A @ A)[0, 0] = 16 (= degree of node 0)
(A @ A)[0, 33] = 4 (shared friends of the two leaders)
Check with NetworkX: 4
~~~
</details>

**Try it (completion).** Confirm the row-sum = degree claim on a node of your choice.

```python
# Completion: pick any node id and check its adjacency row sum equals its degree.
# node = 5
# print("row sum:", int(A[node].sum()), "| G.degree:", G.degree(node))
```

<details><summary>Expected Output (when completed correctly, node 5)</summary>

~~~text
row sum: 4 | G.degree: 4
~~~
</details>

Now a picture of `A`. In the natural order `0..33`, the two leaders stand out: row and column 0 (Mr.
Hi) and row and column 33 (the Officer) are the long dark bands along the edges of the picture, because
they have the most friends. Any other pattern is hard to see.

```python
plt.figure(figsize=(6, 5))
plt.imshow(A, cmap="Blues", interpolation="nearest")
plt.title("Binary adjacency matrix A (34 x 34)")
plt.xlabel("node j"); plt.ylabel("node i")
plt.colorbar(label="edge (1) / no edge (0)", shrink=0.8)
plt.show()
```

The node numbers are arbitrary, so we may list the nodes in any order. Suppose we **already know** which
faction each member joined after the split (the `club` attribute) and list Mr. Hi's faction first, then
the Officer's. Same graph, same friendships -- only the order of the rows and columns changes.

```python
club = nx.get_node_attributes(G, "club")
order = ([n for n in sorted(G.nodes()) if club[n] == "Mr. Hi"]
         + [n for n in sorted(G.nodes()) if club[n] == "Officer"])
A_sorted = nx.to_numpy_array(G, nodelist=order, dtype=int, weight=None)
k = sum(1 for n in G.nodes() if club[n] == "Mr. Hi")

print("Mr. Hi's faction:", k, "members | the Officer's:", len(order) - k)
print("Friendships inside Mr. Hi's faction:", int(A_sorted[:k, :k].sum()) // 2)
print("Friendships inside the Officer's    :", int(A_sorted[k:, k:].sum()) // 2)
print("Friendships across the two factions :", int(A_sorted[:k, k:].sum()))

plt.figure(figsize=(6, 5))
plt.imshow(A_sorted, cmap="Blues", interpolation="nearest")
plt.axhline(k - 0.5, color="red", linewidth=1)
plt.axvline(k - 0.5, color="red", linewidth=1)
plt.title("The same A, nodes listed by faction")
plt.xlabel("position j (Mr. Hi's faction first)")
plt.ylabel("position i")
plt.show()
```

<details><summary>Expected Output</summary>

~~~text
Mr. Hi's faction: 17 members | the Officer's: 17
Friendships inside Mr. Hi's faction: 35
Friendships inside the Officer's    : 32
Friendships across the two factions : 11
~~~
</details>

Two dense blocks appear on the diagonal, and the two off-diagonal blocks are almost empty: 67 of the 78
friendships stay inside a faction and only 11 cross. (Inside a block we divide by 2 because each
friendship appears twice in a symmetric matrix; the cross block `A_sorted[:k, k:]` holds each crossing
friendship once.) But notice what we cheated with: we used the **answer** -- who actually joined whom --
to choose the order. **L26** asks the real question: can an algorithm find an order like this from the
friendships alone, without the labels?

## Step 3 -- Degree and the degree distribution

Degree is the simplest measure of a node's importance: how many friends it has. Let us rank the members.

```python
degree_series = pd.Series(dict(G.degree())).sort_values(ascending=False)
print("Top 5 nodes by degree:")
print(degree_series.head(5).to_string())
print("\nMean degree:", round(degree_series.mean(), 3),
      "| min:", degree_series.min(), "| max:", degree_series.max())
```

<details><summary>Expected Output</summary>

~~~text
Top 5 nodes by degree:
33    17
0     16
32    12
2     10
1      9

Mean degree: 4.588 | min: 1 | max: 17
~~~
</details>

The two leaders (33 and 0) are far and away the **hubs**. The histogram below shows the rest: most
members have only 2 to 4 friends. The distribution is **skewed** -- a few hubs, many members with few
ties. Large social networks (millions of users) show the extreme version of this shape, a
*heavy-tailed* degree distribution; 34 nodes are too few to say more than "skewed".

```python
plt.figure(figsize=(7, 4))
plt.hist(list(dict(G.degree()).values()), bins=range(0, 19), edgecolor="white")
plt.title("Degree distribution (skewed: a few hubs)")
plt.xlabel("degree"); plt.ylabel("number of nodes")
plt.show()
```

The tallest bar is degree 2 (11 members); 23 of the 34 members have degree 2, 3, or 4, and only five
have 9 or more.

**Try it (completion).** Read one node's degree two ways and confirm they agree.

```python
# Completion: report node 2's degree from the Series and from the matrix row sum.
# The two must match. Uncomment and complete:
# print("from Series:", degree_series[2], "| from matrix:", int(A[2].sum()))
```

<details><summary>Expected Output (when completed correctly)</summary>

~~~text
from Series: 10 | from matrix: 10
~~~
</details>

## Step 4 -- Centrality: who is important?

Degree is one notion of "important". **Degree centrality** is the degree divided by `n - 1`, the
largest degree possible (node 33: `17 / 33 = 0.515`). **Betweenness** measures how often a node lies on
the shortest paths between other pairs (a *bridge*). **Closeness** measures how near a node is to
everyone else: `n - 1` divided by the sum of its distances to all other nodes (node 0: its 33 distances
add up to 58, so `33 / 58 = 0.569`). Let us compute all three and compare their top five.

```python
def top5(d):
    return sorted(d.items(), key=lambda kv: kv[1], reverse=True)[:5]

deg_c = nx.degree_centrality(G)
bet_c = nx.betweenness_centrality(G)
clo_c = nx.closeness_centrality(G)

print("Top 5 DEGREE     :", [(n, round(v, 3)) for n, v in top5(deg_c)])
print("Top 5 BETWEENNESS:", [(n, round(v, 3)) for n, v in top5(bet_c)])
print("Top 5 CLOSENESS  :", [(n, round(v, 3)) for n, v in top5(clo_c)])
```

<details><summary>Expected Output</summary>

~~~text
Top 5 DEGREE     : [(33, 0.515), (0, 0.485), (32, 0.364), (2, 0.303), (1, 0.273)]
Top 5 BETWEENNESS: [(0, 0.438), (33, 0.304), (32, 0.145), (2, 0.144), (31, 0.138)]
Top 5 CLOSENESS  : [(0, 0.569), (2, 0.559), (33, 0.55), (31, 0.541), (8, 0.516)]
~~~
</details>

The measures **disagree**, and that is the lesson. The Officer (node 33) has the **most ties** (top
degree), but Mr. Hi (node 0) is the best **bridge** (top betweenness) and the most central by distance
(top closeness). Rank only by degree and you would miss that node 0 is the main **broker**: more
shortest paths between other members run through Mr. Hi than through anyone else (Exercise 3 tests
what happens when he leaves). Let us colour the graph by betweenness to see the bridges.

```python
bet_vals = [bet_c[n] for n in big_first]
plt.figure(figsize=(8, 6))
nx.draw_networkx_edges(G, pos, alpha=0.3, edge_color="#888888")
nodes = nx.draw_networkx_nodes(G, pos, nodelist=big_first,
                               node_color=bet_vals, cmap="plasma",
                               node_size=[30 + 60 * deg[n] for n in big_first])
plt.colorbar(nodes, label="betweenness centrality", shrink=0.8)
plt.title("Karate Club coloured by betweenness (brighter = bigger bridge)")
plt.axis("off")
plt.show()
```

**Try it (completion).** Name the single top node by each measure.

```python
# Completion: print the highest-ranked node of each centrality. top5(d)[0] is the
# (node, value) pair in first place, so top5(d)[0][0] is the node. Complete the
# closeness entry, then uncomment:
# print("top degree:", top5(deg_c)[0][0],
#       "| top betweenness:", top5(bet_c)[0][0],
#       "| top closeness:", top5(___)[0][0])
```

<details><summary>Expected Output (when completed correctly)</summary>

~~~text
top degree: 33 | top betweenness: 0 | top closeness: 0
~~~
</details>

## Step 5 -- Ego networks and local structure

Zoom into one node and its neighbourhood: its **ego network** is the node, its neighbours, and the edges
among them. We also read local structure -- how tight-knit the neighbourhood is (clustering), whether
the graph is one piece (components), and how far apart nodes are (distances).

```python
ego0 = nx.ego_graph(G, 0, radius=1)
print("Mr. Hi's ego network (radius 1):",
      ego0.number_of_nodes(), "nodes,", ego0.number_of_edges(), "edges")
print("Clustering coefficient of node 0:", round(nx.clustering(G, 0), 3))
print("Average clustering (whole graph):", round(nx.average_clustering(G), 3))
print("Connected components:", nx.number_connected_components(G))
print("Shortest path 0 -> 33:", nx.shortest_path_length(G, 0, 33), "hops")
print("Diameter (longest shortest path):", nx.diameter(G))
print("Average shortest-path length:", round(nx.average_shortest_path_length(G), 3))
```

<details><summary>Expected Output</summary>

~~~text
Mr. Hi's ego network (radius 1): 17 nodes, 34 edges
Clustering coefficient of node 0: 0.15
Average clustering (whole graph): 0.571
Connected components: 1
Shortest path 0 -> 33: 2 hops
Diameter (longest shortest path): 5
Average shortest-path length: 2.408
~~~
</details>

The graph is **one connected piece**; any two members are at most **5** hops apart and **2.4** on
average -- a **small world**. Average clustering is high (0.571): your friends tend to be friends with
each other. Let us draw Mr. Hi's ego network on its own.

```python
plt.figure(figsize=(7, 6))
pos_ego = nx.spring_layout(ego0, seed=42)
nx.draw(ego0, pos_ego, with_labels=True, node_size=300,
        node_color="#9bb7d4", edge_color="#aaaaaa", font_size=8)
plt.title("Ego network of node 0 (Mr. Hi)")
plt.show()
```

Mr. Hi's 16 friends have only 18 friendships among themselves (34 edges in the ego network, minus his
own 16) out of 120 possible pairs -- that is his clustering of 0.15. A hub's friends tend to come from
different circles that do not know each other.

**Try it (completion).** Do the same for the other leader: build the Officer's (node 33) ego network
and read the Officer's clustering coefficient. Is the Officer also a hub whose friends mostly do not
know each other?

```python
# Completion: replace each ___ and uncomment.
# ego33 = nx.ego_graph(G, ___, radius=1)
# print("Officer's ego network:", ego33.number_of_nodes(), "nodes,",
#       ego33.number_of_edges(), "edges")
# print("Clustering coefficient of node 33:", round(nx.clustering(G, ___), 3))
```

<details><summary>Expected Output (when completed correctly)</summary>

~~~text
Officer's ego network: 18 nodes, 32 edges
Clustering coefficient of node 33: 0.11
~~~
*(Yes: 17 friends, but only 15 friendships among them out of 136 possible pairs. Both leaders sit
far below the club's average clustering of 0.571 -- each one is friends with many people who are not
friends with each other.)*
</details>

## Your Turn (Exercises)

### Exercise 1 -- Weighted vs. binary row sums

Step 2 built `A` with `weight=None`. Build the **weighted** matrix instead -- leave out `weight=None`,
so each entry holds the number of settings in which the two members interacted -- and print, for nodes
0 and 33, the weighted row sum next to the binary row sum (the degree). Why are they different, and
which one is the degree?

> **Hint:** `W = nx.to_numpy_array(G, nodelist=sorted(G.nodes()))`; then compare `W[n].sum()` with
> `A[n].sum()`.

```python
# TODO: your code here
```

<details><summary>One possible answer (code, output, reading)</summary>

~~~python
W = nx.to_numpy_array(G, nodelist=sorted(G.nodes()))
for n in [0, 33]:
    print(f"node {n}: weighted row sum {int(W[n].sum())} | "
          f"binary row sum {int(A[n].sum())}")
~~~
~~~text
node 0: weighted row sum 42 | binary row sum 16
node 33: weighted row sum 48 | binary row sum 17
~~~
*(The binary row sum counts **ties** -- that is the degree (16 and 17). The weighted row sum adds up
the **strength** of each tie: Mr. Hi's 16 friendships add up to 42 interaction settings, the Officer's
17 add up to 48. Both are useful numbers, but only the binary one is the degree, which is why Step 2
passed `weight=None` before checking "row sum = degree". On this graph the two happen to rank the top
five members in the same order (33, 0, 32, 2, 1); on another graph a member with few but intense ties
could rank high by strength and low by degree.)*
</details>

### Exercise 2 -- "People you may know"

A friend-suggestion feature looks for two people who are **not** friends yet but share many friends.
`A2 = A @ A` already counts the shared friends of every pair. Loop over all pairs `(i, j)` with
`i < j` and `A[i, j] == 0`, and print the three pairs with the most shared friends (break ties by the
smaller node number first). Then look up both members of each pair in `club`: would these suggestions
stay inside a faction?

> **Hint:** collect `(A2[i, j], i, j)` tuples in a list and sort it with
> `key=lambda p: (-p[0], p[1], p[2])`.

```python
# TODO: your code here
```

<details><summary>One possible answer (code, output, reading)</summary>

~~~python
pairs = []
for i in range(len(A)):
    for j in range(i + 1, len(A)):
        if A[i, j] == 0:
            pairs.append((int(A2[i, j]), i, j))
pairs.sort(key=lambda p: (-p[0], p[1], p[2]))
for shared, i, j in pairs[:3]:
    print(f"nodes {i} and {j}: {shared} shared friends "
          f"({club[i]} / {club[j]})")
~~~
~~~text
nodes 2 and 33: 6 shared friends (Mr. Hi / Officer)
nodes 0 and 33: 4 shared friends (Mr. Hi / Officer)
nodes 7 and 13: 4 shared friends (Mr. Hi / Mr. Hi)
~~~
*(This is Unit VI's "people like you" idea on a graph: shared friends play the role that shared
ratings played for recommenders. The top suggestion pairs node 2 (Mr. Hi's faction) with the Officer,
and the second pairs the two leaders themselves -- 4 shared friends, yet they are rivals and the club
split between them. Only 7 and 13 are in the same faction. Shared friends say that two people move in
the same circles, not that they would get along: a real system learns from which suggestions people
accept.)*
</details>

### Exercise 3 -- What if a member leaves?

Step 4 said that betweenness finds the **brokers** -- the members that many shortest paths run
through. Test what that means. Make a copy of `G`, remove node 0 (top betweenness), and print how many
connected pieces are left and their sizes. Repeat with node 33 (top degree). Which removal splits the
club, and which centrality from Step 4 would have warned you?

> **Hint:** `G_cut = G.copy(); G_cut.remove_node(0)`; the sizes are
> `[len(c) for c in nx.connected_components(G_cut)]`.

```python
# TODO: your code here
```

<details><summary>One possible answer (code, output, reading)</summary>

~~~python
for v in [0, 33]:
    G_cut = G.copy()
    G_cut.remove_node(v)
    sizes = sorted((len(c) for c in nx.connected_components(G_cut)),
                   reverse=True)
    print("without node", v, "->", len(sizes), "piece(s), sizes", sizes)
~~~
~~~text
without node 0 -> 3 piece(s), sizes [27, 5, 1]
without node 33 -> 1 piece(s), sizes [33]
~~~
*(Removing Mr. Hi splits the club into three pieces: member 11, whose only friend was Mr. Hi, and a
group of five (4, 5, 6, 10, 16) that reached the rest of the club only through him. Removing the
Officer -- who has **more** ties -- leaves the club in one piece; its paths just get longer (the
average shortest-path length rises from 2.408 to 2.597). Degree did not warn you; **betweenness** did,
because it measures who sits on the routes between others. (Closeness also ranks Mr. Hi first, but it
measures how near he is to everyone, not whether other members' routes pass through him.) It is a
warning, not a guarantee: a node with high betweenness makes many routes longer when it leaves, and it
cuts the graph only when it is the **only** route between two parts. In this club Mr. Hi is the only member whose removal disconnects
anyone.)*
</details>

## Summary

- A **graph** is nodes plus edges; choose **directed/undirected** and **weighted/unweighted** to match
  the relationship. NetworkX builds, inspects, measures, and draws it.
- A graph is also a **matrix**: the **adjacency matrix** `A`. The L18 operations now mean something --
  **row sum = degree** (`A @ ones`, with `weight=None`) and **`A @ A` = common-neighbour counts** (with
  degree on the diagonal). Shared-friend counts on non-friends give friend suggestions.
- Listing the nodes by faction turns `A` into two dense blocks -- but only because we used the labels.
- **Centrality** ranks importance, and the flavours **disagree**: degree (most ties), betweenness (on
  the most shortest paths: the broker), closeness (nearest to all). Report more than one. Removing the
  top-betweenness member split the club; removing the top-degree member did not.
- An **ego network** is one node's local world; **clustering**, **components**, and short **distances**
  describe local cohesion and the small-world structure. The degree distribution is skewed: a few hubs,
  most members with 2 to 4 friends.
- We measured *one graph's* local structure. **L26** asks the global question: do the nodes form
  **communities**, and can an algorithm recover the karate club's two real factions from structure alone?
