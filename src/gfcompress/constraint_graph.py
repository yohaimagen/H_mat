"""Box-only Eq. 4.4 constraints and Definition 4.1 incompatibility graphs.

No scalar indices, numerical operator, coloring, or probes are needed here.
Vertex numbers follow first occurrence in tree/interaction-list order.
"""

from __future__ import annotations

from dataclasses import dataclass

from gfcompress.fixed_pattern import Side
from gfcompress.interactions import TreeLists
from gfcompress.sampling import build_sampling_constraint
from gfcompress.tree import TreeNode


@dataclass(frozen=True)
class CanonicalConstraint:
    """One active box and an unordered zero set (node identity is the key).

    ``random_box`` is the identity-probe source for neighbor-only leaf graphs.
    """

    random_box: TreeNode
    zero_boxes: frozenset[TreeNode]


@dataclass(frozen=True)
class ConstraintGraph:
    """Deduplicated constraints, symmetric adjacency, and original pair owners.

    ``owners`` retains every distinct original (alpha, beta), even on the row
    side. Repeated occurrences of a pair have the same owner. ``input_count``
    counts occurrences before deduplication; ``input_incidences`` counts their
    random-plus-zero incidences. ``incidences`` counts those of unique vertices.
    ``conflict_visits`` counts incidence cross-product hits, before duplicate
    undirected edges are merged; ``edge_count`` counts emitted edges once.
    """

    constraints: tuple[CanonicalConstraint, ...]
    adjacency: tuple[frozenset[int], ...]
    owners: dict[tuple[TreeNode, TreeNode], int]
    input_count: int
    input_incidences: int
    incidences: int
    conflict_visits: int

    @property
    def edge_count(self) -> int:
        return sum(map(len, self.adjacency)) // 2


def build_constraint_graph(
    root: TreeNode,
    lists: TreeLists,
    level: int,
    side: Side = "col",
    *,
    neighbors_only: bool = False,
) -> ConstraintGraph:
    """Build the admissible graph for one level of the shared geometric tree.

    Column sampling of (alpha, beta) randomizes beta and zeros the other boxes
    in nei(alpha) union int(alpha). Row sampling explicitly swaps those roles:
    randomize alpha and zero (nei(beta) union int(beta)) minus alpha. Ownership
    remains in original matrix orientation; rectangular dof expansion is left
    to the eventual probe builder, not this graph.

    With ``neighbors_only``, required pairs and zero sets use only nei, as
    in Sec. 4.1.3 for dense leaves. The active box then carries identity rows,
    not Gaussian rows; incompatibility and deduplication are unchanged.

    Conflicts are enumerated by joining random and zero incidence lists for
    each box, never by scanning all vertex pairs. With expected constant-time
    dict/set operations, construction costs the input-list traversal and
    canonicalization, plus O(I + C + V + E), where I is canonical incidences,
    C is conflict_visits, V vertices, and E emitted undirected edges. Storage
    is O(P + I + V + E), including P distinct owners. E may be quadratic in V;
    this is not a claim of a sparse graph or linear-in-boxes construction.
    """
    if side not in ("col", "row"):
        raise ValueError(f"side must be 'col' or 'row', got {side!r}")

    vertices: dict[CanonicalConstraint, int] = {}
    owners: dict[tuple[TreeNode, TreeNode], int] = {}
    input_count = input_incidences = 0
    pairs = lists.nei if neighbors_only else lists.interaction
    for alpha in root.nodes_at_level(level):
        for beta in pairs[alpha]:
            target, source = (alpha, beta) if side == "col" else (beta, alpha)
            if neighbors_only:
                zeros = frozenset(lists.nei[target]) - {source}
            else:
                constraint = build_sampling_constraint(target, source, lists)
                zeros = frozenset(constraint.zero_boxes)
            key = CanonicalConstraint(source, zeros)
            vertex = vertices.setdefault(key, len(vertices))
            owners[alpha, beta] = vertex
            input_count += 1
            input_incidences += 1 + len(key.zero_boxes)

    random_at: dict[TreeNode, list[int]] = {}
    zero_at: dict[TreeNode, list[int]] = {}
    for canonical, vertex in vertices.items():
        random_at.setdefault(canonical.random_box, []).append(vertex)
        for box in canonical.zero_boxes:
            zero_at.setdefault(box, []).append(vertex)

    adjacency: list[set[int]] = [set() for _ in vertices]
    conflict_visits = 0
    for box, random_vertices in random_at.items():
        for i in random_vertices:
            for j in zero_at.get(box, ()):
                # Eq. 4.4 excludes the random box from its own zero set.
                adjacency[i].add(j)
                adjacency[j].add(i)
                conflict_visits += 1

    return ConstraintGraph(
        constraints=tuple(vertices),
        adjacency=tuple(frozenset(neighbors) for neighbors in adjacency),
        owners=owners,
        input_count=input_count,
        input_incidences=input_incidences,
        incidences=len(vertices) + sum(map(len, zero_at.values())),
        conflict_visits=conflict_visits,
    )
