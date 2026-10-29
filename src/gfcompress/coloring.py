"""Deterministic Algorithm 2.3 coloring and measured fixed-pattern fallback."""

from dataclasses import dataclass
from heapq import heapify, heappop, heappush
from time import perf_counter
from typing import Literal

from gfcompress.constraint_graph import ConstraintGraph
from gfcompress.fixed_pattern import leaf_pattern_cell, pattern_cell
from gfcompress.tree import TreeNode


def _validate_graph(adjacency: tuple[frozenset[int], ...]) -> None:
    for i, neighbors in enumerate(adjacency):
        for j in neighbors:
            if type(j) is not int or not 0 <= j < len(adjacency):
                raise ValueError("neighbor must be an integer vertex in range")
            if i == j or i not in adjacency[j]:
                raise ValueError("graph must be undirected and have no self-edges")


def validate_coloring(adjacency: tuple[frozenset[int], ...], colors: tuple[int, ...]) -> None:
    """Reject malformed graphs, missing/invalid labels, and monochromatic edges."""
    _validate_graph(adjacency)
    if len(colors) != len(adjacency):
        raise ValueError("one color is required per vertex")
    if any(type(color) is not int or color < 0 for color in colors):
        raise ValueError("colors must be nonnegative integers")
    for i, neighbors in enumerate(adjacency):
        if any(colors[i] == colors[j] for j in neighbors):
            raise ValueError("adjacent vertices must have different colors")


def dsatur(adjacency: tuple[frozenset[int], ...]) -> tuple[int, ...]:
    """Greedy DSatur; ties use higher static degree, then smaller vertex index.

    Lazy heap updates discard stale saturation priorities. Updates happen only
    for a newly forbidden color, not each colored neighbor. Costs depend on the
    actual graph: O((V + E) log(V + E + 1)) time and O(V + E) memory. This is a
    deterministic heuristic, not an optimal-coloring guarantee.
    """
    _validate_graph(adjacency)
    colors = [-1] * len(adjacency)
    forbidden: list[set[int]] = [set() for _ in adjacency]
    queue = [(0, -len(neighbors), i) for i, neighbors in enumerate(adjacency)]
    heapify(queue)
    while queue:
        negative_saturation, _, vertex = heappop(queue)
        if colors[vertex] >= 0 or -negative_saturation != len(forbidden[vertex]):
            continue
        color = 0
        while color in forbidden[vertex]:
            color += 1
        colors[vertex] = color
        for neighbor in adjacency[vertex]:
            if colors[neighbor] < 0 and color not in forbidden[neighbor]:
                forbidden[neighbor].add(color)
                heappush(queue, (-len(forbidden[neighbor]), -len(adjacency[neighbor]), neighbor))
    result = tuple(colors)
    validate_coloring(adjacency, result)
    return result


def fixed_pattern_coloring(
    graph: ConstraintGraph, root: TreeNode, level: int, *, leaf: bool = False
) -> tuple[tuple[int, ...], int]:
    """Return labels and nonempty fixed probe count (period 3 for leaves, else 6).

    Labels match the sorted pattern cells used by the fixed schedule, including
    cells with no owning graph vertex. Thus len(set(labels)) need not equal the
    number of fixed probes. No numerical test matrices are constructed.
    """
    cell_for = leaf_pattern_cell if leaf else pattern_cell
    nodes = set(root.nodes_at_level(level))
    cells = sorted({cell_for(node, root) for node in nodes})
    labels = {cell: i for i, cell in enumerate(cells)}
    if len(graph.constraints) != len(graph.adjacency):
        raise ValueError("constraint and adjacency counts must agree")
    for constraint in graph.constraints:
        if constraint.random_box not in nodes or not constraint.zero_boxes <= nodes:
            raise ValueError("constraint boxes must belong to the requested tree level")
        cell = cell_for(constraint.random_box, root)
        if any(cell_for(box, root) == cell for box in constraint.zero_boxes):
            raise ValueError("fixed pattern cannot satisfy the constraint")
    colors = tuple(labels[cell_for(c.random_box, root)] for c in graph.constraints)
    validate_coloring(graph.adjacency, colors)
    return colors, len(cells)


@dataclass(frozen=True)
class ColoringSelection:
    """Selection accounting for one side/level at a common positive probe width.

    Column counts, not elapsed coloring time, decide the schedule. Fixed wins
    ties. Timings cover each candidate's construction and validation only;
    graph construction and eventual numerical probe generation are excluded.
    """

    colors: tuple[int, ...]
    strategy: Literal["fixed", "dsatur"]
    reason: Literal["fixed_cheaper", "fixed_tie", "dsatur_cheaper"]
    vertices: int
    edges: int
    dsatur_colors: int
    fixed_colors: int
    fixed_probes: int
    width: int
    dsatur_seconds: float
    fixed_seconds: float

    @property
    def dsatur_columns(self) -> int:
        return self.dsatur_colors * self.width

    @property
    def fixed_columns(self) -> int:
        return self.fixed_probes * self.width


def select_coloring(
    graph: ConstraintGraph, root: TreeNode, level: int, width: int, *, leaf: bool = False
) -> ColoringSelection:
    """Compare sampling columns at width k+p (or global w_max for leaf=True).

    DSatur emits one probe per used color; the existing fixed schedule emits
    one per occupied pattern cell, even cells without required pairs. Width is
    common to all probes (no rank-policy change). Neither choice wires probes
    into compression here, and neither promises strict savings.
    """
    if type(width) is not int or width <= 0:
        raise ValueError("probe width must be a positive integer")
    start = perf_counter()
    fixed, fixed_probes = fixed_pattern_coloring(graph, root, level, leaf=leaf)
    fixed_seconds = perf_counter() - start
    start = perf_counter()
    candidate = dsatur(graph.adjacency)
    validate_coloring(graph.adjacency, candidate)
    dsatur_seconds = perf_counter() - start
    candidate_count = len(set(candidate))
    prefer_fixed = fixed_probes * width <= candidate_count * width
    return ColoringSelection(
        colors=fixed if prefer_fixed else candidate,
        strategy="fixed" if prefer_fixed else "dsatur",
        reason=(
            ("fixed_tie" if fixed_probes == candidate_count else "fixed_cheaper")
            if prefer_fixed
            else "dsatur_cheaper"
        ),
        vertices=len(graph.constraints),
        edges=graph.edge_count,
        dsatur_colors=candidate_count,
        fixed_colors=len(set(fixed)),
        fixed_probes=fixed_probes,
        width=width,
        dsatur_seconds=dsatur_seconds,
        fixed_seconds=fixed_seconds,
    )
