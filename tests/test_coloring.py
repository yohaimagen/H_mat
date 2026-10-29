"""C.13 coloring, priority-update oracle, and actual fixed-cost fallback."""

from dataclasses import replace
from itertools import combinations, product

import numpy as np
import pytest

from gfcompress.build_tree import build_tree
from gfcompress.coloring import dsatur, fixed_pattern_coloring, select_coloring, validate_coloring
from gfcompress.constraint_graph import build_constraint_graph
from gfcompress.fixed_pattern import Side, build_admissible_schedule
from gfcompress.geometry import FaultMesh
from gfcompress.interactions import build_lists


def _graph(n: int, edges: list[tuple[int, int]]) -> tuple[frozenset[int], ...]:
    adjacency: list[set[int]] = [set() for _ in range(n)]
    for i, j in edges:
        adjacency[i].add(j)
        adjacency[j].add(i)
    return tuple(frozenset(neighbors) for neighbors in adjacency)


def _oracle(adjacency: tuple[frozenset[int], ...]) -> tuple[int, ...]:
    # Test-only slow recomputation catches stale heap entries and bad updates.
    colors = [-1] * len(adjacency)
    while -1 in colors:
        available = [v for v, color in enumerate(colors) if color < 0]
        forbidden = {v: {colors[w] for w in adjacency[v] if colors[w] >= 0} for v in available}
        v = max(available, key=lambda v: (len(forbidden[v]), len(adjacency[v]), -v))
        color = 0
        while color in forbidden[v]:
            color += 1
        colors[v] = color
    return tuple(colors)


@pytest.mark.parametrize(
    "adjacency,count",
    [
        ((), 0),
        (_graph(5, []), 1),
        (_graph(5, list(combinations(range(5), 2))), 5),
        (_graph(7, [(i, i + 1) for i in range(6)]), 2),
        (_graph(8, [(0, 1), (1, 2), (2, 0), (3, 4), (5, 6)]), 3),
    ],
)
def test_basic_graphs(adjacency: tuple[frozenset[int], ...], count: int) -> None:
    colors = dsatur(adjacency)
    validate_coloring(adjacency, colors)
    assert len(set(colors)) == count
    assert colors == _oracle(adjacency) == dsatur(adjacency)


def test_priority_updates_and_deterministic_ties() -> None:
    rng = np.random.default_rng(32)
    for n in range(2, 26):
        for probability in (0.1, 0.4, 0.8):
            adjacency = _graph(
                n, [edge for edge in combinations(range(n), 2) if rng.random() < probability]
            )
            assert dsatur(adjacency) == _oracle(adjacency)
    assert dsatur(_graph(4, list(combinations(range(4), 2)))) == (0, 1, 2, 3)


@pytest.mark.parametrize(
    "adjacency",
    [
        (frozenset({0}),),
        (frozenset({1}), frozenset()),
        (frozenset({-1}),),
        (frozenset({2}),),
        (frozenset({0.5}),),
    ],
)
def test_invalid_graph(adjacency: tuple[frozenset[int], ...]) -> None:
    with pytest.raises(ValueError):
        dsatur(adjacency)
    with pytest.raises(ValueError):
        validate_coloring(adjacency, (0,) * len(adjacency))


@pytest.mark.parametrize("colors", [(), (0,), (0, 0), (-1, 0), (True, 0), (0.5, 1)])
def test_invalid_coloring(colors: tuple[int, ...]) -> None:
    with pytest.raises(ValueError):
        validate_coloring(_graph(2, [(0, 1)]), colors)


def _tree(dim: int, irregular: bool = False):
    points = np.array(list(product(range(8 if dim == 1 else 4), repeat=dim)), dtype=float)
    if irregular:
        points = points[np.random.default_rng(19).random(len(points)) > 0.3]
    mesh = FaultMesh(points, np.full(len(points), 0.1), dof_row=max(2, dim))
    root = build_tree(mesh, m=1)
    return mesh, root, build_lists(root)


@pytest.mark.parametrize("dim", [1, 2, 3])
@pytest.mark.parametrize("irregular", [False, True])
@pytest.mark.parametrize("side", ["col", "row"])
def test_generated_graphs_and_actual_fixed_schedule(dim: int, irregular: bool, side: Side) -> None:
    mesh, root, lists = _tree(dim, irregular)
    for nodes in root.iter_levels():
        level = nodes[0].level
        graph = build_constraint_graph(root, lists, level, side)
        selection = select_coloring(graph, root, level, 5)
        fixed, probes = fixed_pattern_coloring(graph, root, level)
        schedule = build_admissible_schedule(root, lists, level, mesh, k=3, p=2, side=side)
        assert probes == selection.fixed_probes == len(schedule.probes)
        assert selection.fixed_columns == sum(probe.k + probe.p for probe in schedule.probes)
        for pair, vertex in graph.owners.items():
            assert schedule.probes[fixed[vertex]] is schedule.owners[pair]
        validate_coloring(graph.adjacency, selection.colors)
        assert selection.vertices == len(graph.constraints)
        assert selection.edges == graph.edge_count
        assert selection.dsatur_columns == 5 * selection.dsatur_colors
        assert selection.fixed_colors == len(set(fixed))
        assert selection.fixed_seconds >= 0 and selection.dsatur_seconds >= 0
        assert selection.strategy == (
            "fixed" if selection.fixed_columns <= selection.dsatur_columns else "dsatur"
        )
        again = select_coloring(graph, root, level, 5)
        assert (
            replace(
                again,
                dsatur_seconds=selection.dsatur_seconds,
                fixed_seconds=selection.fixed_seconds,
            )
            == selection
        )


def test_fallback_cheaper_and_tie(monkeypatch: pytest.MonkeyPatch) -> None:
    _, root, lists = _tree(1)
    level = root.leaves()[0].level
    graph = build_constraint_graph(root, lists, level)
    fixed, probes = fixed_pattern_coloring(graph, root, level)
    # A valid but deliberately nonoptimal candidate exercises the safety net.
    monkeypatch.setattr(
        "gfcompress.coloring.dsatur", lambda adjacency: tuple(range(len(adjacency)))
    )
    result = select_coloring(graph, root, level, 7)
    assert len(graph.constraints) > probes
    assert result.strategy == "fixed" and result.reason == "fixed_cheaper"
    assert result.colors == fixed
    monkeypatch.setattr("gfcompress.coloring.dsatur", lambda adjacency: fixed)
    result = select_coloring(graph, root, level, 7)
    assert result.strategy == "fixed" and result.reason == "fixed_tie"


def test_unused_fixed_groups_are_not_free_and_empty_tree_level_ties() -> None:
    _, root, lists = _tree(1)
    graph = build_constraint_graph(root, lists, 0)
    result = select_coloring(graph, root, 0, 6)
    assert result.fixed_colors == 0 and result.fixed_probes == 1
    assert result.dsatur_columns == 0 and result.fixed_columns == 6
    assert result.strategy == "dsatur" and result.reason == "dsatur_cheaper"
    empty = select_coloring(graph, root, 100, 6)
    assert empty.colors == () and empty.reason == "fixed_tie"


def test_invalid_schedule_inputs() -> None:
    _, root, lists = _tree(1)
    graph = build_constraint_graph(root, lists, 3)
    for width in (0, -1, True, 1.5):
        with pytest.raises(ValueError, match="width"):
            select_coloring(graph, root, 3, width)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="level"):
        fixed_pattern_coloring(graph, root, 0)
    with pytest.raises(ValueError, match="counts"):
        fixed_pattern_coloring(replace(graph, adjacency=()), root, 3)
    constraint = graph.constraints[0]
    invalid = replace(constraint, zero_boxes=constraint.zero_boxes | {constraint.random_box})
    with pytest.raises(ValueError, match="cannot satisfy"):
        fixed_pattern_coloring(
            replace(graph, constraints=(invalid, *graph.constraints[1:])), root, 3
        )
