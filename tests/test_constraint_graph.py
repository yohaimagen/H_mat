"""C.12: independent small-tree oracle; never used by production code."""

from itertools import combinations, product

import numpy as np
import pytest

from gfcompress.build_tree import build_tree
from gfcompress.constraint_graph import CanonicalConstraint, build_constraint_graph
from gfcompress.fixed_pattern import Side
from gfcompress.geometry import FaultMesh
from gfcompress.interactions import TreeLists, build_lists
from gfcompress.tree import TreeNode


def _tree(dim: int, irregular: bool, dof: int) -> tuple[FaultMesh, TreeNode]:
    points = np.array(list(product(range(8 if dim == 1 else 4), repeat=dim)))
    if irregular:
        points = points[np.random.default_rng(19).random(len(points)) > 0.3]
    mesh = FaultMesh(points.astype(float), np.full(len(points), 0.1), dof_row=dof)
    return mesh, build_tree(mesh, m=1)


def _expected(
    root: TreeNode, lists: TreeLists, level: int, side: Side
) -> dict[tuple[TreeNode, TreeNode], CanonicalConstraint]:
    result = {}
    for alpha in root.nodes_at_level(level):
        for beta in lists.interaction[alpha]:
            if side == "col":
                random = beta
                zeros = set(lists.nei[alpha]) | set(lists.interaction[alpha])
            else:
                random = alpha
                zeros = set(lists.nei[beta]) | set(lists.interaction[beta])
            result[alpha, beta] = CanonicalConstraint(random, frozenset(zeros - {random}))
    return result


@pytest.mark.parametrize("dim,dof", [(1, 2), (2, 2), (3, 3)])
@pytest.mark.parametrize("irregular", [False, True])
@pytest.mark.parametrize("side", ["col", "row"])
def test_graph_matches_independent_oracle(dim: int, dof: int, irregular: bool, side: Side) -> None:
    mesh, root = _tree(dim, irregular, dof)
    assert mesh.n_rows != mesh.n_cols
    lists = build_lists(root)
    nonempty = False
    for nodes in root.iter_levels():
        level = nodes[0].level
        graph = build_constraint_graph(root, lists, level, side)
        expected = _expected(root, lists, level, side)
        assert set(graph.owners) == set(expected)
        assert set(graph.constraints) == set(expected.values())
        assert len(graph.constraints) == len(set(graph.constraints))
        for pair, constraint in expected.items():
            assert graph.constraints[graph.owners[pair]] == constraint
            assert constraint.random_box not in constraint.zero_boxes
        nonempty |= bool(graph.constraints)
        # Deliberately brute-force, test-only Definition 4.1 oracle.
        edges = {
            (i, j)
            for i, j in combinations(range(len(graph.constraints)), 2)
            if graph.constraints[i].random_box in graph.constraints[j].zero_boxes
            or graph.constraints[j].random_box in graph.constraints[i].zero_boxes
        }
        actual = {(i, j) for i, neighbors in enumerate(graph.adjacency) for j in neighbors if i < j}
        assert actual == edges
        assert graph.edge_count == len(edges)
        assert graph.input_count == len(expected)
        assert graph.input_incidences == sum(1 + len(c.zero_boxes) for c in expected.values())
        assert graph.incidences == sum(1 + len(c.zero_boxes) for c in graph.constraints)
        assert graph.conflict_visits == sum(
            int(a.random_box in b.zero_boxes) + int(b.random_box in a.zero_boxes)
            for a, b in combinations(graph.constraints, 2)
        )
        for i, neighbors in enumerate(graph.adjacency):
            assert i not in neighbors
            assert all(i in graph.adjacency[j] for j in neighbors)
        assert graph == build_constraint_graph(root, lists, level, side)
    assert nonempty


@pytest.mark.parametrize("dim,dof", [(1, 2), (2, 2), (3, 3)])
@pytest.mark.parametrize("irregular", [False, True])
def test_explicit_transpose_mapping(dim: int, dof: int, irregular: bool) -> None:
    _, root = _tree(dim, irregular, dof)
    lists = build_lists(root)
    for nodes in root.iter_levels():
        col = build_constraint_graph(root, lists, nodes[0].level, "col")
        row = build_constraint_graph(root, lists, nodes[0].level, "row")
        assert set(row.constraints) == set(col.constraints)
        col_index = {c: i for i, c in enumerate(col.constraints)}
        mapping = {i: col_index[c] for i, c in enumerate(row.constraints)}
        for (alpha, beta), vertex in row.owners.items():
            assert mapping[vertex] == col.owners[beta, alpha]
            assert row.constraints[vertex].random_box is alpha
        for i, neighbors in enumerate(row.adjacency):
            assert {mapping[j] for j in neighbors} == col.adjacency[mapping[i]]


@pytest.mark.parametrize("side", ["col", "row"])
def test_dedup_and_repeated_owners_ignore_zero_order(side: Side) -> None:
    _, root = _tree(1, False, 2)
    lists = build_lists(root)
    level = root.leaves()[0].level
    graph = build_constraint_graph(root, lists, level, side)
    # The paper's 8-cell example has 18 pairs but fewer distinct constraints.
    assert len(graph.owners) == 18
    assert len(graph.constraints) < len(graph.owners)
    assert set(graph.owners.values()) == set(range(len(graph.constraints)))
    repeated = TreeLists(
        nei={box: list(reversed(values)) * 2 for box, values in lists.nei.items()},
        interaction={box: values * 2 for box, values in lists.interaction.items()},
    )
    again = build_constraint_graph(root, repeated, level, side)
    assert again.constraints == graph.constraints
    assert again.owners == graph.owners
    assert again.adjacency == graph.adjacency
    assert again.input_count == 2 * graph.input_count
    assert again.input_incidences == 2 * graph.input_incidences
    assert again.incidences == graph.incidences
    assert again.conflict_visits == graph.conflict_visits


def test_shared_random_or_shared_zeros_alone_do_not_conflict() -> None:
    _, root = _tree(1, False, 2)
    a, b, c, d, e, f = root.leaves()[:6]
    # Two distinct constraints randomize c but have different zero sets.
    # A third randomizes d with the same zero set as the second; no conflict.
    lists = TreeLists(
        nei={a: [a], b: [a, b], e: [a, b], f: [c]},
        interaction={a: [c], b: [d], e: [c], f: [d]},
    )
    # Include all level boxes without adding any additional required pairs.
    for node in root.leaves():
        lists.interaction.setdefault(node, [])
    graph = build_constraint_graph(root, lists, a.level)
    ca, cb = graph.owners[a, c], graph.owners[e, c]
    assert ca != cb
    assert cb not in graph.adjacency[ca]
    cd = graph.owners[b, d]
    assert cd not in graph.adjacency[ca]  # shared zero a, different random boxes
    assert cd not in graph.adjacency[cb]  # identical zero sets, different random
    cf = graph.owners[f, d]
    assert graph.adjacency[cf] == {ca, cb}  # one-way conflicts still make edges
    assert graph.conflict_visits == graph.edge_count == 2


def test_empty_level_and_invalid_side() -> None:
    _, root = _tree(1, False, 2)
    lists = build_lists(root)
    graph = build_constraint_graph(root, lists, 100)
    assert graph.constraints == graph.adjacency == ()
    assert graph.owners == {}
    assert graph.input_count == graph.input_incidences == graph.incidences == 0
    assert graph.conflict_visits == graph.edge_count == 0
    with pytest.raises(ValueError, match="side"):
        build_constraint_graph(root, lists, 2, "invalid")  # type: ignore[arg-type]
