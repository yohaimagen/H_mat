"""Interaction lists `L^int` and the strong-admissibility predicate (Task 1.5).

Per CLAUDE.md and the paper (Levitt & Martinsson 2024, S3, p.5 and Fig. 2/3,
p.6), for a box `alpha`:

- The **neighbor list** `L^nei(alpha)` (Task 1.4, `gfcompress.neighbors`) is
  the set of same-level boxes (including `alpha` itself) whose bounding boxes
  touch or overlap `alpha`'s.
- The **interaction list** `L^int(alpha)` is the set of children of the
  neighbors of `alpha`'s parent, excluding any box that is one of `alpha`'s
  own neighbors. On a regular grid this has at most `6^d - 3^d` entries.
- A pair of boxes `(alpha, beta)` at the same level is **admissible** (the
  block `A(I_alpha, I_beta)` is a candidate for low-rank compression) iff
  `dist(alpha, beta) >= eta * max(diam(alpha), diam(beta))`, where `dist` is
  the Euclidean gap distance between the two axis-aligned bounding boxes (zero
  if they touch or overlap) and `diam` is `TreeNode.diam`.

These two notions are *combinatorially* consistent on the fixed uniform dyadic
grid produced by `build_tree`/`gfcompress.build_tree`: for the right choice of
`eta`, every box in `L^int(alpha)` tests admissible against `alpha` and every
box in `L^nei(alpha)` tests inadmissible. See `DEFAULT_ETA` below for the
value used and why.

The `1/(r + gamma*L)^d` physics decay of the Green's function is used *only*
by `suggest_eta` to recommend a separation parameter from a target relative
error; it is never used as a standalone block-norm admissibility test (per
CLAUDE.md, that would break the level-nested structure that peeling depends
on). The geometric predicate `is_admissible` above is the only admissibility
gate used anywhere else in this package.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from gfcompress.neighbors import neighbor_lists
from gfcompress.tree import TreeNode

#: Default strong-admissibility separation parameter for `is_admissible`.
#:
#: On the fixed uniform dyadic grid built by `build_tree`, same-level boxes
#: are congruent axis-aligned cells of side `s`, so `diam = s * sqrt(d)`. A
#: box `beta` in `alpha`'s interaction list (a child of a neighbor of
#: `alpha`'s parent that is *not* itself a neighbor of `alpha`) is separated
#: from `alpha` by a gap of at least one cell width `s` along some axis, i.e.
#: `dist(alpha, beta) >= s`. Hence
#:
#:     dist(alpha, beta) / max(diam(alpha), diam(beta)) >= 1 / sqrt(d)
#:
#: which is `1/sqrt(2) ~= 0.707` in 2D and `1/sqrt(3) ~= 0.577` in 3D. Any
#: `beta` in `L^nei(alpha)` (including `alpha` itself) touches or overlaps
#: `alpha`, so `dist == 0` and is inadmissible for *any* `eta > 0`.
#:
#: `DEFAULT_ETA = 0.5` is `<= 1/sqrt(d)` for both `d in {2, 3}`, so it
#: classifies every interaction-list box as admissible and every neighbor box
#: as inadmissible -- the geometric predicate and the combinatorial
#: interaction-list/neighbor-list split agree exactly on this grid.
DEFAULT_ETA = 0.5


def box_dist(box_a: NDArray[np.float64], box_b: NDArray[np.float64]) -> float:
    """Euclidean gap distance between two axis-aligned boxes.

    On each axis, the gap is `max(lo_a - hi_b, lo_b - hi_a, 0)`: zero if the
    boxes' extents on that axis overlap or touch, otherwise the size of the
    separating interval. `box_dist` is the Euclidean norm of the per-axis
    gaps, so it is `0` whenever the boxes touch or overlap on every axis
    (consistent with `gfcompress.neighbors.boxes_adjacent`, for which
    adjacent boxes have `box_dist == 0`).

    Args:
        box_a: Axis-aligned box, shape `(d, 2)`, `box_a[i] = (lo_i, hi_i)`.
        box_b: Axis-aligned box, shape `(d, 2)`, `box_b[i] = (lo_i, hi_i)`.

    Returns:
        The Euclidean gap distance between `box_a` and `box_b`, `>= 0`.
    """
    if box_a.shape != box_b.shape:
        raise ValueError(f"box shapes must match, got {box_a.shape} and {box_b.shape}")

    lo_a, hi_a = box_a[:, 0], box_a[:, 1]
    lo_b, hi_b = box_b[:, 0], box_b[:, 1]

    gap = np.maximum(np.maximum(lo_a - hi_b, lo_b - hi_a), 0.0)
    return float(np.linalg.norm(gap))


def is_admissible(alpha: TreeNode, beta: TreeNode, eta: float = DEFAULT_ETA) -> bool:
    """Strong-admissibility predicate `dist(alpha, beta) >= eta * max(diam)`.

    This is the *only* admissibility gate used in this package (per
    CLAUDE.md): it is purely geometric and never replaced by a block-norm or
    physics-decay threshold.

    Args:
        alpha: First box.
        beta: Second box.
        eta: Separation parameter. Defaults to `DEFAULT_ETA`.

    Returns:
        `True` iff `box_dist(alpha.bounding_box, beta.bounding_box) >= eta *
        max(alpha.diam, beta.diam)`.
    """
    dist = box_dist(alpha.bounding_box, beta.bounding_box)
    return dist >= eta * max(alpha.diam, beta.diam)


def interaction_list(alpha: TreeNode, root: TreeNode) -> list[TreeNode]:
    """Compute `L^int(alpha)`: children of `alpha.parent`'s neighbors, minus
    `alpha`'s own neighbors.

    This is a convenience single-node wrapper around `interaction_lists`,
    which computes the per-level map for the whole tree in one pass. For
    repeated queries over many nodes, prefer calling `interaction_lists(root)`
    once and indexing into the result.

    Args:
        alpha: The box whose interaction list is computed.
        root: Root of the geometric cluster tree containing `alpha`.

    Returns:
        List of `TreeNode`s at `alpha`'s level: every child of a neighbor of
        `alpha.parent` (including `alpha.parent` itself, since a node is
        always its own neighbor) that is not in `alpha`'s own neighbor list
        (which includes `alpha` itself). At most `6^d - 3^d` entries on a
        regular grid. Empty if `alpha is root` (the root has no parent).
    """
    level_map = interaction_lists(root)[alpha.level]
    level_nodes = root.nodes_at_level(alpha.level)
    i = next(idx for idx, node in enumerate(level_nodes) if node is alpha)
    return level_map[i]


def interaction_lists(root: TreeNode) -> dict[int, dict[int, list[TreeNode]]]:
    """Build the per-level interaction-list map `L^int` for the tree rooted
    at `root`.

    For each level `l >= 1` present in the tree, and for each node `alpha` at
    level `l`, `L^int(alpha)` is the list of all level-`l` nodes that are
    children of a neighbor of `alpha.parent` (in `nodes_at_level(l-1)`'s
    neighbor map) but are *not* in `alpha`'s own neighbor list `L^nei(alpha)`
    (which includes `alpha` itself).

    Level 0 (the root, which has no parent) maps to `{0: []}`.

    Nodes are keyed exactly as in `gfcompress.neighbors.neighbor_lists`: each
    per-level dict is keyed by the node's index in `root.nodes_at_level(l)`.

    Args:
        root: Root of the geometric cluster tree (e.g. from `build_tree`).

    Returns:
        Mapping `level -> {index_in_level: [interaction-list nodes]}`.
    """
    nei = neighbor_lists(root)
    result: dict[int, dict[int, list[TreeNode]]] = {}

    for level_nodes in root.iter_levels():
        level = level_nodes[0].level
        if level == 0:
            result[level] = {0: []}
            continue

        own_nei = nei[level]
        parent_level_nodes = root.nodes_at_level(level - 1)
        parent_nei = nei[level - 1]

        level_map: dict[int, list[TreeNode]] = {}
        for i, alpha in enumerate(level_nodes):
            own_neighbor_ids = {id(beta) for beta in own_nei[i]}

            parent = alpha.parent
            assert parent is not None  # level >= 1
            parent_idx = next(idx for idx, node in enumerate(parent_level_nodes) if node is parent)

            # All same-level candidates: children of every neighbor of
            # alpha's parent (a node is always its own neighbor, so this
            # includes alpha.parent's own children too).
            candidates: list[TreeNode] = []
            seen_ids: set[int] = set()
            for parent_neighbor in parent_nei[parent_idx]:
                for child in parent_neighbor.children:
                    if id(child) not in seen_ids:
                        seen_ids.add(id(child))
                        candidates.append(child)

            level_map[i] = [beta for beta in candidates if id(beta) not in own_neighbor_ids]
        result[level] = level_map

    return result


def suggest_eta(gamma: float = 0.1, target_rel_error: float = 1e-2, d: int = 3) -> float:
    """Suggest a strong-admissibility parameter `eta` from the
    `1/(r + gamma*L)^d` Green's-function decay.

    This is a *sanity-check / suggestion* helper only -- it is never used as
    an admissibility gate (see module docstring and CLAUDE.md; the actual
    gate is the geometric predicate `is_admissible`, with its own
    grid-consistency-driven default `DEFAULT_ETA`).

    The elastostatic Green's function decays like `1/(r + gamma*L)^d`, where
    `r` is the box-to-box separation and `gamma*L` is a regularization
    accounting for each box's own extent `L`. Comparing the kernel at the
    admissibility boundary (`r = eta * L`, taking the box diameter `diam` as
    the length scale `L`) to its value at zero separation (`r = 0`, i.e.
    touching boxes) gives the relative size

        (gamma*L / (eta*L + gamma*L))^d = (gamma / (eta + gamma))^d.

    `suggest_eta` inverts this for `eta`: it returns the smallest `eta >= 0`
    such that this ratio is `<= target_rel_error`, i.e. the kernel has
    decayed by the requested factor by the time a block becomes admissible.
    Larger `gamma` (boxes with more "self-extent" relative to the kernel's
    regularization) and smaller `target_rel_error` (a stricter accuracy goal)
    both suggest a larger `eta` (more separation required before a block is
    treated as admissible).

    Args:
        gamma: Decay-law shape parameter (the `gamma` in `1/(r+gamma*L)^d`).
            Must be `>= 0`.
        target_rel_error: Desired relative decay factor at the admissibility
            boundary, in `(0, 1)`. Smaller values suggest a larger `eta`.
        d: Spatial dimension (2 or 3); controls the exponent in the decay
            law.

    Returns:
        A suggested `eta >= 0`.
    """
    if gamma < 0:
        raise ValueError(f"gamma must be >= 0, got {gamma}")
    if not (0.0 < target_rel_error < 1.0):
        raise ValueError(f"target_rel_error must be in (0, 1), got {target_rel_error}")
    if d not in (2, 3):
        raise ValueError(f"d must be 2 or 3, got {d}")

    # Solve (gamma / (eta + gamma))^d == target_rel_error for eta >= 0.
    ratio = target_rel_error ** (1.0 / d)
    eta = gamma * (1.0 / ratio - 1.0)
    return float(max(eta, 0.0))
