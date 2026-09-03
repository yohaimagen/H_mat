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

from dataclasses import dataclass

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


def interaction_lists(
    root: TreeNode, nei: dict[TreeNode, list[TreeNode]]
) -> dict[TreeNode, list[TreeNode]]:
    """Build the interaction-list map `L^int` for the tree rooted at `root`.

    For every node `alpha` at level `l >= 1`, `L^int(alpha)` is the list of
    all level-`l` nodes that are children of a neighbor of `alpha.parent`
    (per `nei`) but are *not* in `alpha`'s own neighbor list `L^nei(alpha)`
    (which includes `alpha` itself).

    The root (which has no parent) maps to `[]`.

    Args:
        root: Root of the geometric cluster tree (e.g. from `build_tree`).
        nei: Precomputed neighbor-list map (`gfcompress.neighbors.
            neighbor_lists(root)`); not recomputed here.

    Returns:
        A flat mapping `node -> [interaction-list nodes]`, with one entry per
        node in the tree.
    """
    result: dict[TreeNode, list[TreeNode]] = {}

    for level_nodes in root.iter_levels():
        level = level_nodes[0].level
        if level == 0:
            result[level_nodes[0]] = []
            continue

        for alpha in level_nodes:
            own_neighbors = set(nei[alpha])

            parent = alpha.parent
            assert parent is not None  # level >= 1

            # All same-level candidates: children of every neighbor of
            # alpha's parent (a node is always its own neighbor, so this
            # includes alpha.parent's own children too).
            candidates: list[TreeNode] = []
            seen: set[TreeNode] = set()
            for parent_neighbor in nei[parent]:
                for child in parent_neighbor.children:
                    if child not in seen:
                        seen.add(child)
                        candidates.append(child)

            result[alpha] = [beta for beta in candidates if beta not in own_neighbors]

    return result


@dataclass(frozen=True)
class TreeLists:
    """Node-keyed neighbor/interaction-list maps for a tree, computed once.

    Attributes:
        nei: Flat mapping `node -> L^nei(node)` (`gfcompress.neighbors.
            neighbor_lists`).
        interaction: Flat mapping `node -> L^int(node)` (`interaction_lists`).
    """

    nei: dict[TreeNode, list[TreeNode]]
    interaction: dict[TreeNode, list[TreeNode]]


def build_lists(root: TreeNode, tol: float = 1e-9) -> TreeLists:
    """Build the neighbor and interaction lists for the whole tree once.

    Downstream code (`gfcompress.sampling.build_sampling_constraint`,
    `gfcompress.column_basis.column_bases`, ...) should call this once per
    tree and pass the resulting `TreeLists` around, rather than recomputing
    `L^nei`/`L^int` per query.

    Args:
        root: Root of the geometric cluster tree (e.g. from `build_tree`).
        tol: Adjacency tolerance forwarded to `neighbor_lists`.

    Returns:
        A `TreeLists` with both maps.
    """
    nei = neighbor_lists(root, tol=tol)
    interaction = interaction_lists(root, nei)
    return TreeLists(nei=nei, interaction=interaction)


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
