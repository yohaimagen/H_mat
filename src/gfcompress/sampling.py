"""Sampling-constraint descriptor for admissible blocks (Eq. 4.4, Task 4.1).

Per Levitt & Martinsson (2024), Eq. 4.4, a fixed-pattern test matrix used to
sample an admissible block `(alpha, beta)` (`alpha` = row box, `beta` = col
box) must satisfy:

- the columns belonging to `beta` are filled with i.i.d. Gaussian entries
  (`G_beta`, the "random" box), so that `A(I_alpha, I_beta) @ G_beta` recovers
  a sketch of the block; and
- the columns belonging to every box `gamma in L^nei(alpha) | L^int(alpha) \\
  {beta}` -- i.e. every other box that interacts with `alpha` (its neighbors
  and interaction-list partners) -- are forced to *zero*, so that those
  blocks' contributions do not contaminate the sample for `(alpha, beta)`.

This module provides only the *descriptor*: a per-block dataclass recording
which col-box is random and which col-boxes must be zeroed, expressed as
`dof_col`-based patch-major column-index sets (`TreeNode.col_indices`, per
`gfcompress.geometry.FaultMesh.patch_to_cols`), plus a builder function that
assembles this descriptor from the existing neighbor-list (`L^nei`,
`gfcompress.neighbors.neighbor_lists`) and interaction-list (`L^int`,
`gfcompress.interactions.interaction_lists`) maps for an admissible pair
`(alpha, beta)`.

Assembling the actual `Omega` test matrices from these descriptors (filling
random/zero column blocks, periodic tiling, etc.) is out of scope for this
task -- see Tasks 4.2/4.3.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from gfcompress.interactions import interaction_lists
from gfcompress.neighbors import neighbor_lists
from gfcompress.tree import TreeNode


@dataclass(frozen=True)
class SamplingConstraint:
    """Per-block sampling-constraint descriptor for an admissible pair
    `(alpha, beta)` (Eq. 4.4).

    Attributes:
        alpha: The row box.
        beta: The col box marked *random* (`G_beta`).
        random_box: The col-box marked random; always `beta`.
        random_cols: `beta`'s flattened column-index set (`beta.col_indices`),
            i.e. the columns that must be filled with i.i.d. Gaussian entries.
        zero_boxes: The col-boxes `gamma in L^nei(alpha) | L^int(alpha) \\
            {beta}` that must be zeroed, in a fixed (neighbors-then-
            interaction-list) order with no duplicates.
        zero_cols: The concatenation of `gamma.col_indices` for every `gamma`
            in `zero_boxes`, i.e. the columns that must be forced to zero.
            Disjoint from `random_cols` by construction (`beta` is excluded
            from `zero_boxes`).
    """

    alpha: TreeNode
    beta: TreeNode
    zero_boxes: list[TreeNode] = field(default_factory=list)

    @property
    def random_box(self) -> TreeNode:
        """The col-box marked random; always `beta`."""
        return self.beta

    @property
    def random_cols(self) -> NDArray[np.intp]:
        """`beta`'s flattened column-index set (the "random" columns)."""
        return self.beta.col_indices

    @property
    def zero_cols(self) -> NDArray[np.intp]:
        """Concatenated column-index sets of all `zero_boxes`.

        Returns an empty `intp` array if `zero_boxes` is empty.
        """
        if not self.zero_boxes:
            return np.array([], dtype=np.intp)
        return np.concatenate([gamma.col_indices for gamma in self.zero_boxes])


def build_sampling_constraint(
    alpha: TreeNode, beta: TreeNode, root: TreeNode
) -> SamplingConstraint:
    """Build the Eq. 4.4 sampling-constraint descriptor for the admissible
    pair `(alpha, beta)`.

    `beta` is marked as the *random* col-box (`G_beta`); every col-box `gamma
    in L^nei(alpha) | L^int(alpha)`, except `beta` itself, is collected as a
    *zero* box. `L^nei(alpha)` and `L^int(alpha)` are taken from the existing
    per-level maps (`gfcompress.neighbors.neighbor_lists`,
    `gfcompress.interactions.interaction_lists`) for the tree rooted at
    `root`, so this function does not re-derive admissibility or the
    neighbor/interaction-list combinatorics.

    Args:
        alpha: The row box of the admissible pair.
        beta: The col box of the admissible pair, marked random. May or may
            not itself appear in `L^nei(alpha) | L^int(alpha)`; if it does, it
            is excluded from the zero set (a box cannot be both random and
            zero).
        root: Root of the geometric cluster tree containing `alpha` and
            `beta` (both at the same level).

    Returns:
        A `SamplingConstraint` with `beta` as `random_box` and `zero_boxes =
        L^nei(alpha) | L^int(alpha) \\ {beta}`, deduplicated and ordered
        neighbors-first then interaction-list.

    Raises:
        ValueError: If `alpha` and `beta` are not at the same tree level.
    """
    if alpha.level != beta.level:
        raise ValueError(
            f"alpha and beta must be at the same level, got {alpha.level} and {beta.level}"
        )

    level = alpha.level
    level_nodes = root.nodes_at_level(level)
    alpha_idx = next(idx for idx, node in enumerate(level_nodes) if node is alpha)

    nei = neighbor_lists(root)[level][alpha_idx]
    interactions = interaction_lists(root)[level][alpha_idx]

    zero_boxes: list[TreeNode] = []
    seen_ids: set[int] = set()
    for gamma in (*nei, *interactions):
        if gamma is beta or id(gamma) in seen_ids:
            continue
        seen_ids.add(id(gamma))
        zero_boxes.append(gamma)

    return SamplingConstraint(alpha=alpha, beta=beta, zero_boxes=zero_boxes)
