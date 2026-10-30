"""Internal admissible schedules; no public compressor or leaf-path changes."""

from dataclasses import replace
from time import perf_counter

from gfcompress.coloring import ColoringSelection, select_coloring
from gfcompress.constraint_graph import build_constraint_graph
from gfcompress.fixed_pattern import (
    AdmissibleProbeSchedule,
    PeriodicTestMatrix,
    Side,
    build_admissible_schedule,
    validate_admissible_probe_owners,
)
from gfcompress.geometry import FaultMesh
from gfcompress.interactions import TreeLists
from gfcompress.tree import TreeNode


def build_colored_admissible_schedule(
    root: TreeNode,
    lists: TreeLists,
    level: int,
    mesh: FaultMesh,
    k: int,
    p: int = 10,
    seed: int | None = None,
    side: Side = "col",
) -> tuple[AdmissibleProbeSchedule, ColoringSelection]:
    """Select fixed or DSatur groups, preserving explicit original pair owners.

    Existing basis passes accept the returned schedule directly. Realization
    reuses the fixed descriptor's per-box streams and streaming lifetime: no
    dense probes or sketches are retained here. A repeated random box may
    occur in multiple probes; consumers must use owners, not a box-to-probe
    lookup. They retain the actual realization's G_beta/G_alpha for Eq. 4.3,
    including when seed=None gives different streams to different probes.
    """
    if type(k) is not int or k <= 0 or type(p) is not int or p < 0:
        raise ValueError("k must be positive and p nonnegative integers")
    start = perf_counter()
    graph = build_constraint_graph(root, lists, level, side)
    graph_seconds = perf_counter() - start
    selection = select_coloring(graph, root, level, k + p)
    selection = replace(
        selection,
        graph_build_seconds=graph_seconds,
        graph_incidences=graph.incidences,
        graph_conflict_visits=graph.conflict_visits,
    )
    if selection.strategy == "fixed":
        return build_admissible_schedule(root, lists, level, mesh, k, p, seed, side), selection

    active: dict[int, set[TreeNode]] = {}
    for vertex, constraint in enumerate(graph.constraints):
        active.setdefault(selection.colors[vertex], set()).add(constraint.random_box)
    for vertex, constraint in enumerate(graph.constraints):
        if active[selection.colors[vertex]] & constraint.zero_boxes:
            raise ValueError("selected color violates a zero-box constraint")

    by_color = {
        color: PeriodicTestMatrix(
            pattern=(color,),  # Diagnostic group label; not a periodic cell.
            active_boxes=sorted(boxes, key=lambda box: box.cell_coords),
            n_dofs=mesh.n_cols if side == "col" else mesh.n_rows,
            k=k,
            p=p,
            seed=seed,
            level=level,
            side=side,
        )
        for color, boxes in sorted(active.items())
    }
    owners = {pair: by_color[selection.colors[vertex]] for pair, vertex in graph.owners.items()}
    validate_admissible_probe_owners(owners, side)
    schedule = AdmissibleProbeSchedule(probes=list(by_color.values()), owners=owners)
    if len(schedule.probes) * (k + p) > selection.fixed_columns:
        raise ValueError("selected schedule exceeds fixed sampling columns")
    return schedule, selection
