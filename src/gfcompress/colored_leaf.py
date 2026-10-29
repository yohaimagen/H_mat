"""Internal colored identity probes for neighbor-only dense-leaf extraction."""

from gfcompress.coloring import ColoringSelection, select_coloring
from gfcompress.constraint_graph import build_constraint_graph
from gfcompress.fixed_pattern import LeafProbeSchedule, PeriodicLeafTestMatrix, build_leaf_schedule
from gfcompress.geometry import FaultMesh
from gfcompress.interactions import TreeLists
from gfcompress.tree import TreeNode


def build_colored_leaf_schedule(
    root: TreeNode, lists: TreeLists, level: int, mesh: FaultMesh
) -> tuple[LeafProbeSchedule, ColoringSelection]:
    """Select pair-owned identity probes at the fixed global column width.

    Pass this schedule to ``extract_leaves`` with all far-field factors. Every
    active source shares identity column slots 0..w_beta-1; pair ownership
    isolates its other neighbors, not the residual error from approximate
    peeling. The existing consumer realizes/releases one probe/sample at a
    time and copies each rectangular dense block once. No adjoint sampling is
    needed for dense-leaf extraction.
    """
    nodes = root.nodes_at_level(level)
    if not nodes or any(not node.is_leaf for node in nodes):
        raise ValueError("expected a nonempty leaf level")
    width = max(len(node.col_indices) for node in nodes)
    graph = build_constraint_graph(root, lists, level, neighbors_only=True)
    selection = select_coloring(graph, root, level, width, leaf=True)
    if selection.strategy == "fixed":
        return build_leaf_schedule(root, lists, level, mesh), selection

    active: dict[int, set[TreeNode]] = {}
    for vertex, constraint in enumerate(graph.constraints):
        active.setdefault(selection.colors[vertex], set()).add(constraint.random_box)
    for vertex, constraint in enumerate(graph.constraints):
        if active[selection.colors[vertex]] & constraint.zero_boxes:
            raise ValueError("selected leaf color violates a zero-box constraint")
    by_color = {
        color: PeriodicLeafTestMatrix(
            pattern=(color,),
            active_boxes=sorted(boxes, key=lambda box: box.cell_coords),
            n_cols=mesh.n_cols,
            w_max=width,
        )
        for color, boxes in sorted(active.items())
    }
    owners = {pair: by_color[selection.colors[vertex]] for pair, vertex in graph.owners.items()}
    schedule = LeafProbeSchedule(probes=list(by_color.values()), owners=owners)
    if len(schedule.probes) * width > selection.fixed_columns:
        raise ValueError("selected leaf schedule exceeds fixed sampling columns")
    return schedule, selection
