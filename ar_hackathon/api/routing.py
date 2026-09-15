"""
Amazon Robotics Hackathon - Routing API


*****IMPORTANT*****
Team name: Bezos3
Email address: aprameyaithal@gmail.com
*******************
"""

import heapq
from typing import Dict, List, Optional, Tuple
from ar_hackathon.models.graph_state import GraphState

_FULL_SLOT_FLAG = 1.0
_STATION_PENALTY = 15.0
_INF = 1e18

_adj: Dict[int, List[Tuple[int, float]]] = {}
_edge_bidir: Dict[Tuple[int, int], bool] = {}
_cap_by_key: Dict[object, Optional[int]] = {}
_node_type: Dict[int, str] = {}
_node_cap: Dict[int, float] = {}
_storages: List[int] = []
_static_dist: Dict[int, Dict[int, float]] = {}
_claims: Dict[str, int] = {}
_graph_key = None


def _ensure_graph(state: GraphState) -> None:
    global _adj, _edge_bidir, _cap_by_key, _node_type, _node_cap
    global _storages, _static_dist, _graph_key

    ids = tuple(sorted(n.id for n in state.nodes))
    ekey = tuple(sorted((e.from_node, e.to_node, e.bidirectional, e.weight)
                        for e in state.edges))
    key = (ids, ekey)
    if _graph_key == key:
        return

    _adj = {}
    _edge_bidir = {}
    _cap_by_key = {}
    _node_type = {}
    _node_cap = {}
    _storages = []
    _static_dist = {}
    _graph_key = key

    for n in state.nodes:
        _adj[n.id] = []
        _node_type[n.id] = n.node_type
        _node_cap[n.id] = n.capacity if n.capacity is not None else float('inf')
        if n.node_type == 'storage':
            _storages.append(n.id)

    for e in state.edges:
        if e.from_node == e.to_node:
            continue
        _edge_bidir[(e.from_node, e.to_node)] = e.bidirectional
        key = frozenset((e.from_node, e.to_node)) if e.bidirectional \
            else (e.from_node, e.to_node)
        _cap_by_key[key] = e.capacity
        _adj[e.from_node].append((e.to_node, float(e.weight)))
        if e.bidirectional:
            _edge_bidir[(e.to_node, e.from_node)] = True
            _adj[e.to_node].append((e.from_node, float(e.weight)))


def _compute_static(source: int) -> Dict[int, float]:
    dist = {source: 0.0}
    heap = [(0.0, source)]
    while heap:
        d, u = heapq.heappop(heap)
        if d > dist.get(u, _INF):
            continue
        for v, w in _adj[u]:
            nd = d + w
            if nd < dist.get(v, _INF):
                dist[v] = nd
                heapq.heappush(heap, (nd, v))
    return dist


def _dist_between(a: int, b: int) -> float:
    if a == b:
        return 0.0
    d = _static_dist.get(a)
    if d is None:
        d = _compute_static(a)
        _static_dist[a] = d
    return d.get(b, _INF)


def _build_congestion(state: GraphState):
    edge_occ: Dict[object, int] = {}
    edge_rems: Dict[object, List[float]] = {}
    node_occ: Dict[int, int] = {}
    node_rems: Dict[int, List[float]] = {}

    for unit in state.drive_units:
        if not unit.in_transit:
            node_occ[unit.current_node] = node_occ.get(unit.current_node, 0) + 1
            node_rems.setdefault(unit.current_node, []).append(0.0)
        else:
            dest = unit.transit_destination
            node_occ[dest] = node_occ.get(dest, 0) + 1
            node_rems.setdefault(dest, []).append(float(unit.transit_remaining_time))
            key = frozenset((unit.current_node, dest)) if _edge_bidir.get(
                (unit.current_node, dest)) else (unit.current_node, dest)
            edge_occ[key] = edge_occ.get(key, 0) + 1
            edge_rems.setdefault(key, []).append(float(unit.transit_remaining_time))

    edge_wait: Dict[object, float] = {}
    for key, count in edge_occ.items():
        cap = _cap_by_key.get(key)
        if cap is not None and count >= cap:
            rems = sorted(edge_rems[key])
            edge_wait[key] = rems[min(count - cap, len(rems) - 1)]

    node_wait: Dict[int, float] = {}
    for node_id, count in node_occ.items():
        cap = _node_cap[node_id]
        if count >= cap:
            rems = sorted(node_rems[node_id])
            node_wait[node_id] = rems[min(count - int(cap), len(rems) - 1)]

    return edge_occ, edge_wait, node_occ, node_wait


def _plan_next(state: GraphState, start: int, goal: int,
               edge_wait, node_wait) -> Optional[int]:
    if start == goal:
        return None

    dist = {start: 0.0}
    prev = {start: None}
    heap = [(0.0, start)]

    while heap:
        d, u = heapq.heappop(heap)
        if d > dist.get(u, _INF):
            continue
        if u == goal:
            break
        for v, w in _adj[u]:
            cost = w
            key = frozenset((u, v)) if _edge_bidir.get((u, v)) else (u, v)
            wait = edge_wait.get(key, 0.0)
            if wait > 0.0:
                cost += wait + _FULL_SLOT_FLAG
            node_w = node_wait.get(v, 0.0)
            if node_w > 0.0:
                cost += node_w + _FULL_SLOT_FLAG
            if _node_type[v] == 'station' and v != goal:
                cost += _STATION_PENALTY
            nd = d + cost
            if nd < dist.get(v, _INF):
                dist[v] = nd
                prev[v] = u
                heapq.heappush(heap, (nd, v))

    if goal not in prev:
        return None

    node = goal
    while prev[node] != start:
        node = prev[node]
        if node is None:
            return None
    return node


def _pick_goal(state: GraphState, unit, node_occ: Dict[int, int]) -> Optional[int]:
    if unit.carrying:
        pod = state.get_pod(unit.carrying[0])
        if pod is not None:
            _claims[pod.id] = unit.id
            return pod.destination_station
        return None

    waiting = [p for p in state.active_pods
               if p.carried_by is None and p.current_node is not None]
    active_ids = {p.id for p in state.active_pods}
    carried_ids = {p.id for p in state.active_pods if p.carried_by is not None}
    for pid in list(_claims):
        if pid not in active_ids or (pid in carried_ids and _claims[pid] != unit.id):
            del _claims[pid]
    for pid in unit.carrying:
        _claims[pid] = unit.id

    cands = [p for p in waiting if _claims.get(p.id, unit.id) == unit.id]
    if cands:
        pos = unit.current_node

        def score(p):
            return (_dist_between(pos, p.current_node), p.entry_time,
                    node_occ.get(p.destination_station, 0), p.id)

        best = min(cands, key=score)
        _claims[best.id] = unit.id
        return best.current_node

    if not _storages:
        return None

    pos = unit.current_node
    best_node = None
    best_score = None
    for s in _storages:
        sc = (_dist_between(pos, s), s)
        if best_score is None or sc < best_score:
            best_score = sc
            best_node = s
    return best_node if best_node != pos else None


def drive_unit_next_move(drive_unit_id: int, state: GraphState) -> Optional[int]:
    """
    Determine the next node for a drive unit to move to.

    Pickups and deliveries happen automatically in the engine; this function
    only decides the direction of travel. A returned None means the unit
    waits one time step.

    Args:
        drive_unit_id: ID of the drive unit being routed
        state: GraphState object containing the current state of the floor

    Returns:
        next_node_id: ID of an adjacent node to move to, or None to wait
    """
    try:
        _ensure_graph(state)
        unit = state.get_drive_unit(drive_unit_id)
        if unit is None or unit.in_transit:
            return None

        _, edge_wait, node_occ, node_wait = _build_congestion(state)
        goal = _pick_goal(state, unit, node_occ)
        if goal is None:
            return None
        return _plan_next(state, unit.current_node, goal,
                          edge_wait, node_wait)
    except Exception:
        return None