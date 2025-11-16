import heapq
from geopy.distance import geodesic
import networkx as nx


# ============================================================
#  TRANSFER PENALTY
# ============================================================
def transfer_penalty(node, G):
    """
    Penalty tại node để mô phỏng chi phí chuyển tuyến.
    Có thể chỉnh sửa tùy bài toán.
    """
    degree = len(list(G.neighbors(node)))
    return degree * 0.3       # hệ số 0.3 có thể tinh chỉnh


# ============================================================
#  HEURISTIC ALT = tối đa 4 công thức tam giác landmark
# ============================================================
def alt_heuristic(v, t, landmarks, dist_L_to_v, dist_v_to_L):
    """
    Heuristic ALT: h(v) = max over L of:
        |d(L, t) - d(L, v)|,
        |d(v, L) - d(t, L)|
    """
    h_value = 0
    for L in landmarks:
        h1 = abs(dist_L_to_v[L][t] - dist_L_to_v[L][v])
        h2 = abs(dist_v_to_L[L][t] - dist_v_to_L[L][v])
        h_value = max(h_value, h1, h2)
    return h_value


# ============================================================
#  PRECOMPUTE DISTANCES FROM LANDMARKS (2 CHIỀU)
# ============================================================
def compute_landmark_distances(G, landmarks):
    """
    Tính trước:
        - dist_L_to_v[L][v] : khoảng cách từ landmark → mọi node
        - dist_v_to_L[L][v] : khoảng cách từ mọi node → landmark
    Sử dụng Dijkstra chuẩn.
    """

    dist_L_to_v = {}   # L → v
    dist_v_to_L = {}   # v → L

    for L in landmarks:
        # 1. Từ landmark -> tất cả node
        dist_L_to_v[L] = nx.single_source_dijkstra_path_length(G, L, weight="weight")

        # 2. Từ tất cả node -> landmark (chạy trên graph đảo chiều)
        rev_G = G.reverse(copy=False)
        dist_v_to_L[L] = nx.single_source_dijkstra_path_length(rev_G, L, weight="weight")

    return dist_L_to_v, dist_v_to_L


# ============================================================
#  ALT A* WITH TRANSFER PENALTY
# ============================================================
def alt_a_star_with_penalty(G, orig_node, dest_node, landmarks,
                            dist_L_to_v, dist_v_to_L):
    """
    Thuật toán ALT A* có thêm transfer penalty.
    Trả về:
        - path: đường đi cuối cùng
        - visited_nodes: danh sách node đã duyệt
        - edges: cạnh đã relax (debug/visualization)
    """

    queue = [(0, orig_node)]
    visited = set()
    came_from = {}
    costs = {orig_node: 0}

    edges = []
    visited_nodes = []

    while queue:
        _, current = heapq.heappop(queue)

        if current in visited:
            continue

        visited.add(current)
        visited_nodes.append(current)

        if current == dest_node:
            break

        for nb in G.neighbors(current):
            edge_data = G[current][nb]
            w = edge_data.get("weight", 1)

            # transfer penalty
            p = transfer_penalty(nb, G)

            new_cost = costs[current] + w + p

            # ALT heuristic
            h = alt_heuristic(
                nb, dest_node,
                landmarks,
                dist_L_to_v,
                dist_v_to_L
            )

            priority = new_cost + h

            if nb not in costs or new_cost < costs[nb]:
                costs[nb] = new_cost
                came_from[nb] = current
                heapq.heappush(queue, (priority, nb))
                edges.append((current, nb))

    # reconstruct path
    path = []
    node = dest_node
    while node in came_from:
        path.append(node)
        node = came_from[node]
    path.append(orig_node)
    path.reverse()

    return path, visited_nodes, edges
