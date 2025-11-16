import heapq
from geopy.distance import geodesic
import networkx as nx

def transfer_penalty(node, G):
    """
    Penalty tại node để mô phỏng chi phí chuyển tuyến.
    """
    degree = len(list(G.neighbors(node)))
    return degree * 0.3  # hệ số có thể tinh chỉnh

def alt_heuristic(v, t, landmarks, dist_L_to_v, dist_v_to_L):
    """
    ALT heuristic: max(|d(L,t)-d(L,v)|, |d(v,L)-d(t,L)|)
    """
    h_value = 0
    for L in landmarks:
        h1 = abs(dist_L_to_v[L].get(t, 0) - dist_L_to_v[L].get(v, 0))
        h2 = abs(dist_v_to_L[L].get(t, 0) - dist_v_to_L[L].get(v, 0))
        h_value = max(h_value, h1, h2)
    return h_value

def select_landmarks(G, k=4):
    """
    Chọn k landmarks tự động: corners + farthest-point
    """
    nodes = list(G.nodes)

    # 1. Chọn 4 góc bản đồ
    north = max(nodes, key=lambda n: G.nodes[n]["y"])
    south = min(nodes, key=lambda n: G.nodes[n]["y"])
    east = max(nodes, key=lambda n: G.nodes[n]["x"])
    west = min(nodes, key=lambda n: G.nodes[n]["x"])
    landmarks = list({north, south, east, west})

    # 2. Nếu k > 4, thêm các node xa nhất
    while len(landmarks) < k:
        max_dist = -1
        candidate = None
        for n in nodes:
            if n in landmarks:
                continue
            min_to_landmarks = min(
                geodesic((G.nodes[n]["y"], G.nodes[n]["x"]),
                         (G.nodes[l]["y"], G.nodes[l]["x"])).meters
                for l in landmarks
            )
            if min_to_landmarks > max_dist:
                max_dist = min_to_landmarks
                candidate = n
        if candidate:
            landmarks.append(candidate)
        else:
            break
    return landmarks

def compute_landmark_distances(G, landmarks):
    """
    Precompute distances from landmarks to all nodes and vice versa
    """
    dist_L_to_v = {}
    dist_v_to_L = {}
    for L in landmarks:
        dist_L_to_v[L] = nx.single_source_dijkstra_path_length(G, L, weight="weight")
        rev_G = G.reverse(copy=False)
        dist_v_to_L[L] = nx.single_source_dijkstra_path_length(rev_G, L, weight="weight")
    return dist_L_to_v, dist_v_to_L

def alt_a_star_with_penalty(G, orig_node, dest_node, k_landmarks=4,
                            landmarks=None, dist_L_to_v=None, dist_v_to_L=None):
    """
    ALT A* search with transfer penalty.
    Trả về 3 giá trị: path, visited_nodes, edges
    """
    # --- Landmark selection ---
    if landmarks is None:
        landmarks = select_landmarks(G, k_landmarks)

    # --- Precompute distances nếu chưa có ---
    if dist_L_to_v is None or dist_v_to_L is None:
        dist_L_to_v, dist_v_to_L = compute_landmark_distances(G, landmarks)

    # --- Priority queue ---
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
            w = G[current][nb].get("weight", 1)
            p = transfer_penalty(nb, G)
            new_cost = costs[current] + w + p

            h = alt_heuristic(nb, dest_node, landmarks, dist_L_to_v, dist_v_to_L)
            priority = new_cost + h

            if nb not in costs or new_cost < costs[nb]:
                costs[nb] = new_cost
                came_from[nb] = current
                heapq.heappush(queue, (priority, nb))
                edges.append((current, nb))

    # --- Reconstruct path ---
    path = []
    node = dest_node
    while node in came_from:
        path.append(node)
        node = came_from[node]
    path.append(orig_node)
    path.reverse()

    return path, visited_nodes, edges
