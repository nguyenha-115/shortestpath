import heapq
from geopy.distance import geodesic

def transfer_penalty(node, G):
    """
    Penalty tại một node để mô phỏng chi phí chuyển tuyến (vertex transfer penalty).
    Bạn có thể tinh chỉnh lại rule này cho phù hợp bài nghiên cứu.

    Gợi ý:
    - penalty dựa trên degree (nút có nhiều cạnh → giao lộ → tốn thời gian)
    - penalty dựa trên loại đường (nếu bạn có data)
    """
    degree = len(list(G.neighbors(node)))
    return degree * 0.3      # hệ số 0.3 có thể tinh chỉnh trong phần benchmark


def a_star_with_penalty(G, orig_node, dest_node):
    """
    A* với vertex transfer penalty.
    Dựa vào code A* gốc của bạn nhưng thêm penalty vào new_cost.
    Trả về:
        - path: đường đi cuối cùng
        - visited_nodes: các node đã duyệt (để debug / visualization)
        - edges: danh sách cạnh đã relax
    """

    queue = [(0, orig_node)]
    visited = set()
    came_from = {}
    costs = {orig_node: 0}
    visited_nodes = []
    edges = []

    while queue:
        _, current = heapq.heappop(queue)
        if current in visited:
            continue

        visited.add(current)
        visited_nodes.append(current)

        if current == dest_node:
            break

        for neighbor in G.neighbors(current):
            edge_data = G[current][neighbor]
            edge_weight = edge_data.get("weight", 1)

            # --- tính penalty tại neighbor ---
            penalty = transfer_penalty(neighbor, G)

            # --- cost mới đã bao gồm penalty ---
            new_cost = costs[current] + edge_weight + penalty

            # --- heuristic A*: khoảng cách địa lý đến đích ---
            heuristic = geodesic(
                (G.nodes[neighbor]["y"], G.nodes[neighbor]["x"]),
                (G.nodes[dest_node]["y"], G.nodes[dest_node]["x"])
            ).meters

            priority = new_cost + heuristic

            if neighbor not in costs or new_cost < costs[neighbor]:
                costs[neighbor] = new_cost
                heapq.heappush(queue, (priority, neighbor))
                came_from[neighbor] = current
                edges.append((current, neighbor))

    # --- truy vết đường đi ---
    path = []
    node = dest_node
    while node in came_from:
        path.append(node)
        node = came_from[node]
    path.append(orig_node)
    path.reverse()

    return path, visited_nodes, edges
