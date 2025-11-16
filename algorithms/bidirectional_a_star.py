import heapq
from geopy.distance import geodesic


# -------------------------------
#  PENALTY FUNCTION (Bạn có thể sửa tuỳ bài toán)
# -------------------------------
def transfer_penalty(node, G):
    """
    Penalty tại node để mô phỏng chi phí chuyển tuyến.
    Bạn có thể thay đổi hàm này cho phù hợp với bài nghiên cứu.
    """
    degree = len(list(G.neighbors(node)))
    return degree * 0.3     # hệ số 0.3 có thể tinh chỉnh


# -------------------------------
#  GEODESIC HEURISTIC
# -------------------------------
def heuristic(node, target, G):
    """
    Heuristic sử dụng khoảng cách địa lý (geodesic).
    """
    return geodesic(
        (G.nodes[node]["y"], G.nodes[node]["x"]),
        (G.nodes[target]["y"], G.nodes[target]["x"])
    ).meters


# -------------------------------
#  BIDIRECTIONAL A*
# -------------------------------
def bidirectional_a_star_with_penalty(G, orig_node, dest_node):
    """
    Thuật toán Bi-A* 2 chiều có bổ sung transfer penalty.
    Trả về:
        - path: đường đi ngắn nhất
        - visited: các node đã duyệt
        - edges: danh sách cạnh đã relax (debug/visualization)
    """

    # Priority queues cho forward và backward
    forward_queue = [(0, orig_node)]
    backward_queue = [(0, dest_node)]

    # Cost của mỗi phía
    forward_cost = {orig_node: 0}
    backward_cost = {dest_node: 0}

    # Lưu trace
    came_forward = {}
    came_backward = {}

    # Visited sets
    visited_forward = set()
    visited_backward = set()

    # Node gặp nhau
    meet_node = None
    edges = []

    # -------------------------------
    #  MAIN LOOP
    # -------------------------------
    while forward_queue and backward_queue:

        # =========================
        #       EXPAND FORWARD
        # =========================
        _, current_f = heapq.heappop(forward_queue)
        visited_forward.add(current_f)

        # Kiểm tra gặp backward
        if current_f in visited_backward:
            meet_node = current_f
            break

        # Relax cạnh
        for nb in G.neighbors(current_f):
            edge_data = G[current_f][nb]
            w = edge_data.get("weight", 1)
            p = transfer_penalty(nb, G)

            new_cost = forward_cost[current_f] + w + p

            if nb not in forward_cost or new_cost < forward_cost[nb]:
                forward_cost[nb] = new_cost
                came_forward[nb] = current_f

                priority = new_cost + heuristic(nb, dest_node, G)
                heapq.heappush(forward_queue, (priority, nb))
                edges.append((current_f, nb))

        # =========================
        #       EXPAND BACKWARD
        # =========================
        _, current_b = heapq.heappop(backward_queue)
        visited_backward.add(current_b)

        # Kiểm tra gặp forward
        if current_b in visited_forward:
            meet_node = current_b
            break

        # Relax cạnh backward
        for nb in G.neighbors(current_b):
            edge_data = G[current_b][nb]
            w = edge_data.get("weight", 1)
            p = transfer_penalty(nb, G)

            new_cost = backward_cost[current_b] + w + p

            if nb not in backward_cost or new_cost < backward_cost[nb]:
                backward_cost[nb] = new_cost
                came_backward[nb] = current_b

                priority = new_cost + heuristic(nb, orig_node, G)
                heapq.heappush(backward_queue, (priority, nb))
                edges.append((current_b, nb))

    # -------------------------------
    #  NẾU KHÔNG GẶP NHAU → FAIL
    # -------------------------------
    if meet_node is None:
        return [], list(visited_forward | visited_backward), edges

    # -------------------------------
    #  RECONSTRUCT FORWARD PATH
    # -------------------------------
    forward_path = []
    node = meet_node
    while node in came_forward:
        forward_path.append(node)
        node = came_forward[node]
    forward_path.append(orig_node)
    forward_path.reverse()

    # -------------------------------
    #  RECONSTRUCT BACKWARD PATH
    # -------------------------------
    backward_path = []
    node = meet_node
    while node in came_backward:
        node = came_backward[node]
        backward_path.append(node)

    # -------------------------------
    #  FULL FINAL PATH
    # -------------------------------
    full_path = forward_path + backward_path

    return full_path, list(visited_forward | visited_backward), edges
