"""
Benchmark cho 4 thuật toán:
- Dijkstra với transfer penalty
- A* với transfer penalty
- Bidirectional A*
- ALT A*
"""

import time

from algorithms.dijkstra import dijkstra_with_penalty
from algorithms.a_star import a_star_with_penalty
from algorithms.bidirectional_a_star import bidirectional_a_star_with_penalty
from algorithms.alt_a_star import alt_a_star_with_penalty


def compute_total_cost(G, path):
    """Tính tổng chi phí = sum(edge_weight) + sum(transfer_penalty)."""
    if not path or len(path) < 2:
        return 0

    total = 0
    for i in range(len(path) - 1):
        total += G[path[i]][path[i + 1]].get("weight", 1)
    return total


def benchmark_algorithm(G, start_node, goal_node, algo_name,
                        landmarks=None, landmark_distances=None):

    start_time = time.time()

    # === chạy thuật toán ===
    if algo_name == "dijkstra":
        path, visited, _ = dijkstra_with_penalty(G, start_node, goal_node)

    elif algo_name == "astar":
        path, visited, _ = a_star_with_penalty(G, start_node, goal_node)

    elif algo_name == "bi_astar":
        path, visited, _ = bidirectional_a_star_with_penalty(G, start_node, goal_node)

    elif algo_name == "alt":
        path, visited, _ = alt_a_star_with_penalty(
            G, start_node, goal_node,
            landmarks=landmarks
        )

    else:
        raise ValueError("Unknown algorithm: " + algo_name)

    time_sec = time.time() - start_time
    total_cost = compute_total_cost(G, path)
    nodes_popped = len(visited)
    path_length = len(path)

    return {
        "total_cost": total_cost,
        "time_sec": time_sec,
        "nodes_popped": nodes_popped,
        "path_length": path_length,
        "path": path
    }


def benchmark_all(G, start_node, goal_node,
                  landmarks=None, landmark_distances=None):

    results = {}

    algos = ["dijkstra", "astar", "bi_astar", "alt"]

    for algo in algos:
        if algo == "alt":
            results[algo] = benchmark_algorithm(
                G, start_node, goal_node,
                algo,
                landmarks=landmarks,
                landmark_distances=landmark_distances
            )
        else:
            results[algo] = benchmark_algorithm(G, start_node, goal_node, algo)

    return results
