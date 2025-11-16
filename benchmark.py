"""
Benchmark các thuật toán đường đi:
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

def benchmark_algorithm(G, start_node, goal_node, algo_name, landmarks=None, landmark_distances=None):
    """
    Chạy 1 thuật toán, đo thời gian, số node mở rộng, tổng cost, chiều dài path.
    """
    start_time = time.time()

    if algo_name == "dijkstra":
        path, visited, edges = dijkstra_with_penalty(G, start_node, goal_node)
    elif algo_name == "a_star":
        path, visited, edges = a_star_with_penalty(G, start_node, goal_node)
    elif algo_name == "bi_a_star":
        path, visited, edges = bidirectional_a_star_with_penalty(G, start_node, goal_node)
    elif algo_name == "alt":
        dist_L_to_v, dist_v_to_L = landmark_distances
        path, visited, edges = alt_a_star_with_penalty(
            G, start_node, goal_node, algo_name,
            landmarks=landmarks,
            dist_L_to_v=dist_L_to_v,
            dist_v_to_L=dist_v_to_L
        )
    else:
        raise ValueError(f"Unknown algorithm: {algo_name}")

    end_time = time.time()
    duration = end_time - start_time

    # Tính tổng trọng số đường đi
    total_weight = 0
    for i in range(len(path)-1):
        total_weight += G[path[i]][path[i+1]].get("weight", 1)

    result = {
        "time_sec": duration,
        "nodes_expanded": len(visited),
        "total_weight": total_weight,
        "path_length": len(path),
        "path": path
    }
    return result

def benchmark_all(G, start_node, goal_node, landmarks=None, landmark_distances=None):
    """
    Chạy benchmark cho 4 thuật toán chính
    """
    results = {}
    for algo in ["dijkstra", "a_star", "bi_a_star", "alt"]:
        if algo == "alt":
            res = benchmark_algorithm(G, start_node, goal_node, algo, landmarks, landmark_distances)
        else:
            res = benchmark_algorithm(G, start_node, goal_node, algo)
        results[algo] = res
    return results
