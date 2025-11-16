import random
import time
import math
from graph import load_graph
from algorithms.dijkstra import dijkstra_with_penalty
from algorithms.a_star import a_star_with_penalty
from algorithms.bidirectional_a_star import bidirectional_a_star_with_penalty
from algorithms.alt_a_star import alt_a_star_with_penalty

# ======================
# Hàm tìm node gần nhất
# ======================
def nearest_node(G, coord):
    min_dist = float('inf')
    nearest = None
    for node in G.nodes:
        dist = math.hypot(coord[0]-node[0], coord[1]-node[1])
        if dist < min_dist:
            min_dist = dist
            nearest = node
    return nearest

# ======================
# Hàm benchmark 1 thuật toán
# ======================
def benchmark_algorithm(algo_fn, G, start, goal, landmarks=None):
    start_time = time.time()
    
    if algo_fn.__name__ == "alt_a_star_with_penalty":
        # ALT A* dùng landmarks
        path, visited, _ = algo_fn(G, start, goal, landmarks=landmarks)
    else:
        path, visited, _ = algo_fn(G, start, goal)

    elapsed = time.time() - start_time
    total_cost = sum(
        G[path[i]][path[i+1]].get("weight", 1) + 0.3*len(list(G.neighbors(path[i+1])))
        for i in range(len(path)-1)
    )
    return {
        "total_cost": total_cost,
        "time_sec": elapsed,
        "nodes_popped": len(visited),
        "path_length": len(path)
    }

# ======================
# Benchmark nhiều query
# ======================
def benchmark_all(G, landmarks=None, num_trials=10):
    nodes_list = list(G.nodes)
    algorithms = {
        "dijkstra": dijkstra_with_penalty,
        "astar": a_star_with_penalty,
        "bi_astar": bidirectional_a_star_with_penalty,
        "alt": alt_a_star_with_penalty
    }

    results_all = { algo: [] for algo in algorithms }

    for i in range(num_trials):
        start = random.choice(nodes_list)
        goal = random.choice(nodes_list)
        if start == goal:
            continue

        for algo_name, algo_fn in algorithms.items():
            res = benchmark_algorithm(algo_fn, G, start, goal, landmarks)
            results_all[algo_name].append(res)

    # Tính trung bình
    results_avg = {}
    for algo_name, res_list in results_all.items():
        n = len(res_list)
        results_avg[algo_name] = {
            "total_cost": sum(r["total_cost"] for r in res_list)/n,
            "time_sec": sum(r["time_sec"] for r in res_list)/n,
            "nodes_popped": sum(r["nodes_popped"] for r in res_list)/n,
            "path_length": sum(r["path_length"] for r in res_list)/n
        }

    return results_avg

# ======================
# Main chạy benchmark
# ======================
def main():
    # Load graph
    graph_file = "data/graph/graph_data.pkl"
    print(f"Loading graph from {graph_file} ...")
    G = load_graph(graph_file)
    print(f"Graph loaded: {len(G.nodes)} nodes, {len(G.edges)} edges")

    # Landmarks cho ALT
    landmarks_coords = [
        (105.8181305, 21.001585),
        (105.8196143, 21.0030077),
        (105.8202593, 21.0049961),
        (105.820742, 21.0065424)
    ]
    landmarks = [nearest_node(G, lm) for lm in landmarks_coords]
    print(f"Landmarks: {landmarks}")

    # Chạy benchmark nhiều query
    num_trials = 10
    print(f"Running benchmark with {num_trials} random queries...")
    results = benchmark_all(G, landmarks=landmarks, num_trials=num_trials)

    # In kết quả
    print("\n===== BENCHMARK RESULTS (AVERAGE) =====")
    for algo, res in results.items():
        print(f"\n--- {algo.upper()} ---")
        print(f"Avg Total cost   : {res['total_cost']:.2f}")
        print(f"Avg Time (sec)   : {res['time_sec']:.5f}")
        print(f"Avg Nodes popped : {res['nodes_popped']:.2f}")
        print(f"Avg Path length  : {res['path_length']:.2f}")

if __name__ == "__main__":
    main()
