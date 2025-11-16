"""
Chạy benchmark cho các thuật toán đường đi:
- Dijkstra với penalty
- A* với penalty
- Bidirectional A*
- ALT A*
"""

import networkx as nx
from graph import load_graph
from benchmark import benchmark_all
from algorithms.alt_a_star import compute_landmark_distances
import math

# --- Hàm tìm node gần nhất ---
def nearest_node(G, coord):
    min_dist = float('inf')
    nearest = None
    for node in G.nodes:
        x, y = G.nodes[node].get("x", 0), G.nodes[node].get("y", 0)
        dist = math.hypot(coord[0]-x, coord[1]-y)
        if dist < min_dist:
            min_dist = dist
            nearest = node
    return nearest

def main():
    # 1. Load graph
    graph_file = "data/graph/graph_data.pkl"
    print(f"Loading graph from {graph_file} ...")
    G = load_graph(graph_file)
    print(f"Graph has {len(G.nodes)} nodes and {len(G.edges)} edges")

    # 2. Start / goal node (tọa độ thực tế)
    start_coord = (105.8181305, 21.001585)
    goal_coord  = (105.8205000, 21.0050000)

    start_node = nearest_node(G, start_coord)
    goal_node  = nearest_node(G, goal_coord)

    print(f"Start node: {start_node}")
    print(f"Goal node: {goal_node}")

    # 3. Landmarks cho ALT A*
    landmarks_coords = [
        (105.8181305, 21.001585),
        (105.8195000, 21.0030000),
        (105.8205000, 21.0050000),
        (105.8210000, 21.0065000)
    ]
    landmarks = [nearest_node(G, lm) for lm in landmarks_coords]
    print(f"Landmarks used in graph: {landmarks}")

    # 4. Precompute landmark distances
    print("Precomputing landmark distances...")
    landmark_distances = compute_landmark_distances(G, landmarks)

    # 5. Chạy benchmark
    print("Running benchmark for all algorithms...")
    results = benchmark_all(G, start_node, goal_node, landmarks, landmark_distances)

    # 6. In kết quả chi tiết
    print("\n=== Benchmark Results ===")
    for algo, res in results.items():
        print(f"\nAlgorithm: {algo}")
        print(f"Time (s): {res['time_sec']:.3f}")
        print(f"Nodes expanded: {res['nodes_expanded']}")
        print(f"Total cost: {res['total_weight']:.2f}")
        print(f"Path length (nodes): {res['path_length']}")
        print(f"Path: {res['path']}")

if __name__ == "__main__":
    main()
