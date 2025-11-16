"""
Chạy benchmark cho:
- Dijkstra
- A*
- Bidirectional A*
- ALT A*
"""

from graph import load_graph
from benchmark import benchmark_all
from algorithms.alt_a_star import compute_landmark_distances
import math


def nearest_node(G, coord):
    """Tìm node gần nhất theo tọa độ lon/lat."""
    min_dist = float("inf")
    nearest = None

    for node in G.nodes:
        x = G.nodes[node].get("x", 0)
        y = G.nodes[node].get("y", 0)
        d = math.hypot(coord[0] - x, coord[1] - y)
        if d < min_dist:
            min_dist = d
            nearest = node

    return nearest


def main():
    # Load graph
    G = load_graph("data/graph/graph_data.pkl")
    print(f"Graph loaded: {len(G.nodes)} nodes, {len(G.edges)} edges")

    # Start / goal
    start_coord = (105.8181305, 21.001585)
    goal_coord  = (105.8205000, 21.0050000)

    start_node = nearest_node(G, start_coord)
    goal_node  = nearest_node(G, goal_coord)

    print("Start node:", start_node)
    print("Goal node :", goal_node)

    # Landmark cho ALT
    lm_coords = [
        (105.8181305, 21.001585),
        (105.8195000, 21.0030000),
        (105.8205000, 21.0050000),
        (105.8210000, 21.0065000)
    ]
    landmarks = [nearest_node(G, lm) for lm in lm_coords]

    print("Landmarks:", landmarks)
    landmark_distances = compute_landmark_distances(G, landmarks)

    # chạy benchmark
    results = benchmark_all(
        G, start_node, goal_node,
        landmarks=landmarks,
        landmark_distances=landmark_distances
    )

    # in kết quả
    print("\n===== BENCHMARK RESULTS =====")
    for algo, res in results.items():
        print(f"\n--- {algo.upper()} ---")
        print("Total cost   :", res["total_cost"])
        print("Time (sec)   :", round(res["time_sec"], 4))
        print("Nodes popped :", res["nodes_popped"])
        print("Path length  :", res["path_length"])


if __name__ == "__main__":
    main()
