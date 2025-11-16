from .dijkstra import dijkstra_with_penalty
from .a_star import a_star_with_penalty
from .bidirectional_a_star import bidirectional_a_star_with_penalty
from .alt_a_star import alt_a_star_with_penalty

# Ánh xạ tên thuật toán với hàm tương ứng
ALGORITHMS = {
    "dijkstra": dijkstra_with_penalty,
    "a_star": a_star_with_penalty,
    "bidirectional_a_star": bidirectional_a_star_with_penalty,
    "alt_a_star": alt_a_star_with_penalty
    # "greedy": greedy_best_first_search
}

def find_shortest_path(G, orig_node, dest_node, vehicle, algorithm):

    algo_fn = ALGORITHMS.get(algorithm)
    if algo_fn:
        path, visited, edges = algo_fn(G, orig_node, dest_node)
        return path, visited, edges, [], []

    return None
