import heapq
import random
import networkx as nx


def transfer_penalty(node, G, coef=0.3):
    return G.degree(node) * coef

def alt_heuristic(v, t, landmarks, dist_L_to_v, dist_v_to_L):
    h_value = 0
    for L in landmarks:
        h1 = abs(dist_L_to_v[L].get(t, float('inf')) - dist_L_to_v[L].get(v, float('inf')))
        h2 = abs(dist_v_to_L[L].get(t, float('inf')) - dist_v_to_L[L].get(v, float('inf')))
        h_value = max(h_value, h1, h2)
    return h_value

def compute_landmark_distances(G, landmarks):
    dist_L_to_v = {}
    dist_v_to_L = {}
    for L in landmarks:
        dist_L_to_v[L] = nx.single_source_dijkstra_path_length(G, L, weight="weight")
        dist_v_to_L[L] = nx.single_source_dijkstra_path_length(G.reverse(copy=False), L, weight="weight")
    return dist_L_to_v, dist_v_to_L

def select_landmarks(G, k=4):
    k_hub = k // 2
    k_periphery = k - k_hub

    hubs = sorted(G.degree, key=lambda x: x[1], reverse=True)
    hub_nodes = [n for n, d in hubs[:k_hub]]

    centrality = nx.closeness_centrality(G)
    periphery_nodes = sorted(centrality.items(), key=lambda x: x[1])
    periphery_nodes = [n for n, c in periphery_nodes if n not in hub_nodes][:k_periphery]

    landmarks = hub_nodes + periphery_nodes

    while len(landmarks) < k:
        candidate = random.choice(list(G.nodes()))
        if candidate not in landmarks:
            landmarks.append(candidate)

    return landmarks

def alt_a_star_with_penalty(G, orig_node, dest_node, landmarks,
                            dist_L_to_v, dist_v_to_L):
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

    # Reconstruct path
    path = []
    node = dest_node
    while node in came_from:
        path.append(node)
        node = came_from[node]
    path.append(orig_node)
    path.reverse()
    return path, visited_nodes, edges

if __name__ == "__main__":
    G = nx.DiGraph()
    edges = [
        (0,1,2), (0,2,4), (1,2,1), (1,3,7), (2,3,3),
        (3,4,1), (2,4,5), (4,5,2), (3,5,3)
    ]
    for u,v,w in edges:
        G.add_edge(u,v,weight=w)

    landmarks = select_landmarks(G, k=3)
    print("Selected Landmarks:", landmarks)

    dist_L_to_v, dist_v_to_L = compute_landmark_distances(G, landmarks)

    path, visited_nodes, edges_relaxed = alt_a_star_with_penalty(
        G, 0, 5, landmarks, dist_L_to_v, dist_v_to_L
    )

    print("Path:", path)
    print("Visited Nodes:", visited_nodes)
    print("Edges Relaxed:", edges_relaxed)
