import heapq
from geopy.distance import geodesic

def transfer_penalty(node, G):
    degree = len(list(G.neighbors(node)))
    return degree * 0.3


def dijkstra_with_penalty(G, orig_node, dest_node):
    queue = [(0, orig_node)]
    visited = set()
    came_from = {}
    costs = {orig_node: 0}
    visited_nodes = []
    edges = []

    while queue:
        cost, current = heapq.heappop(queue)

        if current in visited:
            continue
        visited.add(current)
        visited_nodes.append(current)

        if current == dest_node:
            break

        for neighbor in G.neighbors(current):
            edge_data = G[current][neighbor]
            edge_weight = edge_data.get("weight", 1)
            penalty = transfer_penalty(neighbor, G)

            new_cost = costs[current] + edge_weight + penalty

            if neighbor not in costs or new_cost < costs[neighbor]:
                costs[neighbor] = new_cost
                came_from[neighbor] = current
                heapq.heappush(queue, (new_cost, neighbor))
                edges.append((current, neighbor))

    # reconstruct path
    path = []
    node = dest_node
    while node in came_from:
        path.append(node)
        node = came_from[node]
    path.append(orig_node)
    path.reverse()

    return path, visited_nodes, edges
