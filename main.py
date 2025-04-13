from GraphDB import Graph, Node

g1 = Graph()
g1.add_edge("A", "B", 2)
g1.add_edge("A", "C", 5)
g1.add_edge("B", "C", 1)
g1.add_edge("C", "D", 3)
g1.add_edge("D", "A", 4)

g2 = Graph()
edges = [("A", "B", 7), ("A", "C", 9), ("B", "D", 10), ("C", "D", 11),
         ("C", "E", 2), ("D", "F", 6), ("E", "F", 1)]
for u, v, w in edges:
    g2.add_edge(u, v, w)
    g2.add_edge(v, u, w)

g3 = Graph()
g3.add_edge("A", "B", 4)
g3.add_edge("A", "C", 8)
g3.add_edge("C", "D", 7)
g3.add_edge("D", "E", 9)
g3.add_edge("E", "F", 10)
g3.add_edge("F", "G", 2)

g4 = Graph()
nodes = ["A", "B", "C", "D", "E"]
for i in range(len(nodes)):
    for j in range(len(nodes)):
        if i != j:
            g4.add_edge(nodes[i], nodes[j], (i + j) % 4 + 1)

def bellman_ford(graph, start_value):
    dist = {node: float('inf') for node in graph.nodes}
    dist[start_value]=0

    for i in range(len(graph.nodes)-1):
        for node in graph.nodes.values():
            for neighbor, weight in node.adjacent.items():
                if dist[node.value]+weight < dist[neighbor.value]:
                    dist[neighbor.value] = dist[node.value] + weight

    return dist

def dijkstra(graph, start_value):
    dist = {node: float('inf') for node in graph.nodes}
    dist[start_value]=0
    visited = set()

    while len(visited) < len(graph.nodes):
        min_node = None
        min_dist=float('inf')
        for node, d in dist.items():
            if node not in visited and d<min_dist:
                min_dist=d
                min_node = node

        if min_node is None:
            break

        visited.add(min_node)
        current_node = graph.nodes[min_node]

        for neighbor, weight in current_node.adjacent.items():
            if neighbor.value not in visited:
                new_dist = dist[min_node] + weight
                if new_dist < dist[neighbor.value]:
                    dist[neighbor.value] = new_dist
    
    return dist

for graph in {g1,g2,g3,g4}:
    print(graph)
    print("Dijkstra: ", dijkstra(graph, "A"))
    print("Bellman-Ford: ", bellman_ford(graph, "A"))