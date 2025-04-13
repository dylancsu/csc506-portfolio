from GraphDB import Graph, Node

g1 = Graph()
g1.add_edge("A", "B", 2)
g1.add_edge("A", "C", 5)
g1.add_edge("B", "C", 1)
g1.add_edge("C", "D", 3)
g1.add_edge("D", "A", 4)
print("Graph 1:")
print(g1)

g2 = Graph()
edges = [("A", "B", 7), ("A", "C", 9), ("B", "D", 10), ("C", "D", 11),
         ("C", "E", 2), ("D", "F", 6), ("E", "F", 1)]
for u, v, w in edges:
    g2.add_edge(u, v, w)
    g2.add_edge(v, u, w)
print("\nGraph 2:")
print(g2)

g3 = Graph()
g3.add_edge("1", "2", 4)
g3.add_edge("1", "3", 8)
g3.add_edge("3", "4", 7)
g3.add_edge("4", "5", 9)
g3.add_edge("5", "6", 10)
g3.add_edge("6", "7", 2)
print("\nGraph 3:")
print(g3)

g4 = Graph()
nodes = ["A", "B", "C", "D", "E"]
for i in range(len(nodes)):
    for j in range(len(nodes)):
        if i != j:
            g4.add_edge(nodes[i], nodes[j], (i + j) % 4 + 1)
print("\nGraph 4:")
print(g4)

def bellman_ford(graph, start_value):
    dist = {node for node in graph.nodes}
    dist[start_value]=0

    for i in range(len(graph.nodes)-1):
        for node in graph.nodes.values():
            for neighbor, weight in node.adjacent.items():
                if dist[node.value]+weight < dist[neighbor.value]:
                    dist[neighbor.value] = dist[node.value] + weight

    return dist

for graph in {g1,g2,g3,g4}:
    print(graph)
    print("Bellman-Ford: ", bellman_ford(graph, "A"))