from GraphDB import Graph, Node
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import random
import math
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

def generate_random_graph(num_nodes, min_weight=1, max_weight=10):

    g = Graph()
    nodes = [chr(65 + i) for i in range(min(num_nodes, 26))]
    for i in range(len(nodes)):
        for j in range(len(nodes)):
            if i != j and random.random() < 1/math.sqrt(num_nodes):
                weight = np.random.randint(min_weight, max_weight + 1)
                g.add_edge(nodes[i], nodes[j], weight)
    return g



def bellman_ford(graph, start_value):
    dist = {node: float('inf') for node in graph.nodes}
    pred = {node: None for node in graph.nodes}  # Track predecessors
    dist[start_value] = 0

    for i in range(len(graph.nodes)-1):
        for node in graph.nodes.values():
            for neighbor, weight in node.adjacent.items():
                if dist[node.value] + weight < dist[neighbor.value]:
                    dist[neighbor.value] = dist[node.value] + weight
                    pred[neighbor.value] = node.value  # Update predecessor

    return dist, pred

def dijkstra(graph, start_value):
    dist = {node: float('inf') for node in graph.nodes}
    pred = {node: None for node in graph.nodes}  # Track predecessors
    dist[start_value] = 0
    visited = set()

    while len(visited) < len(graph.nodes):
        min_node = None
        min_dist = float('inf')
        for node, d in dist.items():
            if node not in visited and d < min_dist:
                min_dist = d
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
                    pred[neighbor.value] = min_node  # Update predecessor
    
    return dist, pred

def get_path(pred, start, end):
    path = []
    current = end
    while current is not None:
        path.append(current)
        current = pred[current]
    return list(reversed(path))

def visualize_graph(graph, pred=None):
    G = nx.DiGraph()
    
    for node in graph.nodes:
        G.add_node(node)
    
    for node in graph.nodes.values():
        for neighbor, weight in node.adjacent.items():
             G.add_edge(node.value, neighbor.value, weight=weight)
    

    pos = nx.spring_layout(G) 
    
    plt.figure(figsize=(10, 8))
    
    nx.draw_networkx_edges(G, pos, edge_color='black', width=1, arrows=True, arrowstyle='-|>', arrowsize=15)
    
    path_edges = []
    if pred is not None:
        for node, prev in pred.items():
            if prev is not None:
                if G.has_edge(prev, node):
                    path_edges.append((prev, node))
        
        if path_edges:
            nx.draw_networkx_edges(G, pos, edgelist=path_edges, edge_color='green', width=2, arrows=True, arrowstyle='-|>', arrowsize=15)
    
    nx.draw_networkx_nodes(G, pos, node_color='lightblue', node_size=500)
    
    nx.draw_networkx_labels(G, pos)
    
    edge_labels = nx.get_edge_attributes(G, 'weight')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, label_pos=0.3)
    
    plt.title("Graph Visualization (Directed)")
    plt.axis('off')
    plt.show()

for graph in {g1, g2, g3, g4}:
    print(graph)
    dist_dijk, pred_dijk = dijkstra(graph, "A")
    dist_bell, pred_bell = bellman_ford(graph, "A")


    if dist_dijk == dist_bell and pred_dijk == pred_bell:
        print("Both algorithms produced identical results")
        print("Distances:", dist_dijk)
        print("Predecessors:", pred_dijk)

    else:
        print("Results differ between algorithms")

        print("Dijkstra distances:", dist_dijk)
        print("Dijkstra predecessors:", pred_dijk)

        print("Bellman-Ford distances:", dist_bell)
        print("Bellman-Ford predecessors:", pred_bell)

    
 



print("\nVisualizing graphs with shortest paths...")
for graph in {g1, g2, g3, g4}:
    print(f"\nVisualizing {graph}")
    dist_dijk, pred_dijk = dijkstra(graph, "A")
    visualize_graph(graph, pred_dijk)