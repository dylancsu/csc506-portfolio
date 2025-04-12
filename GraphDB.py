class Node:
    def __init__(self, value):
        self.value = value
        self.adjacent = {}  # neighbor -> weight

    def add_edge(self, node, weight):
        self.adjacent[node] = weight

    def __repr__(self):
        return f"Node({self.value})"

class Graph:
    def __init__(self):
        self.nodes = {}

    def add_node(self, value):
        if value not in self.nodes:
            self.nodes[value] = Node(value)
        return self.nodes[value]

    def add_edge(self, from_value, to_value, weight):
        from_node = self.add_node(from_value)
        to_node = self.add_node(to_value)
        from_node.add_edge(to_node, weight)

    def __repr__(self):
        result = ""
        for node in self.nodes.values():
            neighbors = ', '.join(f"{n.value}({w})" for n, w in node.adjacent.items())
            result += f"{node.value}: [{neighbors}]\n"
        return result.strip()
