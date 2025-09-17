import networkx as nx

g = nx.Graph()

g.add_node("A")
g.add_node("B")
g.add_node("C")

g.nodes["A"]["Balls"] = True

print(g.nodes["A"])