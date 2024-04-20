#usage: python pre_process.py < weighted_adjacency_matrix
import sys
import networkx as nx
G = nx.DiGraph()

adj_file = open(sys.argv[1], 'r')
n = int(adj_file.readline().strip())
edges = []

for i in range(n):
    line = adj_file.readline().split()
    for j in range(n):
        weight = int(line[j])
        if weight == 0:
            edges.append((i,j,{'weight':float('inf')}))
        else:
            edges.append((i,j,{'weight':weight}))


G.add_nodes_from(range(n))
G.add_edges_from(edges)
#print(G.edges())

predecessors, distance = nx.floyd_warshall_predecessor_and_distance(G)
#print('distance fdjal;jl', distance)

#print(nx.reconstruct_path(1,2, predecessors))

replaced_edges = {}

G_p = nx.DiGraph()

new_edges = []

#print(G.get_edge_data(1,2)['weight'])

for i in range(n):
    for j in range(n):
        if not i == j:
            if distance[i][j] < G.get_edge_data(i,j)['weight']:
                new_edges.append((i,j,{'weight':distance[i][j]}))
                replaced_edges[i,j] = nx.reconstruct_path(i,j,predecessors)
            else:
                new_edges.append((i,j,{'weight':G.get_edge_data(i,j)['weight']}))
#print(new_edges)
G_p.add_edges_from(new_edges)


print(n)
for i in range(n):
    for j in range(n):
        if i == j:
            print(0, end=' ')
        else:
            print(G_p.get_edge_data(i,j)['weight'], end=' ')
    print()
print(replaced_edges)
