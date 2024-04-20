import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from itertools import combinations
from itertools import permutations
import pickle

G = nx.DiGraph()


number_of_nodes = 30

pos = {f"key{i}": (np.random.uniform(-1200,1200), np.random.uniform(-1500,1500)) for i in range(0, number_of_nodes)}
attributes = []
for p in pos:
    attributes.append((p,{"posX": pos[p][0], "posY": pos[p][1]}))


G.add_nodes_from(attributes)


#keys = [item[0] for item in attributes]
#combinations_list = list(combinations(keys, 2))


keys = [item[0] for item in attributes]
combinations_list = list(permutations(keys, 2))

G.add_edges_from(combinations_list, opacity=0.1)


#nx.draw(G,pos=pos, node_size = 5,alpha=0.1, arrows= False)
def ccw(A,B,C):
    return (C[1]-A[1]) * (B[0]-A[0]) > (B[1]-A[1]) * (C[0]-A[0])

# Return true if line segments AB and CD intersect
def intersect(A,B,C,D):
    return ccw(A,C,D) != ccw(B,C,D) and ccw(A,B,C) != ccw(A,B,D)

A = [[900, 900],[450, 400],[1100, 300],[1400, 1300],[600, 1550]]
A = [[x - 1000 for x in pair] for pair in A]
B = [[900, 1100],[450, 600],[1100, 700],[1400, 1700],[600, 1650]]
B = [[x - 1000 for x in pair] for pair in B]
C = [[1100, 1100],[950, 600],[1300, 700],[1600, 1700],[800, 1650]]
C = [[x - 1000 for x in pair] for pair in C]
D = [[1100, 900],[950, 400],[1300, 300],[1600, 1300],[800, 1550]]
D = [[x - 1000 for x in pair] for pair in D]



for u, v in list(G.edges()):
    point_U = [G.nodes[u]['posX'], G.nodes[u]['posY']]
    point_V = [G.nodes[v]['posX'], G.nodes[v]['posY']]
    
    for i in range(len(A)):
        if(intersect(point_U, point_V, A[i], B[i]) | intersect(point_U, point_V, B[i],C[i] ) | intersect(point_U, point_V, C[i],D[i] ) | intersect(point_U, point_V, D[i],A[i] )) :
            try:
                G.remove_edge(u,v)
            except:
                pass


nx.draw(G,pos=pos, node_size = 5,alpha=0.1, arrows = False)

plt.gca().add_patch(Rectangle((-200/2, -200/2), 200, 200, fill='blue', color='blue'))
plt.gca().add_patch(Rectangle((-300-250, -500-100), 500, 200, fill='blue', color='blue'))
plt.gca().add_patch(Rectangle((200-100, -500-200), 200, 400, fill='blue', color='blue'))
plt.gca().add_patch(Rectangle((500-100, 500-200), 200, 400, fill='blue', color='blue'))
plt.gca().add_patch(Rectangle((-300-100, 600-50), 200, 100, fill='blue', color='blue'))


# richard edit
#plt.show()

import pyvista as pv

points = []

for p in pos:
    points.append([pos[p][0], pos[p][1], 300])


mesh = pv.read('./data.vtu')
mesh_of_probing_points = pv.PolyData(points)
probe_results = mesh_of_probing_points.sample(mesh)



probe_results["U"][:,0:2].shape

UX = probe_results["U"][:,0].tolist()
UY = probe_results["U"][:,1].tolist()

for i, node in enumerate(G.nodes()):
    G.nodes[node]['UX'] = UX[i]
    G.nodes[node]['UY'] = UY[i]





plt.figure(figsize=(10, 10))

for key in G.nodes:
    plt.arrow(G.nodes[key]['posX'], G.nodes[key]['posY'], 5 * G.nodes[key]['UX'], 5 * G.nodes[key]['UY'], 
              head_width=10, head_length=10, fc='r', ec='r')

plt.gca().add_patch(Rectangle((-200/2, -200/2), 200, 200, fill='blue', color='blue'))
plt.gca().add_patch(Rectangle((-300-250, -500-100), 500, 200, fill='blue', color='blue'))
plt.gca().add_patch(Rectangle((200-100, -500-200), 200, 400, fill='blue', color='blue'))
plt.gca().add_patch(Rectangle((500-100, 500-200), 200, 400, fill='blue', color='blue'))
plt.gca().add_patch(Rectangle((-300-100, 600-50), 200, 100, fill='blue', color='blue'))


plt.xlim(-1000, 1000)
plt.ylim(-1000, 1000)
plt.xlabel('X')
plt.ylabel('Y')
#plt.title('Arrows representing velocity vectors')
plt.grid(True)
plt.gca().set_aspect('equal', adjustable='box')

#richard edit
#plt.show()


connected_components = list(nx.strongly_connected_components(G))

# Print the number of connected components
#print("Number of disconnected components:", len(connected_components))

# Print the nodes in each connected component
biggest_indice = 0

for i, component in enumerate(connected_components):
    max_elements = 0

    #print(f"Component {i+1}: {len(component)} keys")
    if len(component) > max_elements:
        max_element = len(component)
        biggest_indice = i


#print(f"Keeping only component {i}")

for i in range(1,len(connected_components)):
    G.remove_nodes_from(connected_components[i])






#theta = 0
#amplitude = 1
#far_field = [amplitude*np.cos(theta), amplitude*np.sin(theta)]

def cost_function(u,v):
    position_X_U = G.nodes[u]['posX']
    position_Y_U = G.nodes[u]['posY']
    position_X_V = G.nodes[v]['posX']
    position_Y_V = G.nodes[v]['posY']

    vel_X_U = G.nodes[u]['UX']
    vel_Y_U = G.nodes[u]['UY']
    vel_X_V = G.nodes[v]['UX']
    vel_Y_V = G.nodes[v]['UY']

    mean_vel = [(vel_X_U+vel_X_V)/2, (vel_Y_U+vel_Y_V)/2]

    distance = np.sqrt((position_X_U-position_X_V)**2 + (position_Y_U-position_Y_V)**2)

    #direction_u_to_v = [position_X_U-position_X_V, position_Y_U-position_Y_V]
    #direction_v_to_u = [position_X_V-position_X_U, position_Y_V-position_Y_U] 
    direction = [position_X_U-position_X_V, position_Y_U-position_Y_V]

    #direction_u_to_v_norm = direction_u_to_v / np.linalg.norm(direction_u_to_v)
    #direction_v_to_u_norm = direction_v_to_u / np.linalg.norm(direction_v_to_u)
    direction_norm = direction / np.linalg.norm(direction)


    #projected_wind_velocity_u_to_v = np.dot(mean_vel, direction_u_to_v_norm)
    #projected_wind_velocity_v_to_u = np.dot(mean_vel, direction_v_to_u_norm)
    projected_wind_velocity = np.dot(mean_vel, direction_norm)

    # return distance * projected_wind_velocity
    # return distance*projected_wind_velocity_u_to_v, distance*projected_wind_velocity_v_to_u
    return distance*projected_wind_velocity



#edge_values_u_to_v = {}
#edge_values_v_to_u = {}
edge_values = {}

for u, v in G.edges():
    #edge_values_u_to_v[(u, v)], edge_values_v_to_u[(u, v)] = cost_function(u, v)
    edge_values[(u, v)]  = cost_function(u, v)

#Normalize
max_value = max(edge_values.values())
min_value = min(edge_values.values())

normalized_data = {}
for key, value in edge_values.items():
    normalized_value = (value - min_value) / (max_value - min_value)
    normalized_data[key] = normalized_value
    
nx.set_edge_attributes(G, normalized_data, 'cost function')


import matplotlib as mpl
from  matplotlib.colors import LinearSegmentedColormap
cmap=LinearSegmentedColormap.from_list('gr',["r", "w", "g"], N=256) 

edges,weights = zip(*nx.get_edge_attributes(G,'cost function').items())
plt.figure(figsize=(10, 10))

nx.draw(G, pos, node_size = 10, edgelist=edges, edge_color=weights, width=1.0, edge_cmap=cmap, arrows = True, with_labels=True)

plt.gca().add_patch(Rectangle((-200/2, -200/2), 200, 200, fill='blue', color='blue'))
plt.gca().add_patch(Rectangle((-300-250, -500-100), 500, 200, fill='blue', color='blue'))
plt.gca().add_patch(Rectangle((200-100, -500-200), 200, 400, fill='blue', color='blue'))
plt.gca().add_patch(Rectangle((500-100, 500-200), 200, 400, fill='blue', color='blue'))
plt.gca().add_patch(Rectangle((-300-100, 600-50), 200, 100, fill='blue', color='blue'))

pickle.dump(G, open('graph_unstructured.pickle', 'wb'))

#print(G.get_edge_data('key9','key13'))
#print(G.get_edge_data('key13','key9'))




print(G.order())
for u in G.nodes():
    for v in G.nodes():
        if not(G.get_edge_data(u,v) == None):
            cost = G.get_edge_data(u,v)['cost function']
            print(int(round(cost*1000)), end=' ')
        else:
            print(0,end=' ')
    print()

#richard edit
plt.show()
