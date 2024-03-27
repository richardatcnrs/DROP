import sys


def read_input():
    input_file = open(sys.argv[1], 'r')

    n = int(input_file.readline().strip())
    weight_matrix = [[None for i in range(n)] for y in range(n)]

    for i in range(n):
        weights = input_file.readline().split()
        for j in range(n):
            weight_matrix[i][j] = int(weights[j].strip())

    

    replaced_edges = eval(input_file.readline().strip())
    input_file.close()
    return n,weight_matrix,replaced_edges


n, weight_matrix, replaced_edges = read_input()
visited = [False] * n
visited[0] = True
total_cost = 0
path = [0]

root = 0

for i in range(1,n):
    current_neighbors = weight_matrix[root]
    #print(current_neighbors)

    #find min neighbor
    temp = {}
    for j in range(n):
        if visited[j] == False:
            temp[j] = current_neighbors[j]

    min_cost = min(temp.values())
    for key in temp.keys():
        if temp[key] == min_cost:
            min_neighbor = key
            break
    root = min_neighbor
    path.append(root)
    total_cost += min_cost
    visited[root] = True

    #min_neighbor = i+1
    #for j in range(len(current_neighbors)):
    #    if (visited[j] == False) and (current_neighbors[j] < current_neighbors[min_neighbor]):
    #        min_neighbor = j
            
    #current = min_neighbor
    #path.append(min_neighbor)
    #total_cost += current_neighbors[min_neighbor]
print(total_cost)
print(path)
