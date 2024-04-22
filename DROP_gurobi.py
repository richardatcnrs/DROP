#usage: python DROP_gurobi.py pre_processed_graph_weights_file 
import sys
import gurobipy as gb
from itertools import groupby


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

def decode_solution(variables, n, replaced_edges, obj_val):
    post_decode = {}
    
    for var in variables:
        temp = var.varName.split('_')
        post_decode[int(temp[1]), int(temp[2])] = var.x

    
    # verify the correctness of solution
    for x in range(1, n):
        t_sum = 0
        for t in range(1,n):
            t_sum += post_decode[x,t]
        if not(t_sum == 1):
            print("solution is not permutation")
           
    
    for t in range(1, n):
        x_sum = 0
        for x in range(1,n):
            x_sum += post_decode[x,t]
        if not(x_sum == 1):
            print("solution is not permutation")
    
    # decode path
    path = [0]
    for t in range(1, n):
        for x in range(1, n):
            if post_decode[x,t] == 1:
                path.append(x)
                break
   

    
    print('Optimal cost =', obj_val)
    
    # convert replaced edges
    
    if len(replaced_edges) > 0:
        converted_path = [0]
        for i in range(len(path)-1):
            if (path[i], path[i+1]) in replaced_edges:
                converted_path.extend(replaced_edges[path[i],path[i+1]])
                
               
            else:
                converted_path.append(path[i]) 
                converted_path.append(path[i+1])
        result = [i[0] for i in groupby(converted_path)]
        print("Optimal path =", result)
              
    else:
        print('Optimal path =', path)


rtb = False
if '-rtb' in sys.argv:
    rtb = True

time = 5
if '-t' in sys.argv:
    time = int(sys.argv[sys.argv.index('-t')+1].strip())

n, weight_matrix, replaced_edges = read_input()

model = gb.Model('quadratic')

vars_map = {}
for i in range(1,n):
    for j in range(1,n):
        var_name = "x_" + str(i) + '_' + str(j)

        vars_map[i,j] = model.addVar(vtype=gb.GRB.BINARY,name=var_name)


# set up the constraints
for v in range(1,n):
    model.addConstr(sum(vars_map[v,t] for t in range(1,n)) == 1)
    model.addConstr(sum(vars_map[t,v] for t in range(1,n)) == 1)

# encode the first step in the route
obj_fn = sum(vars_map[v,1] * weight_matrix[0][v] for v in range(1,n))

# encode the rest of the route
for t in range(1, n-1):
    for v in range(1,n):
        for v_p in range(1,n):
            if not(v == v_p):
                obj_fn += weight_matrix[v][v_p]*vars_map[v,t]*vars_map[v_p,t+1]

# add edge to form cycle

if rtb:
    for v in range(1,n):
        obj_fn += weight_matrix[v][0]*vars_map[v,n-1]


model.setObjective(obj_fn, gb.GRB.MINIMIZE)
model.setParam("OutputFlag", False)
model.setParam("TimeLimit", time)
model.optimize()
decode_solution(model.getVars(), n, replaced_edges, model.objVal)

