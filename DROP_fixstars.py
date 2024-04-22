#usage: python DROP_fixstars.py pre_processed_graph_weights_file 
from amplify import BinaryPoly, SymbolGenerator
from amplify import BinaryQuadraticModel
from amplify.client import FixstarsClient
from amplify import Solver
from amplify.constraint import equal_to
import networkx as nx
import sys
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

def decode_solution(solutions, n, replaced_edges):
    post_decode = (q.decode(solution.values))

    # verify the correctness of solution
    for x in range(1, n):
        t_sum = 0
        for t in range(1,n):
            t_sum += post_decode[x][t]
        if not(t_sum == 1):
            print("solution is not permutation")
           
        

    for t in range(1, n):
        x_sum = 0
        for x in range(1,n):
            x_sum += post_decode[x][t]
        if not(x_sum == 1):
            print("solution is not permutation")
    
    # decode path
    path = [0]
    for t in range(1, n):
        for x in range(1, n):
            if post_decode[x][t] == 1:
                path.append(x)
                break
   

    
    print('Optimal cost =', solution.energy)
    
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

# solver time in milli seconds
time = 5000
if '-t' in sys.argv:
    # convert input time to seconds
    time = int(sys.argv[sys.argv.index('-t')+1].strip())*1000

client = FixstarsClient()
token_file = open('add_path_to_fixstars_token_file','r')
token = token_file.readline()
token_file.close()

n, weight_matrix, replaced_edges = read_input()


# setting up the variables
# need to encode permutation of n-1
gen = SymbolGenerator(BinaryPoly)
q = gen.array(shape=(n,n))

f = BinaryPoly()

# penalty coefficients have to be set manually 
a = 2 * max(max(weight_matrix[i]) for i in range(n))
b = 2 * max(max(weight_matrix[i]) for i in range(n))

# add constraint using FS API
penalties = []
for t in range(1,n):
    temp = sum(q[v,t] for v in range(1,n))
    p = equal_to(temp,1)
    penalties.append(p)

for v in range(1,n):
    temp = sum(q[v,t] for t in range(1,n))
    p = equal_to(temp,1)
    penalties.append(p)


# encoding the first step in the route

for v in range(1,n):
    f += q[v,1] * weight_matrix[0][v]

for t in range(1, n-1):
    for v in range(1,n):
        for v_p in range(1,n):
            if not(v == v_p):
                f += weight_matrix[v][v_p]*q[v,t]*q[v_p,t+1]

# add edge to form cycle
if rtb:
    for v in range(1,n):
        f += weight_matrix[v][0] * q[v,n-1]

model = f

for p in penalties:
    temp = a*p
    model = model + temp


client = FixstarsClient()
client.token = token.strip()
client.parameters.timeout = time

solver = Solver(client)
solver.filter_solution = False

result = solver.solve(model)
print('execution time =',solver.execution_time)

for solution in result:
    decode_solution(solution, n, replaced_edges)

