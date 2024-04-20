#usage: python DROP_dwave.py < pre_processed_graph_weights_file 
import dimod
import dwave_networkx as dnx
import networkx as nx
import dwave.embedding
from dwave.system import LeapHybridCQMSampler 
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
    #print(solution.energy, solution.values)
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

def build_model(n, weight_matrix, replaced_edges):
    model = dimod.ConstrainedQuadraticModel()
   
    model.add_variables('BINARY', ['x_' + str(i) + '_' + str(j) for i in range(1,n) for j in range(1,n)])
    
    # set up constraints
    for i in range(1, n):
        terms = []
        for j in range(1,n):
            terms.append(['x_' + str(i) + '_' + str(j),1])

        model.add_constraint_from_iterable(terms, '==', rhs = 1)
                
    for j in range(1, n):
        terms = []
        for i in range(1,n):
            terms.append(['x_' + str(i) + '_' + str(j),1])

        model.add_constraint_from_iterable(terms, '==', rhs = 1)

    obj = []
    
    # encode the 1st step
    for v in range(1,n):
        obj.append(['x_' + str(v) + '_1', weight_matrix[0][v]])

    # encode the hamiltonian path
    for t in range(1, n-1):
        for v in range(1,n):
            for v_p in range(1,n):
                if not(v == v_p):
                    obj.append(['x_' + str(v) + '_' + str(t), 'x_' + str(v_p) + '_' + str(t+1), weight_matrix[v][v_p]])
    
    # adding edge to form a cycle
    if rtb:
        for v in range(1,n):
            obj.append(['x_' + str(v) + '_' + str(n-1), weight_matrix[v][0]])
    model.set_objective(obj)

    return model

def run_cqm_and_collect_solutions(model, sampler):
    sampleset = sampler.sample_cqm(model, time)
    return sampleset

def process_solutions(sampleset, n, weight_matrix, replaced_edges):
    perm_solutions = []
    for solution in sampleset:
        if check_perm(solution, n) == True:
            perm_solutions.append(solution)
    if len(perm_solutions) == 0:
        print('No valid solution')
        return 0,0
    elif len(perm_solutions) == 1:
        return compute_energy(perm_solutions[0], weight_matrix)
    else:
        min_solution = perm_solutions[0]
        min_path, min_energy = compute_energy(min_solution, weight_matrix)
    
        for i in range(1,len(perm_solutions)):
            current_path, current_energy = compute_energy(perm_solutions[i], weight_matrix)
            if current_energy < min_energy:
                min_energy = current_energy
                min_path = current_path
        
        # replace edges if needed
        if len(replaced_edges) > 0:
            converted_path = [0]
            for i in range(len(min_path)-1):
                if (min_path[i], min_path[i+1]) in replaced_edges:
                    converted_path.extend(replaced_edges[min_path[i],min_path[i+1]])
                
               
                else:
                    converted_path.append(min_path[i]) 
                    converted_path.append(min_path[i+1])
            min_path = [i[0] for i in groupby(converted_path)]
        return min_path, min_energy 

def compute_energy(solution, weight_matrix):
    cost = 0
    path = [0]
    for i in range(1,n):
        if solution['x_{}_1'.format(i)] == 1:
            path.append(i)
            cost += weight_matrix[0][i]
            break
    for t in range(2, n):
        for i in range(1,n):
            if solution['x_{}_{}'.format(i,t)] == 1:
                path.append(i)
                break
    for i in range(1, n-1):
        cost += weight_matrix[path[i]][path[i+1]]
    
    # add cost of the last edge in cycle
    if rtb:
        cost += weight_matrix[path[-1]][0]
    return path, cost

def check_perm(solution, n):
    for i in range(1, n):
        if not(sum(solution['x_{}_'.format(i) + str(j)] for j in range(1,n)) == 1):
            return False

    for j in range(1, n):
        if not(sum(solution['x_{}_'.format(i) + str(j)] for i in range(1,n)) == 1):
            return False

    return True

# option for cycle instead of path
rtb = False
if '-rtb' in sys.argv:
    rtb = True

# set time limit
time = 5
if '-t' in sys.argv:
    time = int(sys.argv[sys.argv.index('-t')+1].strip())


token_file = open('add_path_to_dwave_token','r')
token = token_file.readline()
token_file.close()
n, weight_matrix, replaced_edges = read_input()
model = build_model(n, weight_matrix, replaced_edges)
sampler = LeapHybridCQMSampler()


sampleset = run_cqm_and_collect_solutions(model, sampler)
min_path, min_energy = process_solutions(sampleset, n, weight_matrix, replaced_edges)
print('Optimal cost =', min_energy)
print('Optimal path =', min_path)





