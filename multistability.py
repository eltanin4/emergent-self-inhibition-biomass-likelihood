import time
import ast
import pickle
import itertools
import statistics

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib import rcParams
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import matplotlib.patches as patches
from matplotlib.colors import (
    ListedColormap,
    BoundaryNorm,
    LinearSegmentedColormap,
    LogNorm,
)
from matplotlib.ticker import LogLocator, FuncFormatter
from matplotlib.ticker import FixedLocator, FixedFormatter, NullFormatter

from scipy.integrate import solve_ivp, quad
from scipy.linalg import eigh
from scipy.stats import linregress, norm as stats_norm
from scipy.optimize import curve_fit




#Generalized Lotka-Volterra equations

def Integrate(A_int, r_int, y0, maxtime=50):
    def ode(t, y):
        dydt = np.dot(np.diag(y), (r - np.dot(A, y)))  # GLV
        return dydt
    
    A = A_int.copy()
    r = r_int.copy()
    
    t_span = (0, maxtime)
    solution = solve_ivp(ode, t_span, y0, method='Radau')  # radau
    
    return solution
    
    
#plot all the solution
def multi_plot(solution):
    t = solution.t
    y = solution.y
    plt.figure(figsize=(8, 6))
    for i in range(len(y)):
        plt.plot(t, y[i], label=f'sp_{i}')
    #plt.legend()
    plt.xlabel('time')
    plt.ylabel('species abundance')
    plt.show()
    
    
#def log plot
def multi_log_plot(solution):
    t = solution.t
    y = solution.y
    plt.figure(figsize=(8, 6))
    for i in range(len(y)):
        plt.yscale('log')
        plt.plot(t, y[i], label=f'sp_{i}')
    #plt.legend()
    plt.show()


def show_final_compo(file):
    for i in range (len(file)):
        if file[i] != 0:
            print(f"sp[{i}] = {file[i]}")
    return


def get_final_composition(data, threshold, A, r):
    final = np.copy(data.y[:, -1])
    final = np.where(final < threshold, 0, final)
    surv_ind = np.nonzero(final > 0)[0]
    A_r, r_r, _ = reduced(final, A, r)
    analytic_final = cal_analytic_final_compo(A_r, r_r)
    final[surv_ind] = analytic_final
    return final


def get_binary_compo(data):
    if np.any(data < 0):
        print("Unfeasible composition: Negative value found.")
        return None

    binary_data = (np.array(data) > 0).astype(int)
    return binary_data.tolist()


def reduced(data, A, r):
    
    mask = data > 0
    mask_indices = np.where(mask)[0]
    # print(mask_indices)
    A_pruned = A[mask_indices][:, mask_indices].copy()
    r_pruned = r[mask_indices].copy()
    final_compo_pruned = np.array(data)[mask_indices].copy()
    # print(A_pruned[-1])
    # print(mask_indices)
    return A_pruned, r_pruned, final_compo_pruned


# If all the species in the related to interaction matrix survives then we can find the final composition analytically

def cal_analytic_final_compo(A, r):
    try:
        ans = np.linalg.solve(A, r)
        return ans
    except np.linalg.LinAlgError:
        # singular A
        print(A[-1])
        file_path = 'problem.pkl'
        with open(file_path, 'wb') as file:
            pickle.dump(A, file)
        print("Matrix A is singular. Unable to compute the solution.")
        return None    

# if our reduced interaction matrix is correct then when we find the final composition analytically then all the species composition should be negative.So, let's create a function to check if our final vector is feasible or not


def all_positive(vector):
    return np.all(np.array(vector) > 0)

def all_feasible(data):
    vector = data.y[:,-1].copy()
    return np.all(np.array(vector) >= -1e-5)


# Functions to get maximum and minimum eigen value from the matrix

def max_eigenvalue(matrix):
    eigenvalues = np.linalg.eigvals(matrix)
    max_eigenvalue = np.max(np.real(eigenvalues))  
    return max_eigenvalue


def min_eigenvalue(matrix):
    eigenvalues = np.linalg.eigvals(matrix)
    if len(np.real(eigenvalues)) != 0:
        min_eigenvalue = np.min(np.real(eigenvalues))
    else:
        min_eigenvalue = -100
    return min_eigenvalue


# stability from the jacobian J = -diag(N*) A_r at the fixed point

def is_stable(A_r, N_r):
    J = -np.diag(N_r) @ A_r
    return max_eigenvalue(J) < 0


# If you have a list of final composition and you want the survived specoes after the threshold

def get_final_composition_thres(charlie, threshold):
    charlie_new = [element if element> threshold else 0 for element in charlie]
    return charlie_new

# to check which species have survived 

def give_survive_index(listu):
    nonzero_positions = [i for i, value in enumerate(listu) if value != 0]
    return nonzero_positions


def give_first_surviving_index(list):
    for i,element in enumerate(list):
        if element > 0:
            name = i
            break
    return name


def it_works(a):
    a += a
    return a

# # to check if the state is fixed and stable

def check_fixed_point(final_compo, A, r):
    A_r, r_r, _ = reduced(final_compo, A, r)
    analytic_final = cal_analytic_final_compo(A_r, r_r)
    array1 = np.array(final_compo)[np.array(final_compo) > 0]
    
    check = np.all(analytic_final > 0) and np.all(np.isclose(array1, analytic_final, atol=0.1))
    
    
    return check
    
    
# # to check if the state is uninvasible
def check_uninvasibility(final_compo, A, r):
    def uninvadable_matrix(data, l):
        mask = data > 0
        A_pruned = A[l, mask].copy()
        return A_pruned

    uninvasible = True
    surv_ind = np.nonzero(final_compo > 0)[0]
    A_r, r_r, _ = reduced(final_compo, A, r)
    analytic_final = cal_analytic_final_compo(A_r, r_r)
    # surv_comp = final_compo[surv_ind]
    surv_comp = analytic_final

    check_un = np.setdiff1d(np.arange(len(A)), surv_ind)

    for element in check_un:
        if np.dot(uninvadable_matrix(final_compo, element), surv_comp) < r[element]:
            uninvasible = False
            break

    return uninvasible
    
# # to give the composition of the survived species

def give_survive_composition(listu):
    listt = [element for element in listu if element > 0]
    return listt

# # to check the mean and variance of the elements of the interaction matrix

def get_mean__variance(A):
    upper_triangle_elements = []
    for i in range(len(A)):
        for j in range(i + 1, len(A)):  
            upper_triangle_elements.append(A[i][j])
    mean_value = statistics.mean(upper_triangle_elements)
    variance_value = statistics.variance(upper_triangle_elements) 
    print(f"Mean: {mean_value}")
    print(f"Variance: {variance_value}") 

# # to create a new data arrays

def once(raw, unique, A, r):
    continuee = True
    max_iterations = 100  # max tries
    iteration_count = 0
    np.random.seed(2244)
    while continuee:
        iteration_count += 1
        
        y0 = np.abs(np.random.uniform(0, 1, len(A)))
        data = Integrate(A, r, y0, maxtime=100000)
        
        if all_feasible(data):
            for element in np.logspace(-5, -1, 5):
                final_compo = get_final_composition(data, element,A,r)
              
                if (np.all(final_compo>=0)):
                    if check_fixed_point(final_compo, A, r):
                        A_r, r_r, N_r = reduced(final_compo, A, r)
                        
                        if is_stable(A_r, N_r):
                            if check_uninvasibility(final_compo, A, r):
                                continuee = False
                                new_data = np.hstack([
                                    [0], y0, final_compo, get_binary_compo(final_compo),
                                    [element], [1000]
                                ]).reshape((1, -1))
                                
                                with open(unique, 'wb') as file:
                                    pickle.dump(new_data, file)
    
                                new_data_r = np.hstack([
                                    [0], y0, final_compo, [element], [1000]
                                ]).reshape((1, -1))
    
                                with open(raw, 'wb') as file:
                                    pickle.dump(new_data_r, file)
    
                                print('created')
                                break
        
        # stop after max_iterations
        if iteration_count >= max_iterations:
            print("Error: Maximum iterations reached without finding a solution. Exiting.",iteration_count)
            raise RuntimeError("Maximum iterations reached without finding a solution.")


 # # to start the simulation and find all the states

def start_sim(size,repeat,matrix,r_file,raw_data,raw_unique_data,r_seed):
    print('started')
    start_time = time.time()
    try:
        with open(matrix, 'rb') as file:
            A = pickle.load(file)
    except FileNotFoundError:
        print("Matrix is not available.")   # diagonal of A is r_i/K_i
        return

    try:
        with open(r_file, 'rb') as file:
            r = pickle.load(file)            # growth rates
    except FileNotFoundError:
        print("r is not available.")
        return

    try:
        with open(raw_data, 'rb') as file:
            raw = pickle.load(file)
            raw=raw[-1:]
    except FileNotFoundError:
        once(raw_data,raw_unique_data,A,r)
        with open(raw_data, 'rb') as file:
            raw = pickle.load(file)
        
    with open(raw_unique_data, 'rb') as file:
        raw_uniue = pickle.load(file)
    np.random.seed(r_seed)
    i = raw_uniue[:,0][-1]
    #y0 = np.random.uniform(0,1,len(A))
    n_done = 0      # initial conditions that reached a state
    while n_done < repeat:
        timee = 300
        tom = 5000
        y0 = np.abs(np.random.uniform(0,1,len(A))) #randomly generate the initial condition
    
        data = Integrate(A,r,y0,maxtime=timee )
        got_it = False
        recorded = False
        for h in range(10):
            if not(all_feasible(data)):
                #print('found')
                break
            for element in np.logspace(-5,-1,5):
                Check_point = True
                break_outer = False
                final_compo = get_final_composition(data,element,A,r)
                if (np.all(final_compo>=0)):
                    # print(np.all(final_compo)>0)
                    # print(final_compo.tolist())
                    for p in range(raw_uniue.shape[0]):
                        binary = get_binary_compo(final_compo)
                        #print(p)
                        if all(binary == (raw_uniue[p][(2*len(A))+1:(3*len(A))+1])) and np.all(np.isclose(final_compo, raw_uniue[p][(len(A))+1:(2*len(A))+1], atol=0.1)):
                            Check_point = False
                            break_outer = True
                            recorded = True
                            #print('copied')
                            new_data = np.hstack([([p]),y0,final_compo,([element]),([timee])])
                            raw = np.vstack([raw,new_data])
                            break
                if break_outer:
                        break
            if not(break_outer):
                for element in np.logspace(-24,-1,24):
                    final_compo = get_final_composition(data,element,A,r)
                    if (np.all(final_compo>=0)):
                        if check_fixed_point(final_compo,A,r):
                            A_r,r_r,N_r= reduced(final_compo,A,r)
                            if is_stable(A_r, N_r):
                                if check_uninvasibility(final_compo,A,r):
                                    #print('found')
                                    i += 1
                                    new_data = np.hstack([([i]),y0,final_compo,get_binary_compo(final_compo),([element]),([timee])])
                                    raw_uniue = np.vstack([raw_uniue,new_data])
                                    new_data = np.hstack([([i]),y0,final_compo,([element]),([timee])])
                                    #print('copied')
                                    raw = np.vstack([raw,new_data])
                                    #print(raw,raw_uniue,'gotit')
                                    Check_point = False
                                    recorded = True
                                    break
            if recorded:
                break
            #tom += 100
            timee += tom
            data = Integrate(A,r,data.y[:,-1],maxtime=tom )
        if recorded:
            n_done += 1
            if n_done % 1000 == 0:
                file_path = raw_data
                with open(file_path, 'rb') as file:
                    existing_raw = pickle.load(file)
                combined_data = np.vstack([existing_raw, raw[:-1]])
                with open(file_path, 'wb') as file:
                    pickle.dump(combined_data, file)
                del combined_data
                raw = raw[-1:]
            
            
    file_path = raw_data
    with open(file_path, 'rb') as file:
        existing_raw = pickle.load(file)
    combined_data = np.vstack([existing_raw, raw[:-1]])
    with open(file_path, 'wb') as file:
        pickle.dump(combined_data, file)
    del combined_data
    file_path = raw_unique_data
    with open(file_path, 'wb') as file:
        pickle.dump(raw_uniue, file)
    end_time = time.time()
    elapsed_time = end_time-start_time
    print(f"Elapsed time: {elapsed_time} seconds")
