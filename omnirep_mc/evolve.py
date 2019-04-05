#
# OMNIREP, copyright 2018 moshe sipper, www.moshesipper.com
#

# main evolutionary module

# stuff related to encodings marked with 'enc'
# stuff related to representations marked with 'rep'

from random import randint, choices, seed
from json import loads, dump
from sys import maxsize
from string import ascii_uppercase
from pathlib import Path # correctly use folders -- '/' or '\' -- on both Windows and Linux 
from copy import deepcopy

#import bitstring_and_bitcount as modname
#import float_and_precision_2 as modname
#import program_and_instructions as modname
import image_and_polygons as modname
#import image_and_blocks as modname
#import image_and_blocks_1 as modname
#import image_and_circles_1 as modname


WORST_FITNESS = maxsize # (best fitness is 0)
DATA_FILE = "" # if empty ("") then generate data, otherwise use the given file
ENC_EVOLVE = DATA_FILE == "" # evolve encoding population or not (the latter means standard evolution with fixed representation)
RESULTS_FOLDER = "results/"

def selection(population, fitnesses): # select one individual using tournament selection
    tournament = [randint(0, len(population)-1) for i in range(modname.TOURNAMENT_SIZE)] # select tournament contenders
    tournament_fitnesses = [fitnesses[tournament[i]] for i in range(modname.TOURNAMENT_SIZE)]
    return deepcopy(population[tournament[tournament_fitnesses.index(min(tournament_fitnesses))]]) 

def run_evolutionary_alg(run_num, data_dict):  
    rep_population = modname.rep_init_population(data_dict) 
    enc_population = modname.enc_init_population(ENC_EVOLVE, data_dict) 

    if not ENC_EVOLVE:
        enc_best = deepcopy(enc_population[0])    
    
    enc_fitnesses = [WORST_FITNESS] * modname.ENC_POP_SIZE
    rep_fitnesses = [WORST_FITNESS] * modname.REP_POP_SIZE
    enc_top = [enc_population[i] for i in range(modname.TOP_COUNT)] # arbitrarily initialize best to first TOP_COUNT individuals
    rep_top = [rep_population[i] for i in range(modname.TOP_COUNT)]
    f_best = WORST_FITNESS # init variable that holds best-of-run fitness
    f_best_str = ""

    # begin evolution
    for gen in range(modname.GENERATIONS): 
        best_change = False # has there been a fitness improvement?                
        
        for i in range(modname.REP_POP_SIZE): #compute fitnesses for representation population  
            rep_ind = deepcopy(rep_population[i])
            sum_f = 0
            for j in range(modname.TOP_COUNT):
                enc_ind = deepcopy(enc_top[j])
                f = modname.fitness(enc_ind, rep_ind, data_dict)
                if (f<f_best):
                    best_change = True 
                    rep_best = deepcopy(rep_ind)
                    enc_best = deepcopy(enc_ind)
                    f_best = f
                sum_f += f
            rep_fitnesses[i] = sum_f / modname.TOP_COUNT   

        rep_sorted_fitnesses = sorted(rep_fitnesses)
        for i in range(modname.TOP_COUNT): # fill TOP_COUNT individuals
            rep_top[i] = deepcopy(rep_population[rep_fitnesses.index(rep_sorted_fitnesses[i])])  
#        rep_best = deepcopy(rep_top[0])
        
        nextgen_rep_population = []
        nextgen_rep_population.append(deepcopy(rep_top[0])) # elitism:   
        nextgen_rep_population.append(deepcopy(rep_top[1])) # copy top 2 into next gen

        for i in range(int(modname.REP_POP_SIZE/2)-1): #select-cross-mutate representation population, note: already 2 elite in pop
            parent1 = selection(rep_population, rep_fitnesses)
            parent2 = selection(rep_population, rep_fitnesses)
            child1, child2 = modname.rep_crossover(parent1, parent2)
            nextgen_rep_population.append(modname.rep_mutation(child1,data_dict))
            nextgen_rep_population.append(modname.rep_mutation(child2,data_dict))
                    
        rep_population = nextgen_rep_population
       
        if ENC_EVOLVE: # evolve encodings population (set to False when doing fixed representation)           
            if (gen % modname.ENC_GAP == 0): # evolve encodings every ENC_GAP generations               
                
                for i in range(modname.ENC_POP_SIZE): #compute fitnesses for encoding population
                    enc_ind = deepcopy(enc_population[i])
                    sum_f = 0
                    for j in range(modname.TOP_COUNT):
                        rep_ind = deepcopy(rep_top[j])
                        f = modname.fitness(enc_ind, rep_ind, data_dict)
                        if (f<f_best):
                            best_change = True
                            rep_best = deepcopy(rep_ind)
                            enc_best = deepcopy(enc_ind)
                            f_best = f
                        sum_f += f
                    enc_fitnesses[i] = sum_f / modname.TOP_COUNT
                
                enc_sorted_fitnesses = sorted(enc_fitnesses)
                for i in range(modname.TOP_COUNT): # fill TOP_COUNT individuals
                    enc_top[i] = deepcopy(enc_population[enc_fitnesses.index(enc_sorted_fitnesses[i])]) 
#                enc_best = deepcopy(enc_top[0])
                
                nextgen_enc_population = []
                nextgen_enc_population.append(deepcopy(enc_top[0])) # elitism:
                nextgen_enc_population.append(deepcopy(enc_top[1])) # copy top 2 into next gen
                
                for i in range(int(modname.ENC_POP_SIZE/2)-1): # select-cross-mutate encoding population, note: already 2 elite in pop
                    parent1 = selection(enc_population, enc_fitnesses)
                    parent2 = selection(enc_population, enc_fitnesses)
                    child1, child2 = modname.enc_crossover(parent1, parent2)
                    nextgen_enc_population.append(modname.enc_mutation(child1))
                    nextgen_enc_population.append(modname.enc_mutation(child2))
                                 
                enc_population = nextgen_enc_population

        #if best_change: 
        f_best, f_best_str = modname.get_stats(run_num, gen, f_best, f_best_str, enc_best, rep_best, data_dict) 
        
        if f_best < modname.GOOD_FITNESS: break
    
    return(f_best_str)

def rand_str(N): # return random string of length N, with upper-case characters 
    return ''.join(choices(ascii_uppercase, k=N))

def main():
    print("running ",modname.__name__)
    seed() # initialize internal state of random number generator

    all_data = [] # used when reading data from file --- for comparison with data from previous run
    if DATA_FILE: # then read from file, which is a list of dictionaries, one dictionary per single evolutionary run    
        with open(Path(RESULTS_FOLDER + DATA_FILE)) as f:
            for line in f:
                all_data.append(loads(line))
        
    if DATA_FILE:
        results_fname = RESULTS_FOLDER + DATA_FILE
    else:        
        results_fname = RESULTS_FOLDER + modname.__name__ + "_" + rand_str(6) 
    
    with open(Path(results_fname + ".params"),'w') as f: # write parameters into params file
        f.write(modname.get_params())

    with open(Path(results_fname + ".csv"),'w') as f: # write header in results file
        f.write(modname.get_header())
    
    for i in range(modname.NUM_EXPERIMENTS): # perform NUM_EXPERIMENTS evolutionary runs
        print("\r" + "run " + str(i+1), end = "\x1b[K")
        
        if DATA_FILE: # data was read from file
            data_dict = all_data[i]
        else: # data not read from file -- needs to be generated (and saved)
            data_dict = modname.init_stuff() # get various params and data specific to problem
            with open(Path(results_fname + ".dicts"), 'a') as f:
#                dump(data_dict, f)
                f.write("\n")
        
        f_best_str = run_evolutionary_alg(i+1, data_dict)  
            
        print(results_fname + ".csv");
        print(f_best_str)
        with open(Path(results_fname + ".csv"),'a') as f: 
            f.write(f_best_str)
        
    
if __name__== "__main__":
  main()
