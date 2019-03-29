#
# OMNIREP, copyright 2018 moshe sipper, www.moshesipper.com
#

# problem: regression, sum a_ix^{e_i}
# representation: floating-point values; encoding: precision -- number of decimal places after decimal point, per parameter

from random import randint, random, uniform
from sklearn.metrics import mean_absolute_error
from math import pow
from statistics import mean, median
from common import single_point_crossover

NUM_EXPERIMENTS = 1000 # total number of evolutionary runs
GENERATIONS = 1000 # maximal number of generations to run the coevolutionary algorithm
ENC_POP_SIZE = 50 # size of encoding population, keep it even ...
REP_POP_SIZE = 100 # size of representation population, keep it even ...
REP_PROB_MUTATION = 0.3 # probability of mutation of a single line in a representation individual
ENC_PROB_MUTATION = 0.3 # probability of mutation of a single value (instruction) in an encoding individual
TOURNAMENT_SIZE = 4 # size of tournament for tournament selection
ENC_GAP = 3 # evolve encodings every ENC_GAP generations
TOP_COUNT = 4 # number of top individuals in the other population used to compute fitness
GOOD_FITNESS = 0.001 # stop evolutionary run when reaching this threshold

NUM_VALS_TABLE = 200 # number of values in data table
NUM_COEFFICIENTS = 50
MAX_EXPONENT = 4
ENC_GENOME_SIZE = NUM_COEFFICIENTS 
REP_GENOME_SIZE = ENC_GENOME_SIZE 
REP_MIN_DECIMALS = 1 # minimum number of decimal places per parameter 
REP_MAX_DECIMALS = 8 # maximum number of decimal places per parameter 

def get_params():
    return(\
        "NUM_EXPERIMENTS = "   + str(NUM_EXPERIMENTS)   + "\n" +\
        "GENERATIONS = "       + str(GENERATIONS)       + "\n" +\
        "ENC_POP_SIZE = "      + str(ENC_POP_SIZE)      + "\n" +\
        "REP_POP_SIZE = "      + str(REP_POP_SIZE)      + "\n" +\
        "REP_PROB_MUTATION = " + str(REP_PROB_MUTATION) + "\n" +\
        "ENC_PROB_MUTATION = " + str(ENC_PROB_MUTATION) + "\n" +\
        "NUM_VALS_TABLE = "    + str(NUM_VALS_TABLE)    + "\n" +\
        "TOURNAMENT_SIZE = "   + str(TOURNAMENT_SIZE)   + "\n" +\
        "ENC_GAP = "           + str(ENC_GAP)           + "\n" +\
        "TOP_COUNT = "         + str(TOP_COUNT)         + "\n" +\
        "GOOD_FITNESS = "      + str(GOOD_FITNESS)      + "\n" +\
        "NUM_VALS_TABLE = "    + str(NUM_VALS_TABLE)    + "\n" +\
        "NUM_COEFFICIENTS = "  + str(NUM_COEFFICIENTS)  + "\n" +\
        "MAX_EXPONENT = "      + str(MAX_EXPONENT)        + "\n" +\
        "ENC_GENOME_SIZE = "   + str(ENC_GENOME_SIZE)   + "\n" +\
        "REP_GENOME_SIZE = "   + str(REP_GENOME_SIZE)   + "\n" +\
        "REP_MIN_DECIMALS = "  + str(REP_MIN_DECIMALS)  + "\n" +\
        "REP_MAX_DECIMALS = "  + str(REP_MAX_DECIMALS)  + "\n"    )

def target_func (a, e, x): # target function for evolutionary algorithm to optimize
    exponents = [pow(x,e[i]) for i in range(NUM_COEFFICIENTS)]
    return(sum([a[i]*exponents[i] for i in range(NUM_COEFFICIENTS)]))
    
def init_stuff(): # generate table of independent (x) and dependent (y) values
    t_a = [uniform(0,1) for i in range(NUM_COEFFICIENTS)] # target coefficients
    t_exp = [randint(0,MAX_EXPONENT) for i in range(NUM_COEFFICIENTS)] # target exponents
    x_vals = [uniform(0,1) for x in range(NUM_VALS_TABLE)]
    target_vals = [target_func(t_a, t_exp, x) for x in x_vals]
    x_test = [uniform(0,1) for x in range(NUM_VALS_TABLE)]
    y_test = [target_func(t_a, t_exp, x) for x in x_test]
    return ({"t_a": t_a,
             "t_exp": t_exp,
             "x_vals": x_vals, 
             "target_vals": target_vals,
             "x_test" : x_test,
             "y_test" : y_test
            })

def rep_init_population(data_dict): # initialize population of representations (individual = list of REP_GENOME_SIZE real values)
    return [ [uniform(0,1) for j in range(REP_GENOME_SIZE)] for i in range(REP_POP_SIZE) ]

def enc_init_population(ENC_EVOLVE, data_dict): # initialize population of encodings (individual = list of ENC_GENOME_SIZE integers)
    if ENC_EVOLVE:
        return [ [randint(REP_MIN_DECIMALS, REP_MAX_DECIMALS) for j in range(ENC_GENOME_SIZE)] for i in range(ENC_POP_SIZE)] 
    else: # for testing with fixed representation
        return [ [REP_MAX_DECIMALS for j in range(ENC_GENOME_SIZE)] for i in range(ENC_POP_SIZE)] 

def decode_coefficients(enc_ind, rep_ind): # use precision in encoding individual to decode representation individual  
    return([round(rep_ind[i],enc_ind[i]) for i in range(REP_GENOME_SIZE)])

def fitness(enc_ind, rep_ind, data_dict, test=False): # given encoding individual and representation individual compute fitness
    # each coefficient's number of decimal places determined by corresponding encoding value   
    if test:
        x_vals = data_dict["x_test"]
        target_vals = data_dict["y_test"]
    else:    
        x_vals = data_dict["x_vals"]
        target_vals = data_dict["target_vals"]
    a = decode_coefficients(enc_ind, rep_ind)
    e = data_dict["t_exp"]
    y_vals = [target_func(a, e, x) for x in x_vals]    
    f = mean_absolute_error(target_vals, y_vals)
    return(f)    

def rep_crossover(parent1, parent2):  
    return single_point_crossover (parent1, parent2)   

def enc_crossover(parent1, parent2):  
    return single_point_crossover (parent1, parent2)       

def rep_mutation(ind, data_dict): # mutation of an individual in the representation population
    if (random() >= REP_PROB_MUTATION):
        return(ind) # no mutation 
    else: # perform mutation 
        gene = randint(0, REP_GENOME_SIZE-1)
        ind[gene] = uniform(0,1)
        return(ind)        

def enc_mutation(ind): # mutation of an individual in the encoding population
    if (random() >= ENC_PROB_MUTATION):
        return(ind) # no mutation 
    else: # perform mutation 
        gene = randint(0, ENC_GENOME_SIZE-1)
        newval = randint(REP_MIN_DECIMALS, REP_MAX_DECIMALS)
        while newval == ind[gene]:
            newval = randint(REP_MIN_DECIMALS, REP_MAX_DECIMALS)
        ind[gene] = newval
        return(ind)        
        
def get_header(): # first line of results file
    return "run,gen,f,test,mean_evol_precision,median_evol_precision\n"

def get_stats(run_num, gen, f_best, f_best_str, enc_best, rep_best, data_dict): # create one line of results file
    f = fitness(enc_best, rep_best, data_dict)
    if (f<=f_best):
        print("\n")
        print("gen=",gen, "f=",f, "enc=",enc_best)
        f_best=f
        test= fitness(enc_best, rep_best, data_dict, True)        
        f_best_str = str(run_num) + "," + str(gen) + "," + str(f)  + "," + str(test) + "," +\
                     str(mean(enc_best)) + "," + str(median(enc_best)) + "\n"
    return(f_best, f_best_str)
    