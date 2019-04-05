#
# OMNIREP, copyright 2018 moshe sipper, www.moshesipper.com
#

# problem: find program that emulates output of target program
# representation: program with generic instructions; encoding: instruction set 

import math
from random import uniform, random, randint
from sklearn.metrics import mean_absolute_error
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
ENC_GENOME_SIZE = 5 # number of instructions in encoding individual
REP_GENOME_SIZE = 10 # program length (number of lines), each line is an index into encoding individual, i.e., instruction

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
        "ENC_GENOME_SIZE = "   + str(ENC_GENOME_SIZE)   + "\n" +\
        "REP_GENOME_SIZE = "   + str(REP_GENOME_SIZE)   + "\n"   )

def plus1(x): return(x+1)
def plus2(x): return(x+2)
def plus3(x): return(x+3)
def plus4(x): return(x+4)
def plus5(x): return(x+5)
def minus1(x): return(x-1)
def minus2(x): return(x-2)
def minus3(x): return(x-3)
def minus4(x): return(x-4)
def minus5(x): return(x-5)
def mul2(x): return(x*2)
def mul3(x): return(x*3)
def mul4(x): return(x*4)
def mul5(x): return(x*5)
def mul10(x): return(x*10)
def div2(x): return(x/2)
def div3(x): return(x/3)
def div4(x): return(x/4)
def div5(x): return(x/5)
def div10(x): return(x/10)
def sin(x): return math.sin(x)
def cos(x): return math.cos(x)
def tan(x): return math.tan(x)
def floor(x): return math.floor(x)
def ceil(x): return math.ceil(x)
def degrees(x): return math.degrees(x)
def radians(x): return math.radians(x)
def fabs(x): return math.fabs(x)

functions =\
    [plus1,plus2,plus3,plus4,plus5,minus1,minus2,minus3,minus4,minus5,mul2,mul3,mul4,\
     mul5,mul10,div2,div3,div4,div5,div10,sin,cos,tan,floor,ceil,degrees,radians,fabs]
functions_str = [f.__name__ for f in functions]
NUM_FUNCS = len(functions)

def run_program(enc_ind, rep_ind, x): # run program over a single x value
    for i in range(REP_GENOME_SIZE):
        x = functions[enc_ind[rep_ind[i]]](x)
    return x
        
def init_stuff(): # composition of ENC_GENOME_SIZE randomly selected functions 
    target_enc = [randint(0,NUM_FUNCS-1) for i in range(ENC_GENOME_SIZE)] # target encoding 
    target_rep = [randint(0,ENC_GENOME_SIZE-1) for i in range(REP_GENOME_SIZE)] # target representation (program)
    x_vals = [uniform(0,1) for x in range(NUM_VALS_TABLE)] 
    target_vals = [run_program(target_enc, target_rep, x) for x in x_vals]
    x_test = [uniform(0,1) for x in range(NUM_VALS_TABLE)] 
    y_test = [run_program(target_enc, target_rep, x) for x in x_test]
    return ({"target_enc": target_enc,
             "target_rep": target_rep,
             "x_vals": x_vals,
             "target_vals": target_vals,
             "x_test" : x_test,
             "y_test" : y_test
             })

def rep_init_population(data_dict): # initialize population of representations, individual = list of program lines, each line is an index into encoding individual, i.e., instruction
    return [ [randint(0, ENC_GENOME_SIZE-1) for j in range(REP_GENOME_SIZE)] for i in range(REP_POP_SIZE) ]

def enc_init_population(ENC_EVOLVE, data_dict): # initialize population of encodings, individual = list of instructions (indexes into functions array) 
    if ENC_EVOLVE:
        return [ [randint(0,NUM_FUNCS-1) for j in range(ENC_GENOME_SIZE)] for i in range(ENC_POP_SIZE) ]
#    else: # no encoding evolution, set entire population to a random encoding
#        encoding = [randint(0,NUM_FUNCS-1) for j in range(ENC_GENOME_SIZE)]
#        return [encoding for i in range(ENC_POP_SIZE) ]
    else: # no encoding evolution, set entire population to an encoding with 2 instructions from target, 3 random
        encoding = list(data_dict["target_enc"]) # use list for deep copy
        for j in range(ENC_GENOME_SIZE-2):
            encoding[j] = randint(0,NUM_FUNCS-1)
        return [encoding for i in range(ENC_POP_SIZE) ]
    
def fitness(enc_ind, rep_ind, data_dict, test=False): # given encoding individual and representation individual compute fitness
    if test:
        x_vals = data_dict["x_test"]
        target_vals = data_dict["y_test"]
    else:    
        x_vals = data_dict["x_vals"]
        target_vals = data_dict["target_vals"]
    y_vals= [run_program(enc_ind, rep_ind, x) for x in x_vals]
    f = mean_absolute_error(target_vals, y_vals)
    return(f)   

def rep_crossover(parent1, parent2):  
    return single_point_crossover (parent1, parent2)   

def enc_crossover(parent1, parent2):  
    return single_point_crossover (parent1, parent2)   
        
def rep_mutation(ind, data_dict): # mutation of an individual in the representation population
    if (random() >= REP_PROB_MUTATION):
        return(ind) # no mutation 
    else: # perform mutation, replace random instruction, of random line, with (different) random value
        line = randint(0, REP_GENOME_SIZE-1)
        newval = randint(0, ENC_GENOME_SIZE-1)
        while (newval == ind[line]):
            newval = randint(0, ENC_GENOME_SIZE-1)
        ind[line] = newval                
        return(ind)        

def enc_mutation(ind): # mutation of an individual in the encoding population
    if (random() >= ENC_PROB_MUTATION):
        return(ind) # no mutation 
    else: # perform mutation 
        gene = randint(0, ENC_GENOME_SIZE-1)
        newval = randint(0,NUM_FUNCS-1)  
        while (newval == ind[gene]):
            newval = randint(0,NUM_FUNCS-1)
        ind[gene] = newval
        return(ind)     
        
def get_header(): # first line of results file
    return "run,gen,f,test,target_enc,enc_best,target_rep,rep_best\n"

def get_stats(run_num, gen, f_best, f_best_str, enc_best, rep_best, data_dict): # create one line of results file
    f = fitness(enc_best, rep_best, data_dict)
    if (f<=f_best):
        f_best=f
        test= fitness(enc_best, rep_best, data_dict, True)
        target_enc = data_dict["target_enc"]
        target_rep = data_dict["target_rep"]
        target_enc_decoded = ":".join([functions_str[target_enc[i]] for i in range(ENC_GENOME_SIZE)])
        enc_best_decoded   = ":".join([functions_str[enc_best[i]]   for i in range(ENC_GENOME_SIZE)])
        target_rep_decoded = ":".join([functions_str[target_enc[target_rep[i]]] for i in range(REP_GENOME_SIZE)])
        rep_best_decoded   = ":".join([functions_str[enc_best[rep_best[i]]]     for i in range(REP_GENOME_SIZE)])
        f_best_str = str(run_num) + "," + str(gen) + "," + str(f) + "," + str(test) + "," +\
                    target_enc_decoded + "," + enc_best_decoded + "," + target_rep_decoded + "," + rep_best_decoded + "\n"
    return(f_best, f_best_str)
