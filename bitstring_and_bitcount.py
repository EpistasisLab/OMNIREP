#
# OMNIREP, copyright 2018 moshe sipper, www.moshesipper.com
#

# problem: cubic regression, ax^3+bx^2+cx+d 
# representation: bitstring; encoding: bit count per parameter -- a, b, c, d

from random import uniform, random, randint
from sklearn.metrics import mean_absolute_error
from common import single_point_crossover

NUM_EXPERIMENTS = 1000 # total number of evolutionary runs
GENERATIONS = 1000 # maximal number of generations to run the coevolutionary algorithm
ENC_POP_SIZE = 50 # size of encoding population, keep it even ...
REP_POP_SIZE = 100 # size of representation population, keep it even ...
REP_PROB_MUTATION = 0.3 # bitwise probability of mutation in a representation individual
ENC_PROB_MUTATION = 0.3 # individual probability of mutation in an encoding individual
TOURNAMENT_SIZE = 4 # size of tournament for tournament selection
ENC_GAP = 3 # evolve encodings every ENC_GAP generations
TOP_COUNT = 4 # number of top individuals in the other population used to compute fitness
GOOD_FITNESS = 0.001 # stop evolutionary run when reaching this threshold

NUM_VALS_TABLE = 200 # number of values in regression data table
ENC_GENOME_SIZE = 4 # (a, b, c, d) -- number of parameters per individual in the encoding population
REP_GENOME_SIZE = 120 # number of bits per individual in the representation population
REP_MIN_BITS = 10 # minimum number of bits per parameter = rep individual (4 params: a, b, c, d)

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
        "REP_GENOME_SIZE = "   + str(REP_GENOME_SIZE)   + "\n" +\
        "REP_MIN_BITS = "      + str(REP_MIN_BITS)      + "\n"   )

def target_func (a, b, c, d, x): # target regression function for evolutionary algorithm to optimize
    return(a*x*x*x + b*x*x + c*x + d)
    
def init_stuff(): # generate table of independent (x) and dependent (y) values, y=ax^3+bx^2+cx+d   
    target_a, target_b, target_c, target_d =\
        uniform(0,1), uniform(0,1), uniform(0,1), uniform(0,1) # target a,b,c,d
    x_vals = [uniform(0,1) for x in range(NUM_VALS_TABLE)]
    target_vals = [target_func(target_a, target_b, target_c, target_d, x) for x in x_vals]
    x_test = [uniform(0,1) for x in range(NUM_VALS_TABLE)]
    y_test = [target_func(target_a, target_b, target_c, target_d, x) for x in x_test]
    return ({"target_a": target_a, 
             "target_b": target_b, 
             "target_c": target_c, 
             "target_d": target_d, 
             "x_vals": x_vals, 
             "target_vals": target_vals,
             "x_test" : x_test,
             "y_test" : y_test
            })
    
def rep_init_population(data_dict): # initialize population of representations (individual = list of REP_GENOME_SIZE bits)
    return [ [randint(0, 1) for j in range(REP_GENOME_SIZE)] for i in range(REP_POP_SIZE) ]
        
def enc_init_population(ENC_EVOLVE, data_dict): # initialize population of encodings (individual = list of ENC_GENOME_SIZE integers)
    if ENC_EVOLVE:
        return [ [randint(REP_MIN_BITS, int(REP_GENOME_SIZE/ENC_GENOME_SIZE)) for j in range(ENC_GENOME_SIZE)] for i in range(ENC_POP_SIZE)] # each parameter has at least one bit -- the sign bit -- but set REP_MIN_BITS to a higher value
    else: # for testing with fixed representation: a,b,c,d have equal num bits and encodings don't evolve 
        return [ [int(REP_GENOME_SIZE/ENC_GENOME_SIZE) for i in range(ENC_GENOME_SIZE)] for j in range(ENC_POP_SIZE)] 

def bits_to_decimal(bits): # convert bit representation to decimal
    n=len(bits) 
    if (n<=1):
        return 0
    else:
        sign = 1 if (bits[0] == 0) else -1 # first bit is sign, rest are number
        dec=0
        for i in range(n-1):
            dec += bits[n-1-i]*(2**i)
        dec *= sign/(2**(n-1)-1) 
        return(dec)

def decode_abcd(enc_ind, rep_ind): # use encoding individual to decode (a, b, c, d) of representation individual 
    param = [0] * ENC_GENOME_SIZE # number of coefficients, we're focusing on 4, a,b,c,d
    low = 0
    for i in range(ENC_GENOME_SIZE):
        param[i] = bits_to_decimal(rep_ind[low : (low+enc_ind[i])]) # enc_ind[i] is num bits per param[i]
        low += enc_ind[i]        
    return(param[0], param[1], param[2], param[3]) # assuming 4 coefficients here .... a,b,c,d    
    
def fitness(enc_ind, rep_ind, data_dict, test=False): # given encoding individual and representation individual compute fitness
    if test:
        x_vals = data_dict["x_test"]
        target_vals = data_dict["y_test"]
    else:    
        x_vals = data_dict["x_vals"]
        target_vals = data_dict["target_vals"]
    a, b, c, d = decode_abcd(enc_ind, rep_ind)
    y_vals = [target_func(a, b, c, d, x) for x in x_vals]    
    f = mean_absolute_error(target_vals, y_vals)
    return(f)
    
def rep_crossover(parent1, parent2):  
    return single_point_crossover (parent1, parent2)   

def enc_crossover(parent1, parent2):  
    return single_point_crossover (parent1, parent2)           
                
def rep_mutation(ind, data_dict): # bitwise mutation of an individual in the representation population
    def bitflip(bit): return 1 if (bit == 0) else 0
    for i in range(len(ind)):
        if random() < REP_PROB_MUTATION:
            ind = ind[:i] + [bitflip(ind[i])] + ind[i+1:]
    return(ind)        
  
def enc_mutation(ind): # mutation of an individual in the encoding population
    if (random() >= ENC_PROB_MUTATION):
        return(ind) # no mutation 
    else: # perform mutation 
        gene = randint(0, len(ind)-1)
        ind[gene] = randint(REP_MIN_BITS, int(REP_GENOME_SIZE/ENC_GENOME_SIZE))
        return(ind)        

def get_header(): # first line of results file
    return "run,gen,f,test,target_a,evol_a,a_bits,target_b,evol_b,b_bits,target_c,evol_c,c_bits,target_d,evol_d,d_bits,rep\n"
       
def get_stats(run_num, gen, f_best, f_best_str, enc_best, rep_best, data_dict): # create one line of results file
    f = fitness(enc_best, rep_best, data_dict)
    if (f<=f_best):
        print("**********************")
        f_best=f
        test= fitness(enc_best, rep_best, data_dict, True)        
        a, b, c, d = decode_abcd(enc_best, rep_best)
        a_bits, b_bits, c_bits, d_bits = enc_best[0], enc_best[1], enc_best[2], enc_best[3]
        target_a, target_b, target_c, target_d =\
            data_dict["target_a"], data_dict["target_b"], data_dict["target_c"], data_dict["target_d"]
        f_best_str = str(run_num)    + "," + str(gen) + "," + str(f)      + "," + str(test) + ","\
                     + str(target_a) + "," + str(a)   + "," + str(a_bits) + ","\
                     + str(target_b) + "," + str(b)   + "," + str(b_bits) + ","\
                     + str(target_c) + "," + str(c)   + "," + str(c_bits) + ","\
                     + str(target_d) + "," + str(d)   + "," + str(d_bits) + ","\
                     + "=\"" + ''.join(str(x) for x in rep_best) + "\""   + "\n"
    return(f_best, f_best_str)
