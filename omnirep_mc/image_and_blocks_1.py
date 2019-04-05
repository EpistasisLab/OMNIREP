#
# OMNIREP, copyright 2018 moshe sipper, www.moshesipper.com
#

# problem: evolve blocks to approximate a picture
# representation: start coordinates (x,y) for same-color blocks; 
# encoding:       block direction (horizontal/vertical), block size, block color

from random import randint, random
from PIL import Image, ImageDraw
from pathlib import Path
from sklearn.metrics import mean_absolute_error
from sys import exit
from os import getpid
from common import single_point_crossover

NUM_EXPERIMENTS = 1#10 # total number of evolutionary runs
GENERATIONS = 20000 # maximal number of generations to run the coevolutionary algorithm
ENC_POP_SIZE = 10 # size of encoding population, keep it even ...
REP_POP_SIZE = 20 # size of representation population, keep it even ...
REP_PROB_MUTATION = 0.3 # probability of mutation of a single line in a representation individual
ENC_PROB_MUTATION = 0.3 # probability of mutation of a single value (instruction) in an encoding individual
TOURNAMENT_SIZE = 4 # size of tournament for tournament selection
ENC_GAP = 3 # evolve encodings every ENC_GAP generations
TOP_COUNT = 4 # number of top individuals in the other population used to compute fitness
GOOD_FITNESS = 0 # stop evolutionary run when reaching this threshold

NUM_BLOCKS = 500 # how many blocks compose the picture
MIN_BLOCK_SIZE = 1 # pixels
MAX_BLOCK_SIZE = 10 # pixels
ENC_GENOME_SIZE = NUM_BLOCKS # encoding individual encodes block sizes and colors
REP_GENOME_SIZE = ENC_GENOME_SIZE 
NUM_COLORS = 4 

#IMAGE_FOLDER = 'images/adamhand/bb1/'
#IMAGE_FILE = 'adamhand' 

#IMAGE_FOLDER = 'images/americangothic/bb1/'
#IMAGE_FILE = 'americangothic' 

#IMAGE_FOLDER = 'images/eiffel/bb1/'
#IMAGE_FILE = 'eiffel' 

#IMAGE_FOLDER = 'images/girlpearl/bb1/'
#IMAGE_FILE = 'girlpearl' 

#IMAGE_FOLDER = 'images/m/bb1/'
#IMAGE_FILE = 'm' 

IMAGE_FOLDER = 'images/monalisa/bb2/'
IMAGE_FILE = 'monalisa' 

#IMAGE_FOLDER = 'images/pacman/bb1/'
#IMAGE_FILE = 'pacman' 

#IMAGE_FOLDER = 'images/selfportrait/bb1/'
#IMAGE_FILE = 'selfportrait' 

#IMAGE_FOLDER = 'images/thescream/bb1/'
#IMAGE_FILE = 'thescream' 

def get_params():
    return(\
        "NUM_EXPERIMENTS = "   + str(NUM_EXPERIMENTS)   + "\n" +\
        "GENERATIONS = "       + str(GENERATIONS)       + "\n" +\
        "ENC_POP_SIZE = "      + str(ENC_POP_SIZE)      + "\n" +\
        "REP_POP_SIZE = "      + str(REP_POP_SIZE)      + "\n" +\
        "REP_PROB_MUTATION = " + str(REP_PROB_MUTATION) + "\n" +\
        "ENC_PROB_MUTATION = " + str(ENC_PROB_MUTATION) + "\n" +\
        "TOURNAMENT_SIZE = "   + str(TOURNAMENT_SIZE)   + "\n" +\
        "ENC_GAP = "           + str(ENC_GAP)           + "\n" +\
        "TOP_COUNT = "         + str(TOP_COUNT)         + "\n" +\
        "GOOD_FITNESS = "      + str(GOOD_FITNESS)      + "\n" +\
        "NUM_BLOCKS = "        + str(NUM_BLOCKS)        + "\n" +\
        "MIN_BLOCK_SIZE = "    + str(MIN_BLOCK_SIZE)    + "\n" +\
        "MAX_BLOCK_SIZE = "    + str(MAX_BLOCK_SIZE)    + "\n" +\
        "NUM_COLORS = "        + str(NUM_COLORS)        + "\n" +\
        "ENC_GENOME_SIZE = "   + str(ENC_GENOME_SIZE)   + "\n" +\
        "REP_GENOME_SIZE = "   + str(REP_GENOME_SIZE)   + "\n" +\
        "IMAGE_FOLDER = "      + IMAGE_FOLDER           + "\n" +\
        "IMAGE_FILE = "        + IMAGE_FILE             + "\n" )

def init_stuff(): # get image and image stats  
    target_name = IMAGE_FOLDER + IMAGE_FILE + '.jpg'
    target = Image.open(Path(target_name))
    width, height = target.size
    target=target.convert('P', palette=Image.ADAPTIVE, colors=NUM_COLORS)
    palette = target.getpalette()
    target_pixels = list(target.getdata())
    target_histogram = target.histogram()
    return ({"target_name": target_name,
             "width" : width,
             "height": height,
             "target_pixels" : target_pixels,
             "palette" : palette,             
             "target_histogram" : target_histogram
            })
    
def rep_init_population(data_dict): # init representations population, individual = list of coords [[x,y], ...]
    width, height = data_dict["width"], data_dict["height"]
    return [ [ [randint(0, width-1), randint(0, height-1)] for j in range(REP_GENOME_SIZE)] for i in range(REP_POP_SIZE) ]

def rep_mutation(ind, data_dict): # mutation of an individual in the representation population
    if (random() >= REP_PROB_MUTATION):
        return(ind) # no mutation 
    else: # perform mutation
        width, height = data_dict["width"], data_dict["height"]
        gene = randint(0, REP_GENOME_SIZE-1)
        ind[gene] = [randint(0, width-1), randint(0, height-1)]             
        return(ind)  

def enc_init_population(ENC_EVOLVE, data_dict): # initialize population of encodings, individual = list of [block direction, block size, color]
    if ENC_EVOLVE:
        return [ [ [randint(0,1), randint(MIN_BLOCK_SIZE,MAX_BLOCK_SIZE), randint(0,NUM_COLORS-1)] for j in range(ENC_GENOME_SIZE)] for i in range(ENC_POP_SIZE)] 
    else: # for testing with fixed representation 
        exit("enc_init_population(): not supported")

def enc_mutation(ind): # mutation of an individual in the encoding population
    if (random() >= ENC_PROB_MUTATION):
        return(ind) # no mutation 
    else: # perform mutation 
        gene = randint(0, ENC_GENOME_SIZE-1)
        ind[gene] = [randint(0,1), randint(MIN_BLOCK_SIZE,MAX_BLOCK_SIZE), randint(0,NUM_COLORS-1)]
        return(ind)     

def draw_image(enc_ind, rep_ind, data_dict):  
    width, height = data_dict["width"], data_dict["height"]
    palette = data_dict["palette"]
    image = Image.new('P', (width, height))
    image.putpalette(palette)
    #draw = ImageDraw.Draw(image)
    for i in range(REP_GENOME_SIZE):
        x, y = rep_ind[i][0], rep_ind[i][1] # block start
        direction, size, color = enc_ind[i][0], enc_ind[i][1] , enc_ind[i][2] 
        for j in range(size):            
            image.putpixel((x,y),color) 
            if direction == 0: # horizontal
                x+=1
                if x >= width:
                    x=0
                    if y < height-1: y+=1
            else: # vertical
                y+=1
                if y >= height:
                    y=0
                    if x<width-1: x+=1
    #del draw 
    return image     

def fitness(enc_ind, rep_ind, data_dict): # given encoding individual and representation individual compute fitness    
    target_pixels = data_dict["target_pixels"]
#    target_histogram = data_dict["target_histogram"]
    current = draw_image(enc_ind, rep_ind, data_dict)
    current_pixels = list(current.getdata()) 
#    current_histogram = current.histogram()
#    f = mean_absolute_error(target_pixels, current_pixels)
    f = mean_absolute_error(target_pixels, current_pixels)
    return(f)
    
def rep_crossover(parent1, parent2):  
    return single_point_crossover (parent1, parent2)   

def enc_crossover(parent1, parent2):  
    return single_point_crossover (parent1, parent2)       

        
def get_header(): # first line of results file
    return "run,gen,f\n"

def get_stats(run_num, gen, f_best, f_best_str, enc_best, rep_best, data_dict): # create one line of results file
    f = fitness(enc_best, rep_best, data_dict)
    if (gen%500 == 0):
        image = draw_image(enc_best, rep_best, data_dict)
        image.save(Path(IMAGE_FOLDER + IMAGE_FILE + '_' + str(run_num) + '_' + str(gen) + '.png'), "PNG")
    if (f<f_best):
        print("gen=",gen," f=",f)
        f_best=f
        image = draw_image(enc_best, rep_best, data_dict)
        image.save(Path(IMAGE_FOLDER + IMAGE_FILE + '_' + str(run_num) + '_' + str(getpid()) + '.png'), "PNG")
#        enc_best_str = ":".join([str(enc_best[i]) for i in range(ENC_GENOME_SIZE)])
        f_best_str = str(run_num) + "," + str(gen) + "," + str(f) + "\n"
    return(f_best, f_best_str)
