#
# OMNIREP, copyright 2018 moshe sipper, www.moshesipper.com
#

# problem: evolve circles to approximate a picture
# representation: circle centers; encoding: color and radius of each circle

from random import randint, random
from PIL import Image, ImageDraw
from pathlib import Path
from sklearn.metrics import mean_absolute_error
from sys import exit
from common import single_point_crossover

NUM_EXPERIMENTS = 1 # total number of evolutionary runs
GENERATIONS = 20000 # maximal number of generations to run the coevolutionary algorithm
ENC_POP_SIZE = 10 # size of encoding population, keep it even ...
REP_POP_SIZE = 20 # size of representation population, keep it even ...
REP_PROB_MUTATION = 0.3 # probability of mutation of a single line in a representation individual
ENC_PROB_MUTATION = 0.3 # probability of mutation of a single value (instruction) in an encoding individual
TOURNAMENT_SIZE = 4 # size of tournament for tournament selection
ENC_GAP = 3 # evolve encodings every ENC_GAP generations
TOP_COUNT = 4 # number of top individuals in the other population used to compute fitness
GOOD_FITNESS = 0 # stop evolutionary run when reaching this threshold

NUM_CIRCLES = 100#2000 # how many circles compose the picture
MIN_RADIUS = 1 # minimum radius of circle
MAX_RADIUS = 10#20 # maximum radius of circle
ENC_GENOME_SIZE = NUM_CIRCLES # encoding individual encodes radius of each circle
REP_GENOME_SIZE = NUM_CIRCLES # representation individual represents circle centers and colors
NUM_COLORS = 4 

IMAGE_FOLDER = 'images/adamhand/c4/'
IMAGE_FILE = 'adamhand' 

#IMAGE_FOLDER = 'images/americangothic/c4/'
#IMAGE_FILE = 'americangothic' 

#IMAGE_FOLDER = 'images/eiffel/c4/'
#IMAGE_FILE = 'eiffel' 

#IMAGE_FOLDER = 'images/girlpearl/c4/'
#IMAGE_FILE = 'girlpearl' 

#IMAGE_FOLDER = 'images/m/c4/'
#IMAGE_FILE = 'm' 

#IMAGE_FOLDER = 'images/monalisa/c4/'
#IMAGE_FILE = 'monalisa' 

#IMAGE_FOLDER = 'images/pacman/c4/'
#IMAGE_FILE = 'pacman' 

#IMAGE_FOLDER = 'images/selfportrait/c4/'
#IMAGE_FILE = 'selfportrait' 

#IMAGE_FOLDER = 'images/thescream/c4/'
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
        "NUM_CIRCLES = "       + str(NUM_CIRCLES)      + "\n" +\
        "MIN_RADIUS = "        + str(MIN_RADIUS)       + "\n" +\
        "MAX_RADIUS = "        + str(MAX_RADIUS)       + "\n" +\
        "NUM_COLORS = "        + str(NUM_COLORS)        + "\n" +\
        "ENC_GENOME_SIZE = "   + str(ENC_GENOME_SIZE)   + "\n" +\
        "REP_GENOME_SIZE = "   + str(REP_GENOME_SIZE)    + "\n"  )

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
    
def rep_init_population(data_dict): # init representations population, individual = list of circle centers and radii [[x,y], ...]
    width, height = data_dict["width"], data_dict["height"]
    return [ [ [randint(0, width-1), randint(0, height-1)] for j in range(REP_GENOME_SIZE)] for i in range(REP_POP_SIZE) ]

def enc_init_population(ENC_EVOLVE, data_dict): # init encodings population, individual = list of colors, [[c,r], ....]
    if ENC_EVOLVE:
        return [ [ [randint(0,NUM_COLORS-1), randint(MIN_RADIUS,MAX_RADIUS)] for j in range(ENC_GENOME_SIZE)] for i in range(ENC_POP_SIZE)] 
    else: # for testing with fixed representation 
        exit("enc_init_population(): not supported")
    
def draw_image(enc_ind, rep_ind, data_dict):  
    width, height = data_dict["width"], data_dict["height"]
    palette = data_dict["palette"]
    image = Image.new('P', (width, height))
    image.putpalette(palette)
    draw = ImageDraw.Draw(image)
    for i in range(REP_GENOME_SIZE):
        c_x, c_y = rep_ind[i][0], rep_ind[i][1] # circle center 
        color, rad = enc_ind[i][0], enc_ind[i][1] # color and radius
        draw.ellipse((c_x-rad, c_y-rad, c_x+rad, c_y+rad), fill=color) 
    del draw 
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

def __rep_mutation(ind, data_dict): # mutation of an individual in the representation population
    if (random() >= REP_PROB_MUTATION):
        return(ind) # no mutation 
    else: # perform mutation, replace random [x,y] with new vals
        width, height = data_dict["width"], data_dict["height"]
        gene = randint(0, REP_GENOME_SIZE-1)
        ind[gene] = [randint(0, width-1), randint(0, height-1)]              
        return(ind)  
		
def rep_crossover(parent1, parent2):  
    return single_point_crossover (parent1, parent2)   

def enc_crossover(parent1, parent2):  
    return single_point_crossover (parent1, parent2)       

def rep_mutation(ind, data_dict): # mutation of an individual in the representation population
        width, height = data_dict["width"], data_dict["height"]
        for i in range(REP_GENOME_SIZE):
            if (random() <= REP_PROB_MUTATION):
                ind[i] = [randint(0, width-1), randint(0, height-1)] 
        return(ind)

def enc_mutation(ind): # mutation of an individual in the encoding population
        for i in range(ENC_GENOME_SIZE):
            if (random() <= ENC_PROB_MUTATION):
                ind[i] = [randint(0,NUM_COLORS-1), randint(MIN_RADIUS,MAX_RADIUS)]
        return(ind)        

def __enc_mutation(ind): # mutation of an individual in the encoding population
    if (random() >= ENC_PROB_MUTATION):
        return(ind) # no mutation 
    else: # perform mutation 
        gene = randint(0, ENC_GENOME_SIZE-1)
        ind[gene] = randint(MIN_RADIUS,MAX_RADIUS)
        return(ind)     
        
def get_header(): # first line of results file
    return "run,gen,f,enc_best\n"

def get_stats(run_num, gen, f_best, f_best_str, enc_best, rep_best, data_dict): # create one line of results file
	f = fitness(enc_best, rep_best, data_dict)
	if (gen%500 ==0):
		image = draw_image(enc_best, rep_best, data_dict)
		image.save(Path(IMAGE_FOLDER + IMAGE_FILE + '_' + str(run_num) + '_' + str(gen) + '.png'), "PNG")
	if (f<f_best):
		print("\n  gen=",gen," f=",f)
		print(enc_best)
		f_best=f
		image = draw_image(enc_best, rep_best, data_dict)
		image.save(Path(IMAGE_FOLDER + IMAGE_FILE + '_' + str(run_num) + '_' + str(getpid()) + '.png'), "PNG")
		#        enc_best_str = ":".join([str(enc_best[i]) for i in range(ENC_GENOME_SIZE)])
		f_best_str = str(run_num) + "," + str(gen) + "," + str(f) + "\n"
	return(f_best, f_best_str)
