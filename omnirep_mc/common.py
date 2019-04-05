#
# OMNIREP, copyright 2018 moshe sipper, www.moshesipper.com
#

from random import randint

def single_point_crossover(parent1, parent2): 
# single-point xo between two genomes (lists), assume both equal in size   
    xo_point = randint(1, len(parent1)-1)
    return(parent1[:xo_point]+parent2[xo_point:], parent2[:xo_point]+parent1[xo_point:])
