
from __future__ import print_function

import random
import numpy as np

import tensorflow
import keras
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import *
from tensorflow.keras.callbacks import LearningRateScheduler, ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.regularizers import *
from keras import backend as K

import os
import pandas as pd
import matplotlib.pyplot as plt

from z3 import *

import time

from func import *

tensorflow.compat.v1.logging.set_verbosity(tensorflow.compat.v1.logging.ERROR)
## Equivalence Checking


def calc_experiment(nn1, nn2, eq_type, dataset):

    model_1 = tensorflow.keras.models.load_model(nn1)
    model_2 = tensorflow.keras.models.load_model(nn2)

    
    # unpack model parameters
    NN1_w1, NN1_b1, NN1_w2, NN1_b2 = model_1.get_weights()
    NN2_w1, NN2_b1, NN2_w2, NN2_b2 = model_2.get_weights()

    in_dim = NN1_w1.shape[0]

    NN1_w1 = NN1_w1.tolist()
    NN1_b1 = NN1_b1.tolist()
    NN1_w2 = NN1_w2.tolist()
    NN1_b2 = NN1_b2.tolist()

    NN2_w1 = NN2_w1.tolist()
    NN2_b1 = NN2_b1.tolist()
    NN2_w2 = NN2_w2.tolist()
    NN2_b2 = NN2_b2.tolist()

    w_1_1 = list_to_RealVal(NN1_w1)
    b_1_1 = list_to_RealVal(NN1_b1)
    w_1_2 = list_to_RealVal(NN1_w2)
    b_1_2 = list_to_RealVal(NN1_b2)

    w_2_1 = list_to_RealVal(NN2_w1)
    b_2_1 = list_to_RealVal(NN2_b1)
    w_2_2 = list_to_RealVal(NN2_w2)
    b_2_2 = list_to_RealVal(NN2_b2)

    # model_1 -- BitVec, mnist
    def NN_1(x):
        z1 = vec_add(vec_mat_mul(w_1_1, x), b_1_1)
        a1 = Relu(z1)
        z2 = vec_add(vec_mat_mul(w_1_2, a1), b_1_2)
        return z2

    # model_2 -- BitVec, mnist
    def NN_2(x):
        z1 = vec_add(vec_mat_mul(w_2_1, x), b_2_1)
        a1 = Relu(z1)
        z2 = vec_add(vec_mat_mul(w_2_2, a1), b_2_2)
        return z2
    

    print("Experiment: ", os.path.basename(os.path.normpath(nn)) , "Input dimension: ", in_dim, "Equivalence type: ", eq_type, "Dataset: ", dataset)
    # print('\n')

    # define input
    ## BitVec
    x = [Real('x%s' % (i+1)) for i in range(in_dim)]


    if eq_type == "L2":

        ## for L2 approximate equivalence add 2 more variables
        y = Real('y')
        z = Real('z')



    start_time = time.time()

    ## Solver instances ##
    # Default
    #s = Solver()

    # Quantifier free Non Linear Real Arithmetic - nlSAT solver
    s = Tactic('qfnra-nlsat').solver()
    # s.reset()

    ## define ouput
    # BitVec, mnist
    pred_1 = NN_1(x)
    pred_2 = NN_2(x)


## input constraints ##

    if dataset == "BitVec":

        # BitVec
        s.add(Or(0 == x[0], x[0] == 1))
        s.add(Or(0 == x[1], x[1] == 1))
        s.add(Or(0 == x[2], x[2] == 1))
        s.add(Or(0 == x[3], x[3] == 1))
        s.add(Or(0 == x[4], x[4] == 1))
        s.add(Or(0 == x[5], x[5] == 1))
        s.add(Or(0 == x[6], x[6] == 1))
        s.add(Or(0 == x[7], x[7] == 1))
        s.add(Or(0 == x[8], x[8] == 1))
        s.add(Or(0 == x[9], x[9] == 1))

    elif dataset == "mnist":

        # mnist

        for i in range(len(x)):
            s.add(0 <= x[i], x[i] <= 1)


    elif dataset == "mpc":

        # mpc
        const1 = -2 <= x1, x1 <= 2
        const2 = -1.04 <= x2, x2 <= 1.04
        const3 = -1 <= x3, x3 <= 1
        const4 = -0.8 <= x4, x4 <= 0.8
        const5 = -1.04 <= x5, x5 <= 1.04
        const6 = -0.01 <= x6, x6 <= 0.01

        s.add(const1)
        s.add(const2)
        s.add(const3)
        s.add(const4)
        s.add(const5)
        s.add(const6)

    ## Definitions of NN Equivalence

    if eq_type == "strict":
        # strict equivalence
        s.add(Not(pred_1 == pred_2))
        #s.add(pred_1 != pred_2)

    # ε-approximate equivalence
    elif eq_type == "L1":
        s.add(L1_norm(pred_1, pred_2) > 5) ## L1

    elif eq_type == "Linf":
        s.add(Linf_norm(pred_1, pred_2) > 10) ## Linf

    elif eq_type == "L2":
    ## L2
        s.add(y == z * z)
        s.add(y >= 0)
        s.add(y == L2_norm(pred_1, pred_2))
        s.add(z > 10)

    elif eq_type == "argmax":
        # argmax equivalence
        s.add(Not(argmax(pred_1) == argmax(pred_2))) 
        #s.add(argmax(pred_1) != argmax(pred_2))

    elif eq_type == "paper_argmax":
        # alternative argmax
        p = argmax_eq(pred_1, pred_2)
        s.add(Or(p))


    s.set('timeout', 600000)
    check = s.check()
    print('Result:', check)
    exec_time = time.time() - start_time
    print('Execution time: {:.2f} s'.format(exec_time))
    print('---------------------------------------------')
    # return check, exec_time



if __name__ == "__main__":

    import re
    numbers = re.compile(r'(\d+)')

    def numericalSort(value):
        parts = numbers.split(value)
        parts[1::2] = map(int, parts[1::2])
        return parts


    directory = 'models/Sanity_Check/mnist/arch_1/'

    NN = []
    # iterate over files in that directory
    # for filename in sorted(os.listdir(directory), key=len):
    for filename in sorted(os.listdir(directory), key=numericalSort):
        f = os.path.join(directory, filename)
        # checking if it is a file
        if os.path.isfile(f):
            NN.append(f)
    
    # directory = select_dir('mnist', True, 1) 
    # NN = model_pool(directory)

    # NN=["../models/Sanity_Check/BitVec/bitvec_1_1"]
    # [check,ctime]=calc_experiment(nn1, nn2, True,  "strict", "BitVec")
    # [check,ctime]=calc_experiment(nn1, nn2, True,  "L1", "BitVec")
    # [check,ctime]=calc_experiment(nn1, nn2, True,  "L2", "BitVec")
    # [check,ctime]=calc_experiment(nn1, nn2, True,  "Linf", "BitVec")
    # [check,ctime]=calc_experiment(nn1, nn2, True,  "argmax", "BitVec")
    
    global_time = time.time()
    for nn in NN:

        eq_types = [ "strict", "L1", "L2", "Linf", "argmax"]
    
        for eq in eq_types:
            calc_experiment(nn, nn, eq, "mnist")
            # [check,ctime]=calc_experiment(nn, nn, True,  eq, "BitVec")

    glob_time =  time.time() - global_time
    print('The overall time for the current experiment is: {:.2f} s'.format(glob_time))
# print(check, ctime)



