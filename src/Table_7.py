
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
    NN2_w1, NN2_b1, NN2_w2, NN2_b2, NN2_w3, NN2_b3 = model_2.get_weights()
        
    in_dim = NN1_w1.shape[0]

    # from numpy to python list
    NN1_w1 = NN1_w1.tolist()
    NN1_b1 = NN1_b1.tolist()
    NN1_w2 = NN1_w2.tolist()
    NN1_b2 = NN1_b2.tolist()

    NN2_w1 = NN2_w1.tolist()
    NN2_b1 = NN2_b1.tolist()
    NN2_w2 = NN2_w2.tolist()
    NN2_b2 = NN2_b2.tolist()
    NN2_w3 = NN2_w3.tolist()
    NN2_b3 = NN2_b3.tolist()

    # from list to list of z3.RealVal
    w_1_1 = list_to_RealVal(NN1_w1)
    b_1_1 = list_to_RealVal(NN1_b1)
    w_1_2 = list_to_RealVal(NN1_w2)
    b_1_2 = list_to_RealVal(NN1_b2)

    w_2_1 = list_to_RealVal(NN2_w1)
    b_2_1 = list_to_RealVal(NN2_b1)
    w_2_2 = list_to_RealVal(NN2_w2)
    b_2_2 = list_to_RealVal(NN2_b2)
    w_2_3 = list_to_RealVal(NN2_w3)
    b_2_3 = list_to_RealVal(NN2_b3)

## replicate model architectures

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
        a2 = Relu(z2)
        z3 = vec_add(vec_mat_mul(w_2_3, a2), b_2_3)
        return z3


    print("L2 Equivalence Checking", "Input dimension: ", in_dim, "Equivalence type: ", eq_type, "Dataset: ", dataset)
    # print('\n')
    # define input
    ## BitVec

    x = [Real('x%s' % (i+1)) for i in range(in_dim)]


    

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
    
    if dataset == "mpc":

        # mpc
        pred_1 = NN_controller_1(x)
        pred_2 = NN_controller_2(x)

    else:

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

    if eq_type == "L2_1":
    ## L2
        s.add(y == z * z)
        s.add(y >= 0)
        s.add(y == L2_norm(pred_1, pred_2))
        s.add(z > 1)

    elif eq_type == "L2_10":
    ## L2
        s.add(y == z * z)
        s.add(y >= 0)
        s.add(y == L2_norm(pred_1, pred_2))
        s.add(z > 10)


    # s.set('timeout', 6000)
    check = s.check()
    
    if check == unknown:
        print('Result:', s.reason_unknown())
        exec_time = time.time() - start_time
        print('Execution time: {:.2f} s'.format(exec_time))
        print('---------------------------------------------')
        pass
    else:
        print('Result:', check)
        exec_time = time.time() - start_time
        print('Execution time: {:.2f} s'.format(exec_time))
        print('---------------------------------------------')
    # return check, exec_time



if __name__ == "__main__":

    global_time = time.time()

    model_1_1 = 'models/Equivalence_Checking/BitVec/arch_1/bitvec_1_1.h5'
    model_2_1 = 'models/Equivalence_Checking/BitVec/arch_2/bitvec_2_1.h5'
    model_1_2 = 'models/Equivalence_Checking/BitVec/arch_1/bitvec_1_2.h5'
    model_2_2 = 'models/Equivalence_Checking/BitVec/arch_2/bitvec_2_2.h5'
    mnist_1_1 = 'models/Equivalence_Checking/mnist/arch_1/mnist_1_1.h5'
    mnist_2_1 = 'models/Equivalence_Checking/mnist/arch_2/mnist_2_1.h5'

    
    calc_experiment(model_1_1, model_2_1, 'L2_1', "BitVec")
    calc_experiment(model_1_1, model_2_1, 'L2_10', "BitVec")
    calc_experiment(model_1_2, model_2_2, 'L2_1', "BitVec")
    calc_experiment(model_1_2, model_2_2, 'L2_10', "BitVec")
    calc_experiment(mnist_1_1, mnist_2_1, 'L2_1', "mnist")
    calc_experiment(mnist_1_1, mnist_2_1, 'L2_10', "mnist")
        

    glob_time =  time.time() - global_time
    print('The overall time for the current experiment is: {:.2f} s'.format(glob_time))