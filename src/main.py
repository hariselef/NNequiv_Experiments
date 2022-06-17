
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


## Equivalence Checking
## Architecture 1 vs Architecture 2 --- under all equivalence relations



# load pretrained keras models

# model_1 = tensorflow.keras.models.load_model("C:/Users/haris/Desktop/NN Equivalence Checking/BitVec models/" + name_nn1 + ".h5")




def calc_experiment(nn1, nn2, sanity, eq_type, dataset):

    model_1 = tensorflow.keras.models.load_model(nn1 + ".h5")
    model_2 = tensorflow.keras.models.load_model(nn2 + ".h5")

    


    if sanity == True:
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
        

    # unpack model parameters
    else:
        NN1_w1, NN1_b1, NN1_w2, NN1_b2 = model_1.get_weights()
        NN2_w1, NN2_b1, NN2_w2, NN2_b2, NN2_w3, NN2_b3 = model_2.get_weights()
        
        in_dim = NN1_w1.shape[0]

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

    
    # from numpy to python list
    # from list to list of z3.RealVal



    # replicate model architectures

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



    def NN_controller_1(x):
        z1 = vec_add(vec_mat_mul(w1_1, x), b1_1)
        a1 = Relu(z1)
        z2 = vec_add(vec_mat_mul(w2_1, a1), b2_1)
        a2 = Relu(z2)
        z3 = vec_add(vec_mat_mul(w3_1, a2), b3_1)
        a3 = Relu(z3)
        z4 = vec_add(vec_mat_mul(w4_1, a3), b4_1)
        h1 = HardTanh(z4)
        return h1



    def NN_controller_2(x):

        z1 = vec_add(vec_mat_mul(w1_2, x), b1_2)
        a1 = Relu(z1)
        z2 = vec_add(vec_mat_mul(w2_2, a1), b2_2)
        a2 = Relu(z2)
        z3 = vec_add(vec_mat_mul(w3_2, a2), b3_2)
        a3 = Relu(z3)
        z4 = vec_add(vec_mat_mul(w4_2, a3), b4_2)
        h1 = HardTanh(z4)
        return h1

    print("Experiment: ", 1 , "Input dimension: ", in_dim, "Equivalence type: ", eq_type, "Dataset: ", dataset)
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
    
    if dataset == "mpc":

        # mpc

        pred_1 = NN_controller_1(x)
        pred_2 = NN_controller_2(x)

    else:

        ## define ouput
        # BitVec, mnist
        pred_1 = NN_1(x)
        pred_2 = NN_1(x)



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

    elif eq_type == "L1":
        # ε-approximate equivalence
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
        s.add(Not(arg_max(pred_1) == arg_max(pred_2))) ## my_argmax
        #s.add(argmax(pred_1) != argmax(pred_2))

    elif eq_type == "paper_argmax":

        # paper argmax
        p = argmax_eq(pred_1, pred_2)
        s.add(Or(p))



    s.set('timeout', 600000)
    check = s.check()
    print(check)
    exec_time = time.time() - start_time
    print('Execution time:', exec_time)

    return check, exec_time








if __name__ == "__main__":

# datasets = ["mnist", "mpc", "BitVec"]

# lengths = [10,10,784,6]

# eq_types = ["L1", "L2", "Linf", "argmax", "paper_argmax", "strict"]



# for l in lengths:

# for ds in datasets:

# for eqt in eq_types:
    NN=["../models/Sanity_Check/BitVec/bitvec_1_1","../models/Sanity_Check/BitVec/bitvec_1_2","../models/Sanity_Check/BitVec/bitvec_1_4","../models/Sanity_Check/BitVec/bitvec_1_5","../models/Sanity_Check/BitVec/bitvec_1_6","../models/Sanity_Check/BitVec/bitvec_1_7"]
    # [check,ctime]=calc_experiment(nn1, nn2, True,  "strict", "BitVec")
    # [check,ctime]=calc_experiment(nn1, nn2, True,  "L1", "BitVec")
    # [check,ctime]=calc_experiment(nn1, nn2, True,  "L2", "BitVec")
    # [check,ctime]=calc_experiment(nn1, nn2, True,  "Linf", "BitVec")
    # [check,ctime]=calc_experiment(nn1, nn2, True,  "argmax", "BitVec")
    
    for nn in NN:

        eq_types = [ "strict","L1", "L2", "Linf", "argmax"]
    
        for eq in eq_types:
            [check,ctime]=calc_experiment(nn, nn, True,  eq, "BitVec")






