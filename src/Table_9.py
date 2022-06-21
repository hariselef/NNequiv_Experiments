
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

# Create custom HardTanh Layer (Keras)
class HardTanh(Layer): 
    def __init__(self, name=None, **kwargs):
        super(HardTanh, self).__init__(name=name)
        super(HardTanh, self).__init__(**kwargs)

    def call(self, input_data):
        return tensorflow.clip_by_value(input_data, -1, 1) #* self.kernel

 
def calc_experiment(nn1, nn2, eq_type, dataset):



    model_1 = tensorflow.keras.models.load_model(nn1, custom_objects={'HardTanh':HardTanh})
    model_2 = tensorflow.keras.models.load_model(nn2, custom_objects={'HardTanh':HardTanh})

    
    w1_1, b1_1, w2_1, b2_1, w3_1, b3_1, w4_1, b4_1 = model_1.get_weights()
    w1_2, b1_2, w2_2, b2_2, w3_2, b3_2, w4_2, b4_2 = model_2.get_weights()

    in_dim = w1_1.shape[0]

    # from numpy to list()
    w1_1 = w1_1.tolist()
    b1_1 = b1_1.tolist()
    w2_1 = w2_1.tolist()
    b2_1 = b2_1.tolist()
    w3_1 = w3_1.tolist()
    b3_1 = b3_1.tolist()
    w4_1 = w4_1.tolist()
    b4_1 = b4_1.tolist()

    w1_2 = w1_2.tolist()
    b1_2 = b1_2.tolist()
    w2_2 = w2_2.tolist()
    b2_2 = b2_2.tolist()
    w3_2 = w3_2.tolist()
    b3_2 = b3_2.tolist()
    w4_2 = w4_2.tolist()
    b4_2 = b4_2.tolist()


    # from list to list of RealVal
    w1_1 = list_to_RealVal(w1_1)
    b1_1 = list_to_RealVal(b1_1)
    w2_1 = list_to_RealVal(w2_1)
    b2_1 = list_to_RealVal(b2_1)
    w3_1 = list_to_RealVal(w3_1)
    b3_1 = list_to_RealVal(b3_1)
    w4_1 = list_to_RealVal(w4_1)
    b4_1 = list_to_RealVal(b4_1)

    w1_2 = list_to_RealVal(w1_2)
    b1_2 = list_to_RealVal(b1_2)
    w2_2 = list_to_RealVal(w2_2)
    b2_2 = list_to_RealVal(b2_2)
    w3_2 = list_to_RealVal(w3_2)
    b3_2 = list_to_RealVal(b3_2)
    w4_2 = list_to_RealVal(w4_2)
    b4_2 = list_to_RealVal(b4_2)

    ## reproduce NN architecture
    def NN_controller_1(x):
        z1 = vec_add(vec_mat_mul(w1_1, x), b1_1)
        a1 = Relu(z1)
        z2 = vec_add(vec_mat_mul(w2_1, a1), b2_1)
        a2 = Relu(z2)
        z3 = vec_add(vec_mat_mul(w3_1, a2), b3_1)
        a3 = Relu(z3)
        z4 = vec_add(vec_mat_mul(w4_1, a3), b4_1)
        h1 = hardTanh(z4)
        return h1

    def NN_controller_2(x):
        z1 = vec_add(vec_mat_mul(w1_2, x), b1_2)
        a1 = Relu(z1)
        z2 = vec_add(vec_mat_mul(w2_2, a1), b2_2)
        a2 = Relu(z2)
        z3 = vec_add(vec_mat_mul(w3_2, a2), b3_2)
        a3 = Relu(z3)
        z4 = vec_add(vec_mat_mul(w4_2, a3), b4_2)
        h1 = hardTanh(z4)
        return h1

    print("Experiment:", os.path.basename(os.path.normpath(nn1)), "vs", os.path.basename(os.path.normpath(nn2)), "Input dimension: ", in_dim, "Equivalence type: ", eq_type, "Dataset: ", dataset)
    # print('\n')
    # define input
    ## BitVec

    x = [Real('x%s' % (i+1)) for i in range(in_dim)]



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
        
        x1 = x[0]
        x2 = x[1]
        x3 = x[2]
        x4 = x[3]
        x5 = x[4]
        x6 = x[5]

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
        s.add(L1_norm(pred_1, pred_2) > 0.5)
        # s.add(Abs(pred_1 - pred_2) > 0.5) ## L1



    s.set('timeout', 600000)
    check = s.check()
    
    if check == unknown:
        print('Result:', s.reason_unknown())
        exec_time = time.time() - start_time
        print('Execution time: {:.2f}'.format(exec_time))
        print('---------------------------------------------')
        pass
    else:
        print('Result:', check)
        exec_time = time.time() - start_time
        print('Execution time: {:.2f}'.format(exec_time))
        print('---------------------------------------------')
    # return check, exec_time



if __name__ == "__main__":

    
    mpc_30 = 'models/Equivalence_Checking/nn_mpc/NN_controller30.hdf5'
    mpc_35 = 'models/Equivalence_Checking/nn_mpc/NN_controller35.hdf5'
    mpc_40 = 'models/Equivalence_Checking/nn_mpc/NN_controller40.hdf5'
   

    calc_experiment(mpc_30, mpc_35, "strict", "mpc")
    calc_experiment(mpc_30, mpc_40, "L1", "mpc")
    calc_experiment(mpc_35, mpc_40, "strict", "mpc")
    calc_experiment(mpc_30, mpc_35, "L1", "mpc")
    calc_experiment(mpc_30, mpc_40, "strict", "mpc")
    calc_experiment(mpc_35, mpc_40, "L1", "mpc")
    
        

