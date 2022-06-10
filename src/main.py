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

import time

## Equivalence Checking
## Architecture 1 vs Architecture 2 --- under all equivalence relations

# load pretrained keras models
model_1 = tensorflow.keras.models.load_model("C:/Users/haris/Desktop/NN Equivalence Checking/BitVec models/bitvec_1_1.h5")
model_2 = tensorflow.keras.models.load_model("C:/Users/haris/Desktop/NN Equivalence Checking/BitVec models/bitvec_2_1.h5")

# unpack model parameters
NN1_w1, NN1_b1, NN1_w2, NN1_b2 = model_1.get_weights()
NN2_w1, NN2_b1, NN2_w2, NN2_b2, NN2_w3, NN2_b3 = model_2.get_weights()

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

# define input
## BitVec
x = [Real('x%s' % (i+1)) for i in range(10)]

## mnist
x = [Real('x%s' % (i+1)) for i in range(784)]

## mpc
x = [Real('x%s' % (i+1)) for i in range(6)]

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

# mpc
pred_1 = NN_controller_1(x)
pred_2 = NN_controller_2(x)

## input constraints ##

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

# alternatively 
#for i in range(len(x)):
 #   s.add(Or(0 == x[i], x[i] == 1))
    
# mnist
for i in range(len(x)):
    s.add(0 <= x[i], x[i] <= 1)
    
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

# strict equivalence
s.add(Not(pred_1 == pred_2))
#s.add(pred_1 != pred_2)

# ε-approximate equivalence
s.add(L1_norm(pred_1, pred_2) > 5)   ## L1
s.add(Linf_norm(pred_1, pred_2) > 10) ## Linf

## L2
s.add(y == z * z)
s.add(y >= 0)
s.add(y == L2_norm(pred_1, pred_2))
s.add(z > 10)

# argmax equivalence
s.add(Not(arg_max(pred_1) == arg_max(pred_2))) ## my_argmax
#s.add(argmax(pred_1) != argmax(pred_2))

# paper argmax
p = argmax_eq(pred_1, pred_2)
s.add(Or(p))

s.set('timeout', 600000)    
print(s.check())

#if s.check() == sat:
 #   print(s.model())

exec_time = time.time() - start_time
print('Execution time:', exec_time)