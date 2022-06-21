from __future__ import print_function

import random
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt

from z3 import *

## DNN to SMT encoding ##

def list_to_RealVal(x):
    if type(x[0]) == list:
        result = np.empty((len(x), len(x[0]))).tolist()
        for i in range(len(x)):
            for j in range(len(x[0])):
                result[i][j] = RealVal(x[i][j])
    else:
        result = []
        for i in range(len(x)):
            result.append(RealVal(x[i]))
    return result

# vector matrix multiplication    
def vec_mat_mul(x, y):
    if len(x) == len(y):
        results=[]
        for i in range(len(x[0])):
            item=0
            count=0
            for j in range(len(y)):
                item += x[j][i] * y[j]
                count += 1
                #print(count)
            results.append(item)
        return results
    else: 
        return "The matrix and vector aren't compatible."
        
# vector - matrix addition
def vec_add(x, y):
    if type(x[0]) == list:
        results = []
        for i in range(len(x)):
            rows = []
            for j in range(len(y[0])):
                #item = 0
                for k in range(len(x[0])):
                    item = x[i][k] + y[k][j]
                rows.append(item)
            results.append(rows)
        return results
    
    else:
        results = []
        for i in range(len(x)):
            item = x[i] + y[i]
            results.append(item)
        return results

def relu(x):
    return If(x > 0, x, 0)

def Relu(x):
    return list(map(relu, x))

def hardtanh(x):
    return If(x > 1, 1, If(x < -1, -1, x))

def hardTanh(x):
    return list(map(hardtanh, x))

def is_max(x):  
    x_max = RealVal(-1)
    for i in range(len(x)):
        x_max = If(x[i] > x_max, x[i], x_max) 
    return x_max                               

def argmax(x):  
    i_max = RealVal(-1)
    x_max = RealVal(-100)
    for i in range(len(x)-1, -1, -1):
        x_max = If(x[i] > x_max, x[i], x_max)
        i_max = If(x[i] == x_max, i, i_max)
    return i_max

def arg_max(x):
    x_max = RealVal(-1)
    i_max = IntVal(-1)
    for i in range(len(x)):
        x_max = If(x[i] > x_max, x[i], x_max)
        i_max = If(x[i] == x_max, i, i_max)
    return i_max

def argmaxis(x, index):
    const1=[]
    const2=[]
    const=[]
    i=index
    j=0
    while j < i:    
        const1.append(x[i] > x[j])
        j+=1
    k = i+1
    while k < len(x):
        const2.append(x[i] >= x[k])
        k+=1
    const = const1 + const2
    return const

def argmax_eq(x, y):
    arg=[]
    for i in range(len(x)):
        for j in range(len(y)):
            if i != j:
                arg.append(And(argmaxis2(x, i) + argmaxis2(y, j)))
    return arg

def Softmax(x):
    e = RealVal(2.718281)
    e_sum=0
    softmax = []
    for i in range(len(x)):
        e_sum += e**x[i]
    for i in range(len(x)):
        softmax.append(e**x[i] / e_sum)
    return softmax

def Abs_f(x):
    return If(x <= 0, -x, x)

def Abs(x):
    return list(map(Abs_f, x))

def Lp_norm(x, y, p):
    sub = []
    norm = []
    for i in range(len(x)):
        sub.append(x[i] - y[i])
    
    absolute = Abs(sub)
    for i in range(len(x)):
        norm.append(absolute[i]**p)
    
    sum_norm = sum(norm)
    res = sum_norm ** (1/p)
    return res

def Lp_norm2(x, y, p):
    sub = []
    norm = []
    for i in range(len(x)):
        sub.append(x[i] - y[i])
    
    absolute = Abs(sub)
    for i in range(len(x)):
        norm.append(absolute[i]**p)
    
    sum_norm = sum(norm)
    return sum_norm

def L1_norm(x, y):
    sub = []
    norm = []
    for i in range(len(x)):
        sub.append(x[i] - y[i])
    
    absolute = Abs(sub)
    return sum(absolute)

def L2_norm(x, y):
    sub = []
    norm = []
    for i in range(len(x)):
        sub.append(x[i] - y[i])
    
    absolute = Abs(sub)
    for i in range(len(x)):
        norm.append(absolute[i] * absolute[i])
    
    return sum(norm)

def Linf_norm(x, y):
    sub = []
    for i in range(len(x)):
        sub.append(x[i] - y[i])
        
    norm = is_max(Abs(sub))
    return norm