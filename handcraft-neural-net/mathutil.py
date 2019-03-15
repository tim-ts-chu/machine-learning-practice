
import numpy as np

def soft_max(input_vects):
    input_exp = np.exp(input_vects)
    return input_exp/np.sum(input_exp, axis=1).reshape(-1, 1)

def cross_entropy(predict_vects, target_vects):
    return -np.sum(np.log(predict_vects+1e-7)*target_vects)

def sigmoid(input_vects):
    return 1.0 / (1.0 + np.exp(-input_vects))

def derivative_sigmoid(input_vects):
    return sigmoid(input_vects)*(1-sigmoid(input_vects))
