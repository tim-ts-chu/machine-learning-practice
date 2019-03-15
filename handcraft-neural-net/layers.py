
import abc
import numpy as np
import mathutil

class Layer(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def inference(self, input_vects):
        pass
    @abc.abstractmethod
    def forward(self, input_vects):
        pass
    @abc.abstractmethod
    def backward(self, input_vects):
        pass
    @abc.abstractmethod
    def update(self, weight_gradient, bias_gradient, learning_rate):
        pass
    @abc.abstractmethod
    def dump(self):
        pass

class HiddenLayer(Layer):
    def __init__(self, input_dim, hidden_dim):
        self._input_dim = input_dim
        self._hidden_dim = hidden_dim
        self._weight = 0.1*np.random.randn(input_dim, hidden_dim)
        self._bias = 0.1*np.random.randn(1, hidden_dim)
        self._activation = mathutil.sigmoid
        self._activation_derivative = mathutil.derivative_sigmoid

    def inference(self, input_vects):
        #print(input_vects)
        affine = np.dot(input_vects, self._weight) + self._bias
        return (affine, self._activation(affine))

    def forward(self, input_vects):
        # forward path are inputs, so just return it
        # dimension n*k
        return input_vects

    def backward(self, prop_backward, activation_input):
        # pre_back_path is an n*h array
        layer_backward = prop_backward * self._activation_derivative(activation_input)
        prop_backward = np.dot(layer_backward, self._weight.T)
        return layer_backward, prop_backward
        # also return an n*h array

    def update(self, weight_gradient, bias_gradient, learning_rate):
        self._weight -= learning_rate*weight_gradient
        self._bias -= learning_rate*bias_gradient

    def dump(self):
        print('weight:\n', self._weight)
        print('bias:\n', self._weight)

class OutputLayer(Layer):
    def __init__(self, hidden_dim, output_dim):
        self._hidden_dim = hidden_dim
        self._output_dim = output_dim
        self._weight = 0.1*np.random.randn(hidden_dim, output_dim)
        self._bias = 0.1*np.random.randn(1, output_dim)
        self._normalizer = mathutil.soft_max

    def inference(self, input_vects):
        affine = np.dot(input_vects, self._weight) + self._bias
        return affine, self._normalizer(affine)

    def forward(self, input_vects):
        # forward path are inputs, so just return it
        # return an n*k array
        return input_vects

    def backward(self, predict_vects, target_vects):
        # swc means softmax with crossentropy
        # return an n*h array
        layer_backward = predict_vects - target_vects
        prop_backward = np.dot(layer_backward, self._weight.T)
        return layer_backward, prop_backward

    def update(self, weight_gradient, bias_gradient, learning_rate):
        self._weight -= learning_rate*weight_gradient
        self._bias -= learning_rate*bias_gradient

    def dump(self):
        print('weight:\n', self._weight)
        print('bias:\n', self._weight)




