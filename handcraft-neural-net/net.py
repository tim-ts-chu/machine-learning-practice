
import numpy as np
import layers
import mathutil

class NeuralNet:
    def __init__(self, data_dim, target_dim, hidden_dims):

        self._hidden_layers = []
        self._output_layer = None

        previous_dim = data_dim
        for hidden_dim in hidden_dims:
            self._hidden_layers.append(layers.HiddenLayer(previous_dim, hidden_dim))
            previous_dim = hidden_dim

        self._output_layer = layers.OutputLayer(previous_dim, target_dim)

    def update(self, data_vects, target_vects, learning_rate):
        batch_size = len(data_vects)
        # forward path
        forward_path = []
        affine_path = []
        infer_vects = data_vects
        for layer in self._hidden_layers:
            forward_path.append(layer.forward(infer_vects))
            affine, infer_vects = layer.inference(infer_vects)
            affine_path.append(affine)

        forward_path.append(self._output_layer.forward(infer_vects))
        affine, infer_vects = self._output_layer.inference(infer_vects)
        affine_path.append(affine)

        # backward path
        backward_path = []
        layer_backward, prop_backward = self._output_layer.backward(infer_vects, target_vects)
        backward_path.append(layer_backward)
        for idx in range(len(self._hidden_layers)-1, -1, -1):
            layer_backward, prop_backward = self._hidden_layers[idx].backward(prop_backward, affine_path[idx])
            backward_path.append(layer_backward)
        backward_path.reverse()

        # calculate gradient and update
        for idx in range(0, len(self._hidden_layers)):
            weight_gradient = np.dot(forward_path[idx].T, backward_path[idx])/batch_size
            bias_gradient = np.sum(backward_path[idx], axis=0).reshape(1,-1)/batch_size
            self._hidden_layers[idx].update(weight_gradient, bias_gradient, learning_rate)

        weight_gradient = np.dot(forward_path[-1].T, backward_path[-1])/batch_size
        bias_gradient = np.sum(backward_path[-1], axis=0).reshape(1,-1)/batch_size
        self._output_layer.update(weight_gradient, bias_gradient, learning_rate)

    def predict(self, input_vects):
        previous_infer = input_vects 
        for layer in self._hidden_layers:
            affine, previous_infer = layer.inference(previous_infer)
        affine, result = self._output_layer.inference(previous_infer)
        return result

    def predict_encode(self, predict_vects):
        out = np.zeros_like(predict_vects)
        argmax = np.argmax(predict_vects, axis=1)
        for i in range(len(predict_vects)):
            out[i, argmax[i]] = 1.0
        return out

    def accuracy(self, predict_vects, target_vects):
        total_num = len(predict_vects)
        correct_num = 0.0
        for i in range(total_num):
            if np.array_equal(predict_vects[i], target_vects[i]):
                correct_num += 1.0
        return correct_num/total_num

    def loss(self, predict_vects, target_vects):
        ce = mathutil.cross_entropy(predict_vects, target_vects)
        return ce/len(predict_vects)

    def dump(self):
        for layer in self._hidden_layers:
            print('------')
            layer.dump()
        print('------')
        self._output_layer.dump()

        


