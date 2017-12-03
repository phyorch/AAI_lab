import numpy as np
class Layer(object):
    def __init__(self, layer_order, layer_size):
        self.size = layer_size
        self.order = layer_order
        self.node = 3 * np.random.randn(self.size, 1)
        self.error = np.random.randn(self.size, 1)
        self.bias = 3 * np.random.randn(self.size, 1)
    def Theta_init(self, next_layer_size):
        self.theta = np.random.randn(next_layer_size, self.size)

def activation(x):
    x = 1 / (1 + np.exp(-x))
    return x

def FP_process(Network, input):# input is array
    for l in range(len(Network)):
        layer = Network[l]
        if layer.order == 0:  # if we are calculating the input layer to seconde layer
            # node_matrix.append(input)
            layer.node = input
        else:  # we use previous node array to calculate the new one
            layer = Network[l]
            prior = Network[l - 1]
            layer.node = activation(np.dot(prior.theta, prior.node) + layer.bias)
    return Network


def BP_process(Network, y):# output is one of the dataset's y vector
    for l in range(len(Network)-1, -1, -1):
        if l == len(Network) - 1:
            layer = Network[l]
            layer.error = (layer.node - y) * layer.node * (1 - layer.node)
        else:
            layer = Network[l]
            post_layer = Network[l + 1]
            layer.error = np.dot(layer.theta.T, post_layer.error) * layer.node * (1 - layer.node)  # back propagation
    return Network

def Network_initail(size, layer_num=3):
    Network = []
    for i in range(layer_num):
        layer = Layer(i, size[i])
        if i < layer_num - 1:
            layer.Theta_init(size[i + 1])
        Network.append(layer)
    return Network




'''data = pd.read_csv('C:/Users/Phyorch/Desktop/Learning/Mchine learning/project and homework/lab3/train.csv')
data.as_matrix
data = np.array(data)
labels = data[:, 0]
labels_network = code(labels)
data = data[:, 1:]
X = data[:39000, :]/255
Y = labels_network[:39000, :]
sizelist = [784, 30, 10]
test_X = data[39000:42000, :]/255
test_Y = labels[39000:42000]

Network = Network_initail(sizelist)
Network = Training(Network, X, Y)
right, accuracy, predict = Test(Network, test_X, test_Y)'''


#images shape = (50000,32,32,3)
def train(images, labels):
    learning_rate = 0.1
    sizelist = [32*32*3, 50, 10]
    Network = Network_initail(sizelist)
    for m in range(len(images)):
        image = images[m]
        #grey = np.sum(image, axis=2)
        y = labels[m]
        image = image.flatten()  # the RGB image is transfered to 3 layer of monochromatic image in a row array
        image.shape = (image.shape[0], 1)
        y.shape = (y.shape[0], 1)
        Network = FP_process(Network, image)
        Network = BP_process(Network, y)
        for l in range(len(Network) - 1):
            ll = l + 1
            layer = Network[l]
            post_layer = Network[ll]
            layer.theta -= learning_rate * np.dot(post_layer.error, layer.node.T)
            post_layer.bias -= learning_rate * post_layer.error
    return Network
    #raise NotImplementedError


#images shape = (10000,32,32,3)
def predict(images, idx, Network):
    prediction = []
    correct = 0
    for m in range(len(images)):
        image = images[m]
        y = idx[m]
        image = image.flatten()
        image.shape = (image.shape[0], 1)
        Network = FP_process(Network, image)
        evaluate = Network[-1].node
        evaluate.shape = (1, evaluate.shape[0])
        evaluate = evaluate[0]
        evaluate = evaluate.tolist()
        values = evaluate.index(max(evaluate)) # find the position of the max value of the output , the posistion is the class of the image
        prediction.append(values)
        if values == idx[m]:
            correct += 1
    accuracy = correct / len(idx)
    prediction = np.array(prediction)
    return prediction#, correct, accuracy
    # Return a Numpy ndarray with the length of len(images).
    # e.g. np.zeros((len(images),), dtype=int) means all predictions are 'airplane's
    # raise NotImplementedError
