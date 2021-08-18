import numpy as np

class Layer():
        
    def __init__(self, model, f, d_f, input_dims = None, output_dims = None, input_layer=False, output_layer=False, learning_rate=0.001):
        
        self.model = model
        self.input_dims = input_dims
        self.output_dims = output_dims
        self.learning_rate = learning_rate
        
        # Parameters
        self.a = None
        self.z = None
        self.W = None
        self.b = None
        
        self.dW = None
        self.db = None
        self.da = None
        self.dz = None
        
        self.input_layer = input_layer
        self.output_layer = output_layer
        
        # Activation Functions
        self.f = f
        self.d_f = d_f
        
        # Adjacent layers set during backpropagation
        self.next_layer = None
        self.prev_layer = None
    
    
    def random_init(self):
        
        # Kaiming Weight Initialization
        self.W = np.random.randn(self.output_dims, self.input_dims)*np.sqrt(2)/np.sqrt(self.input_dims)
        
        # Xavier Weight Initialization
        self.b = np.zeros(shape=(self.output_dims, 1))
      
    
    def get_prev_a(self):
        if self.input_layer:
            return self.model.data
        return self.prev_layer.a
    

    def forward_pass(self):
        prev_a = self.get_prev_a()
        self.z = self.W.dot(prev_a) + self.b
        self.a = self.f(self.z)
        
    
    def backpropagate(self):
        prev_a = self.get_prev_a()
        
        if self.output_layer:
            delta = self.model.calc_d_J(self.a)
        else:
            delta = self.next_layer.da
            
        m = prev_a.shape[1]
        
        self.dz = delta * self.d_f(self.z)
        self.dW = self.dz.dot(prev_a.T)/m
        self.db = np.sum(self.dz, axis=1, keepdims=True)
        self.da = self.W.T.dot(self.dz)
        
    def learn(self):
        self.W = self.W - self.learning_rate * self.dW
        self.b = self.b - self.learning_rate * self.db

class NeuralNetwork():
    
    def __init__(self, architecture, input_size, cost_function, train_data=None, train_labels=None, learning_rate=0.001):
        
        self.learning_rate = learning_rate
        self.architecture = architecture
        self.cost_function = cost_function

        # Create Layers
        self.layers = self.create_layers(architecture, input_size)
        
        # Cost Function
        self.J, self.d_J = cost_functions[cost_function]
        
    
    def calc_J(self, y_hat):
        return self.J(self.labels, y_hat)
    
    
    def calc_d_J(self, y_hat):
        return self.d_J(self.labels, y_hat)
    
    
    def calc_accuracy(self, test_data, test_labels, error_func="MSE"):
        self.data = test_data
        self.labels = test_labels
        
        # Forward Pass and get output
        self.forward_pass()
        y_hat = self.layers[-1].a
        
        if error_func == "MSE":
            return np.sum((self.labels - y_hat)**2 ).squeeze() / (y_hat.shape[1]*2)
        elif error_func == "MAE":
            return np.sum(np.abs(y_hat-self.labels))/self.labels.shape[1]
        elif error_func == "RMSE":
            return np.sqrt(np.sum((self.labels - y_hat)**2 ).squeeze() / (y_hat.shape[1]*2))
        else:
            y_pred = np.where(y_hat > 0.5, 1, 0)
            return (y_pred == self.labels).mean()     
    
    def create_layers(self, architecture, input_size):
        
        layers = []
        
        for i, config in enumerate(architecture):
            input_dims = input_size if i == 0 else layers[-1].output_dims
            output_dims = config["num_nodes"]
            f, d_f = activation_functions[config["activation"]]
            layer = Layer(self, f, d_f, input_dims, output_dims, input_layer=(i==0), output_layer=(i==len(architecture)-1), learning_rate=self.learning_rate)
            
            if i != 0:
                layers[-1].next_layer = layer
                layer.prev_layer = layers[-1]
            
            
            layers.append(layer)
        
        for layer in layers:
            layer.random_init()
            
        return layers
    
    def add_data(self, train_data, train_labels):
        self.data = train_data
        self.labels = train_labels
        
    def forward_pass(self):
        for layer in self.layers:            
            layer.forward_pass()
            
    def backward_pass(self):
        for layer in reversed(self.layers):
            layer.backpropagate()

    def learn(self):
        for layer in self.layers:
            layer.learn()
    
    def train(self, epochs):
        history = []
        for i in range(epochs):
            self.forward_pass()
            cost = self.calc_J(self.layers[-1].a)
            history.append(cost)     
            self.backward_pass()
            self.learn()
        
        # Training done. Return history
        return history

# COST FUNCTIONS

def cross_entropy_sigmoid(y, y_hat):
    m = y.shape[1]
    cost = (1./m) * (-np.dot(y,np.log(y_hat).T) - np.dot(1-y, np.log(1-y_hat).T))
    cost = np.squeeze(cost)
    return cost


def cross_entropy_sigmoid_derivative(y, y_hat):
    return (-(np.divide(y, y_hat) - np.divide(1 - y, 1 - y_hat)))


def mean_squared(y, y_hat):
    return  np.sum((y - y_hat)**2 ).squeeze() / (y_hat.shape[1]*2)

def d_mean_squared(y, y_hat):
    return (y_hat - y)


cost_functions = {"cross_entropy_sigmoid" : (cross_entropy_sigmoid, cross_entropy_sigmoid_derivative),
                  "mean_squared" : (mean_squared, d_mean_squared)
                 }

# ACTIVATION FUNCTIONS

import numpy as np

def sigmoid(x):
    s = 1/(1+np.exp(-x))
    return s

def d_sigmoid(x):
    s = sigmoid(x)
    return s*(1-s)

def relu(x):
    return np.maximum(0,x)

def d_relu(x):
    r = np.where(x > 0, 1, 0)
    return r

def tanh(x):
    return np.tanh(x)

def d_tanh(x):
    d = tanh(x)
    return 1 - d*d


activation_functions = {"sigmoid" : (sigmoid, d_sigmoid) , "relu" : (relu, d_relu), "tanh" : (tanh, d_tanh)}