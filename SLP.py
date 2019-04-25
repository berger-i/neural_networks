import numpy as np

class SLP(object):
    def __init__(self, num_of_input_neurons, num_of_output_neurons):        
        self.num_of_input_neurons = num_of_input_neurons
        self.num_of_output_neurons = num_of_output_neurons
        self.weights = np.random.rand(num_of_output_neurons, num_of_input_neurons+1) #+1 for bias        

    def __str__(self):
        out_str = 'num_of_input_neurons=' + str(self.num_of_input_neurons)
        out_str += '\nnum_of_output_neurons=' + str(self.num_of_output_neurons)
        out_str += '\nweights=\n' + str(self.weights)
        return out_str        

    def run(self, x):
        x_with_bias = np.vstack(([1], x))
        product = np.matmul(self.weights, x_with_bias)
        y = product.sum(axis=1)
        print('self.weights\n',self.weights)
        print('x\n',x)
        print ('product\n',product)
        print ('y\n',y)
        return y        

if __name__ == "__main__":
    num_of_input_neurons = 3 
    num_of_output_neurons = 2
    my_net = SLP(num_of_input_neurons, num_of_output_neurons)
    x = np.random.rand(num_of_input_neurons,1)
    print (str(my_net))
    my_net.run(x)