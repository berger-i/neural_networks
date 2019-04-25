import numpy as np

class SLP_explicit_bias(object):
    def __init__(self, num_of_input_neurons, num_of_output_neurons):        
        self.num_of_input_neurons = num_of_input_neurons
        self.num_of_output_neurons = num_of_output_neurons
        self.weights = np.random.rand(num_of_output_neurons, num_of_input_neurons)
        self.bias = np.random.rand(num_of_output_neurons,1)

    def __str__(self):
        out_str = 'num_of_input_neurons=' + str(self.num_of_input_neurons)
        out_str += '\nnum_of_output_neurons=' + str(self.num_of_output_neurons)
        out_str += '\nweights=\n' + str(self.weights)
        out_str += '\nbias=\n' + str(self.bias)
        return out_str        

    def run(self, x):
        #cross_product = np.cros(self.weights, x) 
        #x = np.transpose(x)
        #self.weights = np.transpose(self.weights)

        product = np.matmul(self.weights, x)
        product_with_bias =  product + self.bias
        y = product_with_bias.sum(axis=1)
        print('self.weights\n',self.weights)
        print('x\n',x)
        print('self.bias\n',self.bias)
        print ('product\n',product)
        print ('product_with_bias\n',product_with_bias)
        print ('y\n',y)
        return y
    pass    


if __name__ == "__main__":
    num_of_input_neurons = 3 
    num_of_output_neurons = 2
    my_net = SLP_explicit_bias(num_of_input_neurons, num_of_output_neurons)
    x = np.random.rand(num_of_input_neurons,1)
    print (str(my_net))
    my_net.run(x)