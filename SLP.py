import numpy as np

class SLP(object):
    def __init__(self, num_of_input_neurons, num_of_output_neurons, activation=None):        
        self.num_of_input_neurons = num_of_input_neurons
        self.num_of_output_neurons = num_of_output_neurons
        self.activation = activation

        self.weights = np.random.rand(num_of_output_neurons, num_of_input_neurons+1) #+1 for bias        

    def __str__(self):
        out_str = 'num_of_input_neurons=' + str(self.num_of_input_neurons)
        out_str += '\nnum_of_output_neurons=' + str(self.num_of_output_neurons)
        out_str += '\nweights=\n' + str(self.weights)
        return out_str        

    def learn(self, learning_sample):
        # delta_w_i = ni * o_i * delta_omega
        ni = 0.01
        for x, y in learning_sample:
            network_output = self.run(x)
            delta_omega = network_output - y
            delta_w_bias = ni * np.array([[1]]) * delta_omega
            delta_w_data = np.transpose(ni * x * delta_omega)
            if delta_w_data.shape == (1,):
                delta_w_data = [delta_w_data]
            delta_w_total = np.hstack((delta_w_bias,delta_w_data))
            self.weights -= delta_w_total
        pass

    def calculate_error_function(self, testing_sample):
        sum_squares = 0
        for x, y in testing_sample:
            network_output = self.run(x)
            sum_squares += (network_output - y) ** 2
        return sum_squares/len(testing_sample)

    def run(self, x):
        x_with_bias = np.vstack(([1], x))
        product = np.matmul(self.weights, x_with_bias)
        y = product.sum(axis=1)
        if self.activation == 'heaviside':
            y = np.heaviside(y, 0)


        #print('self.weights\n',self.weights)
        #print('x\n',x)
        #print ('product\n',product)
        #print ('y\n',y)
        return y        

def generate_learning_sample_sine(sample_size):
    learning_sample = []
    for _ in range(sample_size):
        x = np.random.random(1) * np.pi * 2 - np.pi
        y = np.sin(x)
        learning_sample.append((x,y))
    return learning_sample

def generate_learning_sample_identity(sample_size):
    learning_sample = []
    for _ in range(sample_size):
        x = np.random.random(1)         
        learning_sample.append((x,x))
    return learning_sample

def generate_learning_sample_and(sample_size):
    learning_sample = []
    for _ in range(sample_size):
        x1 = np.random.random_integers(0,1)
        x2 = np.random.random_integers(0,1)
        x = np.array([[x1],[x2]])
        y = x1 & x2        
        learning_sample.append((x,y))
    return learning_sample

def test_sine():
    #learn sine function
    num_of_input_neurons = 1
    num_of_output_neurons = 1
    my_net = SLP(num_of_input_neurons, num_of_output_neurons)

    learning_sample = generate_learning_sample_sine(1000)
    testing_sample = generate_learning_sample_sine(1000)
    
    before = my_net.calculate_error_function(testing_sample)
    my_net.learn(learning_sample)
    after = my_net.calculate_error_function(testing_sample)
    
    print (before, after)
    print (my_net.weights)

def test_identity():
    num_of_input_neurons = 1
    num_of_output_neurons = 1
    my_net = SLP(num_of_input_neurons, num_of_output_neurons)

    learning_sample = generate_learning_sample_identity(100000)
    testing_sample = generate_learning_sample_identity(100000)
    
    before = my_net.calculate_error_function(testing_sample)
    my_net.learn(learning_sample)
    after = my_net.calculate_error_function(testing_sample)
    
    print (before, after)
    print (my_net.weights)

def test_and():
    num_of_input_neurons = 2
    num_of_output_neurons = 1
    my_net = SLP(num_of_input_neurons, num_of_output_neurons, 'heaviside')

    learning_sample = generate_learning_sample_and(1000)
    testing_sample = generate_learning_sample_and(1000)
    
    before = my_net.calculate_error_function(testing_sample)
    my_net.learn(learning_sample)
    after = my_net.calculate_error_function(testing_sample)
    
    print (before, after)
    print (my_net.weights)



if __name__ == "__main__":    
    #test_sine()
    test_identity()
    #test_and()
