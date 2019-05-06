import unittest
import numpy as np
import SLP

class Test_SLP(unittest.TestCase):
    def generate_learning_sample_sine(self, sample_size):
        learning_sample = []
        for _ in range(sample_size):
            x = np.random.random(1) * np.pi * 2 - np.pi
            y = np.sin(x)
            learning_sample.append((x, y))
        return learning_sample

    def generate_learning_sample_identity(self, sample_size):
        learning_sample = []
        for _ in range(sample_size):
            x = np.random.random(1)
            learning_sample.append((x, x))
        return learning_sample


    def generate_learning_sample_and(self, sample_size):
        learning_sample = []
        for _ in range(sample_size):
            x1 = np.random.random_integers(0, 1)
            x2 = np.random.random_integers(0, 1)
            x = np.array([[x1], [x2]])
            y = x1 & x2
            learning_sample.append((x, y))
        return learning_sample

    def test_sine(self):
        #learn sine function
        num_of_input_neurons = 1
        num_of_output_neurons = 1
        my_net = SLP.SLP(num_of_input_neurons, num_of_output_neurons)

        learning_sample = self.generate_learning_sample_sine(1000)
        testing_sample = self.generate_learning_sample_sine(1000)

        before = my_net.calculate_error_function(testing_sample)
        my_net.learn(learning_sample)
        after = my_net.calculate_error_function(testing_sample)

        print(before, after)
        print(my_net.weights)

        self.assertGreater(before, after)

    def test_identity(self):
        num_of_input_neurons = 1
        num_of_output_neurons = 1
        my_net = SLP.SLP(num_of_input_neurons, num_of_output_neurons)

        learning_sample = self.generate_learning_sample_identity(100000)
        testing_sample = self.generate_learning_sample_identity(100000)

        before = my_net.calculate_error_function(testing_sample)
        my_net.learn(learning_sample)
        after = my_net.calculate_error_function(testing_sample)

        print(before, after)
        print(my_net.weights)

        self.assertGreater(before, after)

    def test_and(self):
        num_of_input_neurons = 2
        num_of_output_neurons = 1
        my_net = SLP.SLP(num_of_input_neurons, num_of_output_neurons, 'heaviside')

        learning_sample = self.generate_learning_sample_and(1000)
        testing_sample = self.generate_learning_sample_and(1000)

        before = my_net.calculate_error_function(testing_sample)
        my_net.learn(learning_sample)
        after = my_net.calculate_error_function(testing_sample)


        self.assertGreater(before, after)

if __name__ == '__main__':
    unittest.main()
