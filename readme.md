# Feed-forward Neural Network built ==without using a Machine Learning framework==

###1. Getting Started
- Download and install `jupyter-notebook`
- Clone the repo
- Run `jupyter-notebook` to launch the notebook.

###2. "Jest gimme the code!"


```python
import numpy as np
import scipy.special
import matplotlib.pyplot
```


```python
class neuralNetwork:
    def __init__(self, input_nodes, 
                 hidden_nodes, 
                 output_nodes, 
                 learning_rate):
        self.inodes = input_nodes
        self.hnodes = hidden_nodes
        self.onodes = output_nodes
        self.lr = learning_rate
        self.activation_function = lambda x:scipy.special.expit(x)
        
        # initialize and create weights
        self.wih = (np.random.rand(self.hnodes, self.inodes) - 0.5)
        self.who = (np.random.rand(self.onodes, self.hnodes) - 0.5)
        pass
    
    def train(self, inputs_list, targets_list):
        # convert inputs list to 2d array
        inputs = np.array(inputs_list, ndmin=2).T
        targets = np.array(targets_list, ndmin=2).T
                
        hidden_inputs = np.dot(self.wih, inputs) # calculate signals into hidden layer
        hidden_outputs = self.activation_function(hidden_inputs) #calculate the signals emerging from hidden layer
        
        final_inputs = np.dot(self.who, hidden_outputs) # calculate signals into final output layer
        final_outputs = self.activation_function(final_inputs) # calculate the signals emerging from final output layer
        
        output_errors = targets - final_outputs # error is the (target - actual)
        
        # hidden layer error is the output_errors, split by weights, recombined at hidden nodes
        hidden_errors = np.dot(self.who.T, output_errors)
        
        # update the weights for the links between the hidden and output layers
        self.who += self.lr * np.dot((output_errors * final_outputs * (1.0 - final_outputs)), np.transpose(hidden_outputs))
        
        # update the weights for the links between the input and hidden layers
        self.wih += self.lr * np.dot(hidden_errors * hidden_outputs * (1.0 - hidden_outputs), np.transpose(inputs))
        pass
    
    def query(self, inputs_list):
        inputs = np.array(inputs_list, ndmin=2).T # Convert inputs list to 2d array
        
        hidden_inputs = np.dot(self.wih, inputs) # Calculate signals into hidden layer
        hidden_outputs = self.activation_function(hidden_inputs) # Calculate the signals emerging from hidden layer.
        
        final_inputs = np.dot(self.who, hidden_outputs) # Calculate signals into final input layer
        final_outputs = self.activation_function(final_inputs) # Calculate the signals emerging from the final output layer
        
        return final_outputs
        pass
```


```python
input_nodes = 784
hidden_nodes = 100
output_nodes = 10
learning_rate = 0.3

n = neuralNetwork(input_nodes, hidden_nodes, output_nodes, learning_rate)
```


```python
training_data_file = open('data/mnist_train.csv', 'r')
training_data_list = training_data_file.readlines()
training_data_file.close()
```


```python
for record in training_data_list:
    all_values = record.split(',')
    inputs = (np.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
    targets = np.zeros(output_nodes) + 0.01
    targets[int(all_values[0])] = 0.99
    n.train(inputs, targets)
```


```python
test_data_file = open('data/mnist_test.csv', 'r')
test_data_list = test_data_file.readlines()
test_data_file.close()
```


```python
scorecard = [] # used to store performance
for record in test_data_list:
    all_values = record.split(',')
    correct_label = int(all_values[0])
    inputs = (np.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
    outputs = n.query(inputs)
    label = np.argmax(outputs)
    if (label == correct_label):
        scorecard.append(1)
    else:
        scorecard.append(0)
        
scorecard_array = np.asarray(scorecard);
print("Accuracy equals: ", scorecard_array.sum() / scorecard_array.size)
```

    Accuracy equals:  0.945
    
### 3. Credits
@makeyourfirstneuralnetwork gets credit for most of this code. He's got an amazing book called "Make Your Own Neural Network" that you should check out.

And last but not least, I must and will give due credit to **God**, the author of intelligence. I'm sure he chuckles under His breath while the AI community burns the midnight oil and gobs of GPU's trying to discover better ways to create intelligence.

