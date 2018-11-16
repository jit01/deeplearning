import numpy as np


# to load data in neural network
input_data=np.array([-1,2])
weights={
    'node_0':np.array([3,5]),
    'node_1':np.array([1,5]),
    'output':np.array([2,-1])
}

# to find the first node output value
node_0_input=(input_data*weights['node_0']).sum()
node_0_output=np.tanh(node_0_input)

# To find second node output value
node_1_input=(input_data*weights['node_1']).sum()
node_1_output=np.tanh(node_1_input)

# to find the hidden layer value
hidden_layer_outputs=np.array(node_0_output,node_1_output)

print("At hidden layer",hidden_layer_outputs)

output=(hidden_layer_outputs*weights['output']).sum()

print("output",output)