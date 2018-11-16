import numpy as np

# creating first 2 neural network
input_data=np.array([2,3])

# assining the weightage to tha each node
weights={
    'node_0':np.array([1,1]),
    'node_1':np.array([-1,1]),
    'output':np.array([2,-1])
}
# calculating the first node value
node_0_value=(input_data*weights['node_0']).sum()
node_1_value=(input_data*weights['node_1']).sum()

# calculating first hidden node value
hidden_layer_values=np.array([node_0_value,node_1_value])

print("At the hidden layer nodes value are ",hidden_layer_values)
# calculating the final output with g=hidden layer
output=(hidden_layer_values*weights['output']).sum()

print("Final value after calculating ",output)