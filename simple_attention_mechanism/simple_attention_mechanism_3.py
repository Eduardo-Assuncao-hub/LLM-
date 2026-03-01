"""
A simple self-attention mechanism with trainable weights:

Implemente the model weight parameters that will be optimized during the model training. This is the way the real self-attention mechanism works.
- The goal is still to compute the context vector.

https://www.youtube.com/watch?v=2-PYMkJ0OxY&list=PLQRyiBCWmqp5twpd8Izmaxu5XRkxd5yC-&index=12

"""

import torch
from dataset_utils import data_inputs

#Note: He we still computing the context vector in respect to input 2. In reality, we need to cumpute the context vector
# for each of thr inputs.
inputs = data_inputs()

#Defining place holds
x_2 = inputs[1] # input 2
d_in = inputs.shape[1]
print("d_in: ", d_in)
d_out = 2 #Output embbeding size. The same dimension of context vector.

#Generate the weights parameters
torch.manual_seed(123)
# The first way of performing weights parameters. For this we use torch parameter that is a wrapper around the tensor in Pytorch.
W_query = torch.nn.Parameter(torch.rand(d_in, d_out))
print(f"W_query: {W_query}")
W_key = torch.nn.Parameter(torch.rand(d_in, d_out))
print(f"W_query: {W_key}")
W_value = torch.nn.Parameter(torch.rand(d_in, d_out))
print(f"W_query: {W_value}")

# Now compute the query in respect to second input
query_2 = x_2 @ W_query
# Compute the keys and values
keys = inputs @ W_key
values = inputs @ W_value 
print(f"keys shape: {keys.shape}")
print(f"keys: {keys}")

#Now lets cumpute the attention scores.
att_scores_2 = query_2 @ keys.T
# Computs the attention weights
d_k = keys.shape[1]
att_weights_2 = torch.softmax(att_scores_2 / d_k**0.5, dim=-1)
print(f"att_weights_2: {att_weights_2}") 
#Compute the context vector in respect to input 2
context_vec_2 = att_weights_2 @ values