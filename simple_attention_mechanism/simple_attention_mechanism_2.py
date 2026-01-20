"""
A simple self-attention mechanism without trainable weights:

Generalization of the simple_attention_mechanism_1 by computing the attention weights for all other inputs (tokens). In the simple_attention_mechanism_1,
we just computed the attention weights in respect to the input 2.
Reference: https://www.youtube.com/watch?v=Dj1fjQNQl2g&list=PLQRyiBCWmqp5twpd8Izmaxu5XRkxd5yC-&index=10
"""
import torch
from dataset_utils import data_inputs
#Generate the attenction scores matrix (container to stores the attention scores wights)
attn_scores = torch.empty(6, 6)

#Generate the data to be used in this task
inputs = data_inputs()
print(f"inputs: {inputs}", )

#Iterate in the input matrix
#Note: Using for loop is not an efficient way
for i, x_i in enumerate(inputs):
    for j, x_j in enumerate(inputs):
        attn_scores[i, j] = torch.dot(x_i, x_j)

print(f"attn_scores: {attn_scores}")

# Calculate the attention scores in an efficient way
attn_scores = inputs@inputs.T
print(f"attn_scores: {attn_scores}")
attn_weights = torch.softmax(attn_scores, dim=1)
print(f"attn_weights: {attn_weights}")

# Calculate the context vectors
all_context_vecs = attn_weights@inputs
print(f"all_context_vecs: {all_context_vecs}")