"""
A simple attention mechanisme without trainable weights.
https://www.youtube.com/watch?v=2rluVS_ap9M&list=PLQRyiBCWmqp5twpd8Izmaxu5XRkxd5yC-&index=9
"""
import torch

inputs = torch.tensor(
    [[0.43, 0.15, 0.89], 
     [0.55, 0.87, 0.66], 
     [0.57, 0.85, 0.64], 
     [0.22, 0.58, 0.33], 
     [0.77, 0.25, 0.10], 
     [0.05, 0.80, 0.55]])

# Step 1) Compute the attention scores (in respect to input 2)
#2nd input token is the query
query = inputs[1]
attn_scores_2 = torch.empty(inputs.shape[0])
for idx, input_i in enumerate(inputs):
    attn_scores_2[idx] = torch.dot(input_i, query)

print(attn_scores_2)

# Step 2) Compute the attention weights by normalizing the attention scores.
attn_weights_2 = torch.softmax(attn_scores_2, dim=0)
print(f"attn_weights_2: {attn_weights_2}")

# Step 3) Compute the context vector z²
context_vec_2 = torch.zeros(query.shape)
for i,x_i in enumerate(inputs):
    context_vec_2 += attn_weights_2[i]*x_i 

print(f"context_vec_2: {context_vec_2}")