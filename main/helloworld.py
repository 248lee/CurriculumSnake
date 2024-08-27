import torch
import torch.nn.functional as F

# Example tensor with shape (batch_size, num_of_logits)
logits = torch.tensor([[2.0, 1.0, 0.1], [0.5, 2.2, 3.3]])

# Convert logits to probabilities using softmax
probabilities = F.softmax(logits, dim=1)

print(probabilities)

# Sample an index according to the probability distribution for each batch
# torch.multinomial expects the input tensor to have probability values that sum up to 1 along the specified dimension
# num_samples=1 means we are sampling one index per batch
import numpy as np
table = np.zeros((2, 3))
for i in range(1000):
    samples = torch.multinomial(probabilities, num_samples=1)

    # If you want a 1-dimensional tensor of samples instead of 2D with size (batch_size, 1)
    samples = samples.squeeze()
    table[0, samples[0].item()] += 1
    table[1, samples[1].item()] += 1

print(table / 1000)

