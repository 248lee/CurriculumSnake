import torch as th
import numpy as np
x = th.tensor(20.0, requires_grad=True)
tau = 2.0 * x
tau.retain_grad()
print(tau.grad)
y = 3.0 * tau
y.retain_grad()
print(y)
tau2 = 5269.0 * x
tau2.retain_grad()
z = 9487.0 * tau2
z.retain_grad()
y.backward()
print(tau.grad)
print(x.grad)
print(y.grad)