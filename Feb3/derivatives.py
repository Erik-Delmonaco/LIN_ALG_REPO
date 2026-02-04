import torch

x = torch.tensor(3.0,requires_grad = True)
y = torch.tensor(-2.0,requires_grad = True)
z = torch.tensor(-3.0,requires_grad = True)
f = (3*z**2 + 4*x**3*y**3 + y**3*z + z*x**3*y) / (3*z**2*y**2 + 3*x*z**2 + 5)
f.backward()
print(x.grad)
print(y.grad)
print(z.grad)