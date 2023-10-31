import torch
x = torch.ones(5)
y = torch.zeros(3)
w = torch.rand(5,3,requires_grad=True)
b = torch.rand(3,requires_grad=True)
z = torch.matmul(x,w)+b
loss = torch.nn.functional.binary_cross_entropy_with_logits(z,y)

print(f"Gradient function for z = {z.grad_fn}")
print(f"Gradient function for loss = {loss.grad_fn}")


