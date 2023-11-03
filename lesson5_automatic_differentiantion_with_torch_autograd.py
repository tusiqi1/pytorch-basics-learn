import torch
x = torch.ones(5)
y = torch.zeros(3)
w = torch.rand(5,3,requires_grad=True)
b = torch.rand(3,requires_grad=True)
z = torch.matmul(x,w)+b
loss = torch.nn.functional.binary_cross_entropy_with_logits(z,y)

print(f"Gradient function for z = {z.grad_fn}")
print(f"Gradient function for loss = {loss.grad_fn}")
# (1)
# We can only obtain the “grad” properties for the leaf nodes of the computational graph, which have
# “requires_grad” property set to True. For all other nodes in our graph, gradients will not be available.
# (2)
# We can only perform gradient calculations using backward once on a given graph, for performance reasons.
# If we need to do several backward calls on the same graph, we need to pass “retain_graph=True” to the
# backward call.

loss.backward()
print(w.grad)
print(b.grad)

#  when we have trained the model and just want to apply it to some input data, i.e.:
#  (1) we only want to do forward computations through the network.
#  (2) To speed up computations when you are only doing forward pass, because computations on tensors that do not track gradients would be more efficient.
#  We can stop tracking computations by surrounding our computation code with "torch.no_grad()" block:

z = torch.matmul(x, w)+b
print(z.requires_grad)
# close grad method 1
with torch.no_grad(): # use with keyword, return object's function "__enter__()" is called, and the return value of this method is assigned to the variable after as.
    z = torch.matmul(x, w)+b
    # When all the code blocks following with have been executed, the __exit__() method of the previously returned object is called.
print(z.requires_grad)
# close grad method 2
z = torch.matmul(x, w)+b
print(z.requires_grad)
z_det = z.detach()
print(z_det.requires_grad)
# 注：反向传播时是通过一个有向的无环图计算的，每次进行反向传播时该有向无环图会重新构建，基于这个才实现了控制流语句，并且让用户可以在每次迭代中改变形状、大小和操作


inp = torch.eye(4, 5, requires_grad=True)
out = (inp+1).pow(2).t()
out.backward(torch.ones_like(out), retain_graph=True)
print(f"First call\n{inp.grad}") # 2(inp+1)
out.backward(torch.ones_like(out), retain_graph=True)
print(f"\nSecond call\n{inp.grad}") # 2(inp+1) + 2(inp_2+1) accumulates the gradients
inp.grad.zero_() # ------ 这句话暂未明白
out.backward(torch.ones_like(out), retain_graph=True)
print(f"\nCall after zeroing gradients\n{inp.grad}")
