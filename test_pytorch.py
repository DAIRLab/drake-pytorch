import pydrake.symbolic as sym
import torch
import drake_pytorch
import numpy as np

x = sym.MakeVectorVariable(5,'x')
y = sym.MakeVectorVariable(3,'y')
z1 = sym.MakeVectorVariable(2,'z1')
z2 = sym.MakeVectorVariable(2,'z2')
z = np.vstack([z1, z2])
tx = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0], requires_grad=True)
ty = torch.tensor([-1.0, 2.5, -.5], requires_grad=True)
tz = torch.eye(2)

expr_1 = x[0] + sym.abs(x[0])/(x[3] + x[0]) + sym.sin(x[0] + 1) + x[1] + sym.atan2(x[0], x[4]) + sym.floor(x[3])**2
expr_2 = sym.log(x[3]) + x[4]*x[2] / sym.exp(expr_1) + x[0]**2 + sym.atan(x[1]) + sym.min(x[0], x[1]+x[2])
expr_3 = x[0]*y[2] + sym.sin(x[1] + y[0])*y[1]
expr_4 = z1[0] + z1[1] + z2[0] + z2[1]

expr_list = [expr_1, expr_2]
expr_all = np.array([expr_1, expr_2, expr_3])

# scalar expression
[func, string] = drake_pytorch.sym_to_pytorch(expr_1, x, y)
print(string)
print(func(tx))

# list expression
[func, string] = drake_pytorch.sym_to_pytorch(expr_list, x)
print(string)
print(func(tx))

# batched call
tx_batch = tx.clone().unsqueeze(0).repeat(10, 1)
print(func(tx_batch))

# simple matrix variable
[func, string] = drake_pytorch.sym_to_pytorch(expr_4, z)
print(string)
print(func(tz))

# matrix variable with matrix batch
tz_batch = tz.clone().unsqueeze(0).repeat(2, 1, 1).unsqueeze(0).repeat(4, 1, 1, 1)
print(func(tz_batch))