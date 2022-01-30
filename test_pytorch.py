import pydrake.symbolic as sym
import torch
import drake_pytorch
import numpy as np

x = sym.MakeVectorVariable(5,'x')
y = sym.MakeVectorVariable(3,'y')
tx = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0], requires_grad=True)
ty = torch.tensor([-1.0, 2.5, -.5], requires_grad=True)

expr_1 = x[0] + sym.abs(x[0])/(x[3] + x[0]) + sym.sin(x[0] + 1) + x[1] + sym.atan2(x[0], x[4]) + sym.floor(x[3])**2
expr_2 = sym.log(x[3]) + x[4]*x[2] / sym.exp(expr_1) + x[0]**2 + sym.atan(x[1]) + sym.min(x[0], x[1]+x[2])
expr_3 = x[0]*y[2] + sym.sin(x[1] + y[0])*y[1]

expr_list = [expr_1, expr_2]
expr_all = np.array([expr_1, expr_2, expr_3])

[func, string] = drake_pytorch.sym_to_pytorch(expr_1, x, y)
print(string)
print(func(tx))

[func, string] = drake_pytorch.sym_to_pytorch(expr_list, x)
print(string)
print(func(tx))
