# drake-pytorch
A simple library to convert Drake symbolic expressions to PyTorch functions,
which can then be evaluated and differentiated on PyTorch Tensors.

In the interest of efficiency, this is done by converting the symbolic
expression to a string, and then that string to a Python function.

This supports both single and vector (via Python lists) expressions, which can
be converted to a function of a variable number of arguments.

Requires torch and numpy to be locally installed.

## Example
```
import pydrake.symbolic as sym
import torch
import symbolic
import numpy as np

x = sym.MakeVectorVariable(2,'x')
y = sym.MakeVectorVariable(2,'y')

# expr is a length-2 list, depending on the x and y variables
expr = [x[0]*x[1], sym.sin(y[1]) * x[0]]

# Convert to a function with two arguments, (x,y)
[func, string] = symbolic.sym_to_pytorch(expr, x, y)

# Create PyTorch inputs and evaluate the function
x_torch = torch.tensor([1.0, 2.0], requires_grad=True)
y_torch = torch.tensor([3.0, 4.0], requires_grad=True)

result = func(x_torch, y_torch)
```