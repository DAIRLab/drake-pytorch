# drake-pytorch
A simple library to convert Drake symbolic expressions to PyTorch functions,
which can then be evaluated and differentiated on PyTorch Tensors.

In the interest of efficiency, this is done by converting the symbolic
expression to a string, and then that string to a Python function.

This supports both single and vector (via Python lists) expressions, where the
argument is a single vector.

Future extensions will support variable length arguments.