import numpy as np
import types
import torch
import pydrake.symbolic as sym

# Convert an expression to a function in pytorch. The arguments for the returned
# function are be `x[i]`, where index `i` is derived from the ordering
# in `sym_vars`. Returns [func, func_string]
# 
# @param expr Either a pydrake.symbolic.Expression, or a list of expressions
# @param sym_vars An ordered list of symbolic variables
# 
# @return func A function object, which can be evaluated on a pytorch Tensor
# @return A string expressing the entire python function
def sym_to_pytorch(expr, sym_vars):
	if isinstance(expr, sym.Expression):
		expr_string = sym_to_pytorch_string(expr, sym_vars)
		func_string = 'def my_func(x):\n  return ' + expr_string
	elif isinstance(expr, list) or isinstance(expr, np.ndarray):
		if isinstance(expr, np.ndarray):
			shape = expr.shape
			expr = np.reshape(expr, -1)
		else:
			shape = (len(expr), 1)
		func_string = 'def my_func(x):\n  ret = torch.tensor(['
		for expr_i in expr[:-1]:
			expr_string = sym_to_pytorch_string(expr_i, sym_vars)
			func_string = func_string + expr_string + ", "
		expr_string = sym_to_pytorch_string(expr[-1], sym_vars)
		func_string = func_string + expr_string + '])\n'
		func_string = func_string + '  return torch.reshape(ret, ' + str(shape) + ')'
		print(func_string)
	else:
		raise ValueError('expr must be a drake symbolic Expression or a list')

	code = compile(func_string, 'tmp.py', 'single')
	func = types.FunctionType(code.co_consts[0], globals())
	return func, func_string

# Convert a single expression to a string. The arguments for the returned
# expression will be `x[i]`, where index `i` is derived from the ordering
# in `sym_vars`
# 
# @param expr The pydrake.symbolic.Expression to convert
# @param sym_vars An ordered list of symbolic variables
# @return A string expressing the function f(x)
def sym_to_pytorch_string(expr, sym_vars):
	# If it's a float, just return the expression
	if isinstance(expr, float):
		return str(expr)

	# switch based on the expression kind
	kind = expr.get_kind()
	ctor, expr_args = expr.Unapply()
	if kind == sym.ExpressionKind.Constant:
		if len(expr_args) != 1:
			raise ValueError('Unexpected symbolic Constant of length != 1')
		return str(expr_args[0])
	elif kind == sym.ExpressionKind.Var:
		id_vectorized = np.vectorize(sym.Variable.get_id)
		id_vectorized(sym_vars)
		var_index = np.where(id_vectorized(sym_vars) == expr_args[0].get_id())[0]

		if len(var_index) == 0:
			raise ValueError('Expression contains variable not in sym_vars list')
		if len(var_index) > 1:
			raise ValueError('sym_vars contains duplicate variable')
		return 'x[' + str(var_index[0]) + ']'
	elif kind == sym.ExpressionKind.Add:
		ret = '(' + sym_to_pytorch_string(expr_args[0], sym_vars)
		for arg in expr_args[1:]:
			ret = ret + ' + ' + sym_to_pytorch_string(arg, sym_vars)
		return ret + ')'
	elif kind == sym.ExpressionKind.Mul:
		ret = '(' + sym_to_pytorch_string(expr_args[0], sym_vars)
		for arg in expr_args[1:]:
			ret = ret + ' * ' + sym_to_pytorch_string(arg, sym_vars)
		return ret + ')'
	elif kind == sym.ExpressionKind.Div:
		ret = '(' + sym_to_pytorch_string(expr_args[0], sym_vars) + ' / '
		ret = ret + sym_to_pytorch_string(expr_args[1], sym_vars)
		return ret + ')'
	elif kind == sym.ExpressionKind.Log:
		return sym_to_pytorch_simple_fun('log', expr_args[0], sym_vars)
	elif kind == sym.ExpressionKind.Abs:
		return sym_to_pytorch_simple_fun('abs', expr_args[0], sym_vars)
	elif kind == sym.ExpressionKind.Exp:
		return sym_to_pytorch_simple_fun('exp', expr_args[0], sym_vars)
	elif kind == sym.ExpressionKind.Pow:
		return 'torch.pow(' + sym_to_pytorch_string(expr_args[0], sym_vars) + \
			', ' + sym_to_pytorch_string(expr_args[1], sym_vars) + ')'
	elif kind == sym.ExpressionKind.Sin:
		return sym_to_pytorch_simple_fun('sin', expr_args[0], sym_vars)
	elif kind == sym.ExpressionKind.Cos:
		return sym_to_pytorch_simple_fun('cos', expr_args[0], sym_vars)
	elif kind == sym.ExpressionKind.Tan:
		return sym_to_pytorch_simple_fun('tan', expr_args[0], sym_vars)
	elif kind == sym.ExpressionKind.Asin:
		return sym_to_pytorch_simple_fun('asin', expr_args[0], sym_vars)
	elif kind == sym.ExpressionKind.Acos:
		return sym_to_pytorch_simple_fun('acos', expr_args[0], sym_vars)
	elif kind == sym.ExpressionKind.Atan:
		return sym_to_pytorch_simple_fun('atan', expr_args[0], sym_vars)
	elif kind == sym.ExpressionKind.Atan2:
		return 'torch.atan2(' + sym_to_pytorch_string(expr_args[0], sym_vars) + \
			', ' + sym_to_pytorch_string(expr_args[1], sym_vars) + ')'
	elif kind == sym.ExpressionKind.Sinh:
		return sym_to_pytorch_simple_fun('sinh', expr_args[0], sym_vars)
	elif kind == sym.ExpressionKind.Cosh:
		return sym_to_pytorch_simple_fun('cosh', expr_args[0], sym_vars)
	elif kind == sym.ExpressionKind.Tanh:
		return sym_to_pytorch_simple_fun('tanh', expr_args[0], sym_vars)
	elif kind == sym.ExpressionKind.Min:
		return 'torch.min(' + sym_to_pytorch_string(expr_args[0], sym_vars) + \
			', ' + sym_to_pytorch_string(expr_args[1], sym_vars) + ')'
	elif kind == sym.ExpressionKind.Max:
		return 'torch.max(' + sym_to_pytorch_string(expr_args[0], sym_vars) + \
			', ' + sym_to_pytorch_string(expr_args[1], sym_vars) + ')'
	elif kind == sym.ExpressionKind.Ceil:
		return sym_to_pytorch_simple_fun('ceil', expr_args[0], sym_vars)
	elif kind == sym.ExpressionKind.Floor:
		return sym_to_pytorch_simple_fun('floor', expr_args[0], sym_vars)
	else:
		raise ValueError('Unsupported symbolic: ' + str(kind))


# Helper function to convert a simple function of a single argument
def sym_to_pytorch_simple_fun(name, arg, sym_vars):
	return 'torch.' + name + '(' + sym_to_pytorch_string(arg, sym_vars) + ')'