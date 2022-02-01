import numpy as np
import types
import torch
import pydrake.symbolic as sym

# Convert an expression to a function in pytorch. The arguments for the returned
# function are be `x[i]`, where index `i` is derived from the ordering
# in `sym_args`. Returns [func, func_string]
# 
# @param expr Either a pydrake.symbolic.Expression, or a list of expressions
# @param *sym_args The arguments to the symbolic expression. Multiple arguments
# 	can be provided. 
# 
# @return func A function object, which can be evaluated on a pytorch Tensor
# 	The function takes a variable number of arguments, matching the list in
#		sym_args
# @return A string expressing the entire python function
TORCH_ARGS = 't'

def sym_to_pytorch(expr, *sym_args):
	str_list = []
	if isinstance(expr, sym.Expression):
		str_list.append(f'def my_func(*{TORCH_ARGS}):\n  return ')
		str_list.extend(sym_to_pytorch_string(expr, *sym_args))
	elif isinstance(expr, list) or isinstance(expr, np.ndarray):
		if isinstance(expr, np.ndarray):
			shape = expr.shape
			nonbatch_dims = len(shape)
			expr = np.reshape(expr, -1)
		else:
			shape = (len(expr), 1)
		str_list.append(f'def my_func(*{TORCH_ARGS}):\n')
		# detect batch dimension list from first argument and assume the rest follow
		str_list.append(f'  batch_dims = {TORCH_ARGS}[0].shape[:-{len(sym_args[0].shape)}]\n')
		str_list.append('  ret = torch.cat((')
		for expr_i in expr:
			str_list.append("(")
			str_list.extend(sym_to_pytorch_string(expr_i, *sym_args))
			str_list.append(").unsqueeze(-1), ")
		str_list.append('), dim = -1)\n')
		str_list.append(f'  return torch.reshape(ret, batch_dims + {shape})')
	else:
		raise ValueError('expr must be a drake symbolic Expression or a list')
	func_string = ''.join(str_list)

	code = compile(func_string, 'tmp.py', 'single')
	func = types.FunctionType(code.co_consts[0], globals())
	return func, func_string

# Convert a single expression to a string. The arguments for the returned
# expression will be `x[i]`, where index `i` is derived from the ordering
# in `sym_args`
# 
# @param expr The pydrake.symbolic.Expression to convert
# @param sym_args An ordered list of symbolic variables
# @return A string expressing the function f(x)
def sym_to_pytorch_string(expr, *sym_args):
	# If it's a float, just return the expression
	if isinstance(expr, float):
		return str(expr)

	str_list = []

	# switch based on the expression kind
	kind = expr.get_kind()
	ctor, expr_args = expr.Unapply()
	if kind == sym.ExpressionKind.Constant:
		if len(expr_args) != 1:
			raise ValueError('Unexpected symbolic Constant of length != 1')
		return ['torch.tensor(', str(expr_args[0]), ')']
	elif kind == sym.ExpressionKind.Var:
		return [substitute_variable_string(expr_args[0], *sym_args)]
	elif kind == sym.ExpressionKind.Add:
		if type(expr_args[0]) == float and expr_args[0] == 0.0:
			if len(expr_args) <= 2:
				return [sym_to_pytorch_string(expr_args[1], *sym_args)]
			expr_args = expr_args[1:]
		str_list.append('(')
		str_list.extend(sym_to_pytorch_string(expr_args[0], *sym_args))
		for arg in expr_args[1:]:
			str_list.append(' + ')
			str_list.extend(sym_to_pytorch_string(arg, *sym_args))
		str_list.append(')')
		return str_list
	elif kind == sym.ExpressionKind.Mul:
		if type(expr_args[0]) == float and expr_args[0] == 1.0:
			if len(expr_args) <= 2:
				return [sym_to_pytorch_string(expr_args[1], *sym_args)]
			expr_args = expr_args[1:]
		str_list.append('(')
		str_list.extend(sym_to_pytorch_string(expr_args[0], *sym_args))
		for arg in expr_args[1:]:
			str_list.append(' * ')
			str_list.extend(sym_to_pytorch_string(arg, *sym_args))
		str_list.append(')')
		return str_list
	elif kind == sym.ExpressionKind.Div:
		str_list.append('(')
		str_list.extend(sym_to_pytorch_string(expr_args[0], *sym_args))
		str_list.append(' / ')
		str_list.extend(sym_to_pytorch_string(expr_args[1], *sym_args))
		str_list.append(')')
		return str_list
	elif kind == sym.ExpressionKind.Log:
		return sym_to_pytorch_simple_fun('log', expr_args[0], *sym_args)
	elif kind == sym.ExpressionKind.Abs:
		return sym_to_pytorch_simple_fun('abs', expr_args[0], *sym_args)
	elif kind == sym.ExpressionKind.Exp:
		return sym_to_pytorch_simple_fun('exp', expr_args[0], *sym_args)
	elif kind == sym.ExpressionKind.Pow:
		str_list.append('((')
		str_list.extend(sym_to_pytorch_string(expr_args[0], *sym_args))
		str_list.append(') ** (')
		str_list.extend(sym_to_pytorch_string(expr_args[1], *sym_args))
		str_list.append('))')
		return str_list
	elif kind == sym.ExpressionKind.Sin:
		return sym_to_pytorch_simple_fun('sin', expr_args[0], *sym_args)
	elif kind == sym.ExpressionKind.Cos:
		return sym_to_pytorch_simple_fun('cos', expr_args[0], *sym_args)
	elif kind == sym.ExpressionKind.Tan:
		return sym_to_pytorch_simple_fun('tan', expr_args[0], *sym_args)
	elif kind == sym.ExpressionKind.Asin:
		return sym_to_pytorch_simple_fun('asin', expr_args[0], *sym_args)
	elif kind == sym.ExpressionKind.Acos:
		return sym_to_pytorch_simple_fun('acos', expr_args[0], *sym_args)
	elif kind == sym.ExpressionKind.Atan:
		return sym_to_pytorch_simple_fun('atan', expr_args[0], *sym_args)
	elif kind == sym.ExpressionKind.Atan2:
		str_list.append('torch.atan2(')
		str_list.extend(sym_to_pytorch_string(expr_args[0], *sym_args))
		str_list.append(', ')
		str_list.extend(sym_to_pytorch_string(expr_args[1], *sym_args))
		str_list.append(')')
		return str_list
	elif kind == sym.ExpressionKind.Sinh:
		return sym_to_pytorch_simple_fun('sinh', expr_args[0], *sym_args)
	elif kind == sym.ExpressionKind.Cosh:
		return sym_to_pytorch_simple_fun('cosh', expr_args[0], *sym_args)
	elif kind == sym.ExpressionKind.Tanh:
		return sym_to_pytorch_simple_fun('tanh', expr_args[0], *sym_args)
	elif kind == sym.ExpressionKind.Min:
		str_list.append('torch.min(')
		str_list.extend(sym_to_pytorch_string(expr_args[0], *sym_args))
		str_list.append(', ')
		str_list.extend(sym_to_pytorch_string(expr_args[1], *sym_args))
		str_list.append(')')
		return str_list
	elif kind == sym.ExpressionKind.Max:
		str_list.append('torch.max(')
		str_list.extend(sym_to_pytorch_string(expr_args[0], *sym_args))
		str_list.append(', ')
		str_list.extend(sym_to_pytorch_string(expr_args[1], *sym_args))
		str_list.append(')')
		return str_list
	elif kind == sym.ExpressionKind.Ceil:
		return sym_to_pytorch_simple_fun('ceil', expr_args[0], *sym_args)
	elif kind == sym.ExpressionKind.Floor:
		return sym_to_pytorch_simple_fun('floor', expr_args[0], *sym_args)
	else:
		raise ValueError('Unsupported symbolic: ' + str(kind))


# Helper function to convert a simple function of a single argument
def sym_to_pytorch_simple_fun(name, arg, *sym_args):
	str_list = []
	str_list.append('torch.' + name + '(')
	str_list.extend(sym_to_pytorch_string(arg, *sym_args))
	str_list.append(')')
	return str_list

def substitute_variable_string(var, *sym_args):
	id_vectorized = np.vectorize(sym.Variable.get_id)
	for index, sym_var in enumerate(sym_args):
		var_index = np.where(id_vectorized(sym_var) == var.get_id())

		if var_index[0].size > 0:
			var_index_string = '[..., ' + \
				', '.join([str(i[0]) for i in var_index]) + \
				']'
			return f'{TORCH_ARGS}[{index}]{var_index_string}'
	raise ValueError('Expression contains variable not in sym_args list: ' + str(var))