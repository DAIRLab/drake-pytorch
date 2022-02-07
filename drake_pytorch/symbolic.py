import numpy as np
import types
import torch
import pydrake.symbolic as sym
import pdb
import sympy
from sympy import Symbol, sympify, simplify, trigsimp
from functools import reduce
import operator
from enum import Enum

# string formatting literals
TORCH_ARGS = 'torch_args'
INTERMEDIATE_VARIABLE_PREFIX = 'var'
SYMPY_SYMBOL_PREFIX = 's'

# simplification settings
class Simplifier(Enum):
	ALL = 1
	TRIGONLY = 2
	QUICKTRIG = 4
	NONE = 4

_simplifier_map = {
	Simplifier.ALL: simplify,
	Simplifier.TRIGONLY: trigsimp,
	Simplifier.QUICKTRIG: lambda x: trigsimp(x, quick=True),
	Simplifier.NONE: lambda x: x
}


# Convert an expression to a function in pytorch. The arguments for the
# returned function are be `x[i]`, where index `i` is derived from the
# ordering in `sym_args`. Returns [func, func_string]
# 
# @param expr Either a pydrake.symbolic.Expression, or a list of expressions
# @param *sym_args The arguments to the symbolic expression. Multiple arguments
# 	can be provided. 
# 
# @return func A function object, which can be evaluated on a pytorch Tensor
# 	The function takes a variable number of arguments, matching the list in
#		sym_args
# @return A string expressing the entire python function


def sym_to_pytorch(expr, *sym_args, simplify_computation=Simplifier.ALL):
	simplifier = _simplifier_map[simplify_computation]
	str_list = []
	str_list.append(f'def my_func(*{TORCH_ARGS}):\n')
	# detect batch dimension list from first argument and assume the rest follow
	str_list.append(f'  batch_dims = {TORCH_ARGS}[0].shape[:-{len(sym_args[0].shape)}]\n')
	if isinstance(expr, sym.Expression):
		python_lines = []
		vardict = {}
		sympy_expr_i = sym_to_sympy(expr, vardict)
		if simplify_computation:
			sympy_expr_i = simplifier(sympy_expr_i)
		rev_vardict = {vardict[k]:k for k in vardict}
		expr_name = sympy_to_pytorch_string(sympy_expr_i, {}, python_lines, rev_vardict, sym_args)
		str_list.extend(python_lines)
		str_list.append(f'  return {expr_name}\n')
	elif isinstance(expr, list) or isinstance(expr, np.ndarray):
		if isinstance(expr, np.ndarray):
			shape = expr.shape
			nonbatch_dims = len(shape)
			expr = np.reshape(expr, -1)
		else:
			shape = (len(expr), 1)
		vardict = {}
		memos = {}
		python_lines = []
		expr_names = []
		for i, expr_i in enumerate(expr):
			sympy_expr_i = sym_to_sympy(expr_i, vardict)
			if simplify_computation:
				sympy_expr_i = simplifier(sympy_expr_i)
			rev_vardict = {vardict[k]:k for k in vardict}
			expr_names.append(sympy_to_pytorch_string(sympy_expr_i, memos, python_lines, rev_vardict, sym_args))
		str_list.extend(python_lines)
		expr_indices = ', '.join(expr_names)
		str_list.append(f'  ret = torch.stack(({expr_indices},), dim = -1)\n')
		str_list.append(f'  return torch.reshape(ret, batch_dims + {shape})')
	else:
		raise ValueError('expr must be a drake symbolic Expression or a list')
	func_string = ''.join(str_list)

	code = compile(func_string, 'tmp.py', 'single')
	func = types.FunctionType(code.co_consts[0], globals())
	return func, func_string


# # # # # # # # # # # # # # # # # # # # # # # #
# helper functions / data for drake -> sympy  #
# # # # # # # # # # # # # # # # # # # # # # # #
def _fastpow(*x):
	#print(x[1], type(x[1]), float(x[1]) == 2.0)
	if float(x[1]) == 2.0:
		return x[0] * x[0]
	else:
		return x[0] ** x[1]

_constant_dict = {
	1.0: "1",
	-1.0: "-1"
}

def _sympy_constant_cast(x):
	fx = float(x)
	if fx in _constant_dict:
		return sympify(_constant_dict[fx])
	return sympify(str(x))


sympy_ops = {
	sym.ExpressionKind.Add: sympy.Add,
	sym.ExpressionKind.Mul: sympy.Mul,
	sym.ExpressionKind.Div: lambda x, y: x / y,
	sym.ExpressionKind.Log: sympy.log,
	sym.ExpressionKind.Abs: sympy.Abs,
	sym.ExpressionKind.Exp: sympy.exp,
	sym.ExpressionKind.Pow: _fastpow,
	sym.ExpressionKind.Sin: sympy.sin,
	sym.ExpressionKind.Cos: sympy.cos,
	sym.ExpressionKind.Tan: sympy.tan,
	sym.ExpressionKind.Asin: sympy.asin,
	sym.ExpressionKind.Acos: sympy.acos,
	sym.ExpressionKind.Atan: sympy.atan,
	sym.ExpressionKind.Atan2: sympy.atan2,
	sym.ExpressionKind.Sinh: sympy.sinh,
	sym.ExpressionKind.Cosh: sympy.cosh,
	sym.ExpressionKind.Tanh: sympy.tanh,
	sym.ExpressionKind.Min: sympy.Min,
	sym.ExpressionKind.Max: sympy.Max,
	sym.ExpressionKind.Ceil: sympy.ceiling,
	sym.ExpressionKind.Floor: sympy.floor
}


# # # # # # # # # # # # # # # # # # # # # # # #
# helper functions / data for sympy -> torch  #
# # # # # # # # # # # # # # # # # # # # # # # #
def _reduction(delim):
	return lambda x: f' {delim} '.join(x)

def sympy_to_pytorch_simple_fun(name):
	return lambda x: f'torch.{name}(' + ', '.join(x) + ')'

def _fastpow_string(xpower):
	x = xpower[0]
	power = xpower[1]
	return f'{x} ** {power}'

def _sympy_constant_string(x):
	return f'{float(x)} * torch.ones(batch_dims)'

def _sympy_expression_key(expr):
	expr_top_level = expr.func
	if issubclass(expr_top_level, sympy.Number) or \
		issubclass(expr_top_level, sympy.NumberSymbol):
		return sympy.Float(expr)
	else:
		return expr


_torch_ops = {
	sympy.Add: _reduction('+'),
	sympy.Mul: _reduction('*'),
	sympy.div: sympy_to_pytorch_simple_fun('div'),
	sympy.log: sympy_to_pytorch_simple_fun('log'),
	sympy.Abs: sympy_to_pytorch_simple_fun('abs'),
	sympy.exp: sympy_to_pytorch_simple_fun('exp'),
	sympy.Pow: _fastpow_string,
	sympy.sin: sympy_to_pytorch_simple_fun('sin'),
	sympy.cos: sympy_to_pytorch_simple_fun('cos'),
	sympy.tan: sympy_to_pytorch_simple_fun('tan'),
	sympy.asin: sympy_to_pytorch_simple_fun('asin'),
	sympy.acos: sympy_to_pytorch_simple_fun('acos'),
	sympy.atan: sympy_to_pytorch_simple_fun('atan'),
	sympy.atan2: sympy_to_pytorch_simple_fun('atan2'),
	sympy.sinh: sympy_to_pytorch_simple_fun('sinh'),
	sympy.cosh: sympy_to_pytorch_simple_fun('cosh'),
	sympy.tanh: sympy_to_pytorch_simple_fun('tanh'),
	sympy.Min: sympy_to_pytorch_simple_fun('min'),
	sympy.Max: sympy_to_pytorch_simple_fun('max'),
	sympy.ceiling: sympy_to_pytorch_simple_fun('ceil'),
	sympy.floor: sympy_to_pytorch_simple_fun('floor'),
}

# Convert a drake symbolic expression to sympy
# 
# @param expr The drake expression to convert
# @param vardict Dictionary which corresponds drake variables to sympy Symbols 
# @return The sympy expression
def sym_to_sympy(expr, vardict):
	#pdb.set_trace()
	# If it's a float, just return the expression
	if isinstance(expr, float):
		if expr == 1.0:
			return sympify("1")
		return _sympy_constant_cast(expr)

	str_list = []

	# switch based on the expression kind
	kind = expr.get_kind()
	ctor, expr_args = expr.Unapply()
	if kind == sym.ExpressionKind.Constant:
		if len(expr_args) != 1:
			raise ValueError('Unexpected symbolic Constant of length != 1')
		#pdb.set_trace()
		return _sympy_constant_cast(expr_args[0])
	elif kind == sym.ExpressionKind.Var:
		#pdb.set_trace()
		var_id = expr_args[0].get_id()
		#print(expr_args[0], var_id)
		#pdb.set_trace()
		if var_id in vardict:
			out = vardict[var_id]
		else:
			out = Symbol(f"{SYMPY_SYMBOL_PREFIX}_{len(vardict)}")
			vardict[var_id] = out
		return out
	else:
		# expression combines arguments / is not leaf node
		# first, sympify constituents
		sympy_args = [sym_to_sympy(expr_arg, vardict) for expr_arg in expr_args]
		if any([type(sa) == type((0,)) for sa in sympy_args]):
			pdb.set_trace()
		try:
			return sympy_ops[kind](*sympy_args)
		except KeyError:
			raise ValueError('Unsupported expression type ' + str(kind))




# Convert a sympy expression to a string. The arguments for the returned
# string will be `x[i]`, where index `i` is derived from the ordering
# in `sym_args`
# 
# @param expr The sympy to convert
# @param memos Memoization dictionary of previously-seen expressions
# @param lines python lines which calculated memoized expressions
# @param vardict Dictionary which corresponds sympy Symbols to drake variables
# @param sym_args An ordered list of drake symbolic variables
# @return The variable name the expression is stored in to
def sympy_to_pytorch_string(expr, memos, lines, vardict, sym_args):
	#pdb.set_trace()
	expr_key = _sympy_expression_key(expr)
	if expr_key in memos:
		return memos[expr_key]
	expr_top_level = expr.func
	if issubclass(expr_top_level, sympy.Number) or \
		issubclass(expr_top_level, sympy.NumberSymbol):
		# number, add float
		value = _sympy_constant_string(expr)

	elif issubclass(expr_top_level, sympy.Symbol):
		# variable, index into
		value = substitute_variable_id_string(vardict[expr], sym_args)

	else:
		# get equivalent torch operation
		torch_string_callback = _torch_ops[expr_top_level]

		# squaring and inversion hacks
		expr_args = expr.args
		if torch_string_callback is _fastpow_string:
			#pdb.set_trace()
			if issubclass(expr_args[1].func, sympy.Integer) \
				and int(expr_args[1]) == 2:
				expr_args = [expr_args[0], expr_args[0]]
				torch_string_callback = _reduction('*')
			if issubclass(expr_args[1].func, sympy.Integer) \
				and int(expr_args[1]) == -1:
				expr_args = [_sympy_constant_cast(1.0), expr_args[0]]
				torch_string_callback = _torch_ops[sympy.div]
		args = [sympy_to_pytorch_string(arg, memos, lines, vardict, sym_args) \
			for arg in expr_args]
		value = torch_string_callback(args)
	name = f'{INTERMEDIATE_VARIABLE_PREFIX}_{len(memos)}'
	memos[expr_key] = name
	lines.append(f'  {name} = {value}\n')
	return name

def substitute_variable_id_string(var_id, sym_args):
	id_vectorized = np.vectorize(sym.Variable.get_id)
	for index, sym_var in enumerate(sym_args):
		var_index = np.where(id_vectorized(sym_var) == var_id)

		if var_index[0].size > 0:
			var_index_string = '[:, ' + \
				', '.join([str(i[0]) for i in var_index]) + \
				']'
			return f'{TORCH_ARGS}[{index}]{var_index_string}'
	raise ValueError('Expression contains variable id not in sym_args list: ' + str(var_id))
