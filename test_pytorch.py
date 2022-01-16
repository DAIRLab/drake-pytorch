# import pdb; pdb.set_trace()
import pydrake.symbolic as sym
import torch
import symbolic
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

[func, string] = symbolic.sym_to_pytorch(expr_1, x, y)
print(string)
print(func(tx))

[func, string] = symbolic.sym_to_pytorch(expr_list, x)
print(string)
print(func(tx))


from pydrake.common import FindResourceOrThrow
from pydrake.multibody.parsing import Parser
from pydrake.multibody.plant import AddMultibodyPlantSceneGraph
from pydrake.systems.analysis import Simulator
from pydrake.systems.framework import DiagramBuilder
from pydrake.multibody.tree import JacobianWrtVariable

builder = DiagramBuilder()
plant, _ = AddMultibodyPlantSceneGraph(builder, 0.0)
Parser(plant).AddModelFromFile(
    FindResourceOrThrow("drake/examples/pendulum/Pendulum.urdf"))
plant.Finalize()

plant_sym = plant.ToSymbolic()

context = plant_sym.CreateDefaultContext()
r = sym.MakeVectorVariable(3, 'r')
q = sym.MakeVectorVariable(plant_sym.num_positions(), 'q')
v = sym.MakeVectorVariable(plant_sym.num_velocities(), 'v')
plant_sym.SetPositionsAndVelocities(context, np.array([q,v]))
# r = np.array([1,2,3]).T

print(r)


wrt = JacobianWrtVariable.kV
world_frame = plant_sym.world_frame()
frame = plant_sym.GetFrameByName("arm")
J = plant_sym.CalcJacobianTranslationalVelocity(
    context=context, with_respect_to=wrt, frame_B=frame,
    p_BoBi_B=r, frame_A=world_frame, frame_E=world_frame)

print(J)

print(type(J))

[func, string] = symbolic.sym_to_pytorch(J, q, v, r)

print(string)

print(plant_sym.GetPositionsAndVelocities(context))

qt = torch.tensor([1.0])
vt = torch.tensor([2.0])
rt = torch.tensor([3.0, 4.0, 5.0])

print(func(qt, vt, rt))


# Check accuracy of the expression
[func, string] = symbolic.sym_to_pytorch(expr_all, x, y)
result_torch = func(tx, ty)


sym_stack = np.hstack((x,y))
value_stack = np.hstack((tx.detach().numpy(), ty.detach().numpy()))
value_vectorized = np.vectorize(sym.Expression)
value_stack_expr = value_vectorized(value_stack)

subs_dict = dict(zip(sym_stack, value_stack_expr))
[_, value_1] = expr_1.Substitute(subs_dict).Unapply()
[_, value_2] = expr_2.Substitute(subs_dict).Unapply()
[_, value_3] = expr_3.Substitute(subs_dict).Unapply()

error = np.hstack((value_1, value_2, value_3)) - result_torch.detach().numpy()
print(error)