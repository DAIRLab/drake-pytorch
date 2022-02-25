from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Tuple, Callable, Type, Union, Optional

import torch
from torch import Tensor
from torch.nn import Module, ModuleList, Parameter, ParameterList
import drake_pytorch
import numpy as np

from pydrake.all import (Parser, AddMultibodyPlantSceneGraph, DiagramBuilder,
                         FindResourceOrThrow, MultibodyForces_, Expression,
                         RotationalInertia_, SpatialInertia_, UnitInertia_,
                         JacobianWrtVariable, MakeVectorVariable, Variable)

import pdb
import timeit





name = 'chain'
builder = DiagramBuilder()
plant, _ = AddMultibodyPlantSceneGraph(builder, 0.0)
Parser(plant).AddModelFromFile('./three_links.urdf', name)
plant.Finalize()


sym_plant = plant.ToSymbolic()
context = sym_plant.CreateDefaultContext()

model_index = sym_plant.GetModelInstanceByName(name)

body_indices = sym_plant.GetBodyIndices(model_index)
body_frame_ids = [sym_plant.GetBodyFrameIdOrThrow(body_index) for body_index in body_indices]
bodies = [sym_plant.get_body(body_index) for body_index in body_indices]

joint_indices = sym_plant.GetJointIndices(model_index)
joints = [sym_plant.get_joint(joint_index) for joint_index in joint_indices]


world_frame = sym_plant.world_frame()

free_root = sym_plant.GetUniqueFreeBaseBodyOrThrow(model_index)


q = MakeVectorVariable(sym_plant.num_positions(), 'q')
v = MakeVectorVariable(sym_plant.num_velocities(), 'v')

sym_plant.SetPositionsAndVelocities(context, model_index, np.hstack([q,v]))

print('creating inertial parameters')
body_inertia_variables = []
for body in bodies:
    body_name = body.name()

    # construct inertial symbolic parameters
    # mass of body
    m_B = Variable(f'm_{body_name}')

    # origin Bo to c.o.m. Bo in body axes B
    P_BoBcm_B = MakeVectorVariable(3, f'com_{body_name}')

    # unit inertia (Ixx Iyy Izz Ixy Ixz Iyz) about c.o.m. Bcm in body axes
    I_BBcm_B = MakeVectorVariable(6, f'I_{body_name}')


    # set symbolic quantities
    body.SetMass(context, m_B)

    # construct SpatialInertia from sym quantities
    body_spatial_inertia = SpatialInertia_[Expression](m_B,
        P_BoBcm_B, UnitInertia_[Expression](*I_BBcm_B))

    body.SetSpatialInertiaInBodyFrame(context, body_spatial_inertia)
    body_inertia_variables.append(
        np.hstack((m_B, P_BoBcm_B, I_BBcm_B)))


body_inertia_variable_matrix = np.vstack(body_inertia_variables)
print('Calculating mass matrix')
M = sym_plant.CalcMassMatrixViaInverseDynamics(context)

print('Generating pytorch mass matrix function')
[func_M, string_M] = drake_pytorch.sym_to_pytorch(M, q, body_inertia_variable_matrix, simplify_computation = drake_pytorch.Simplifier.QUICKTRIG)

print('Printing generated code\n\n')
print(string_M)
print('\n\n')


bivm = torch.rand((len(bodies), 10))
bivm.requires_grad = True


q = torch.tensor([1.,2,3,4,5,6,7,8,9])
q[0:4] /= q[0:4].norm()
#pdb.set_trace()
print('Estimating computational cost')
N = 2 ** 10
BATCH = 2 ** 6

T_sym = timeit.timeit(lambda: func_M(q, bivm).sum().backward(retain_graph=True), number=N)
T_sym_batch = timeit.timeit(lambda: func_M(q.unsqueeze(0).repeat(BATCH, 1), bivm.unsqueeze(0).repeat(BATCH, 1, 1)).sum().backward(retain_graph=True), number=N // BATCH)


print(f'Serial computation cost per matrix: {T_sym / N} [s]')
print(f'Batched (batch_size={BATCH}) computation cost per matrix: {T_sym_batch / N} [s]')

