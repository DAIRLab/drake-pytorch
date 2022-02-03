import torch
import drake_pytorch
import numpy as np

from pydrake.all import (Parser, AddMultibodyPlantSceneGraph, DiagramBuilder,
                         FindResourceOrThrow, MultibodyForces_, Expression,
                         RotationalInertia_, SpatialInertia_, UnitInertia_,
                         JacobianWrtVariable, MakeVectorVariable, Variable)

#
# Build the symbolic plant and context
#

builder = DiagramBuilder()
plant, _ = AddMultibodyPlantSceneGraph(builder, 0.0)
Parser(plant).AddModelFromFile(
    FindResourceOrThrow("drake/examples/pendulum/Pendulum.urdf"))
plant.Finalize()
sym_plant = plant.ToSymbolic()
context = sym_plant.CreateDefaultContext()

#
# symbolic variables
#

# position of the joint w.r.t. the base
r_joint = MakeVectorVariable(3, 'r_joint')

# position of a point of interest on the arm body
r = MakeVectorVariable(3, 'r')

# mass and inertia of the arm link
m = Variable('m')

# NOTE: I is scaled by the mass, so the actual moment of inertia is m*I
# This is moment of inertia about the link origin (joint), not COM
I = MakeVectorVariable(3, 'I')

# pendulum state
q = MakeVectorVariable(sym_plant.num_positions(), 'q')
v = MakeVectorVariable(sym_plant.num_velocities(), 'v')

#
# Pytorch copies of symbolic variables
#
r_joint_pt = torch.tensor([-1, .5, 2])
r_pt = torch.tensor([3.0, 4.0, 5.0])
m_pt = torch.tensor([1])
I_pt = torch.tensor([.1, .2, .2])

q_pt = torch.tensor([1.0])
v_pt = torch.tensor([2.0])
tau_pt = torch.tensor([-1])

#
# Set mass, inertia, and position variables
#
arm = sym_plant.GetBodyByName("arm")
arm.SetMass(context, m)

inertia = arm.CalcSpatialInertiaInBodyFrame(context)
# I_SScm_E = RotationalInertia_[Expression](I[0], I[1], I[2])
# inertia_sym = SpatialInertia_[Expression].MakeFromCentralInertia(1, inertia.get_com(), I_SScm_E)
# arm.SetSpatialInertiaInBodyFrame(context, inertia_sym)

sym_inertia = SpatialInertia_[Expression](m, inertia.get_com(),
    UnitInertia_[Expression](I[0], I[1], I[2]))
arm.SetSpatialInertiaInBodyFrame(context, sym_inertia)

#
# Get the joint named "theta" and change its position w.r.t. the parent
#
joint=sym_plant.GetJointByName("theta")
joint_frame_on_parent = joint.frame_on_parent()
X_parent_joint = joint_frame_on_parent.GetFixedPoseInBodyFrame()
X_parent_joint.set_translation(r_joint)
joint_frame_on_parent.SetPoseInBodyFrame(context, X_parent_joint)


#
# Set the symbolic state
#
sym_plant.SetPositionsAndVelocities(context, np.array([q,v]))

#
# Calculate the position of point r on the frame
#

world_frame = sym_plant.world_frame()
pos = sym_plant.CalcPointsPositions(
    context=context, frame_B=arm.body_frame(),
    p_BQi=r, frame_A=world_frame)

#
# Calculate the jacobian of point r on the arm
#
wrt = JacobianWrtVariable.kV
J = sym_plant.CalcJacobianTranslationalVelocity(
    context=context, with_respect_to=wrt, frame_B=arm.body_frame(),
    p_BoBi_B=r, frame_A=world_frame, frame_E=world_frame)

#
# Calculate the mass matrix
#
M = sym_plant.CalcMassMatrixViaInverseDynamics(context)

# 
# Example: convert J to a function of (q,v,r,r,r_joint)
#
[func_J, string_J] = drake_pytorch.sym_to_pytorch(J, q, v, r, r_joint)
J_pt = func_J(q_pt, v_pt, r_pt, r_joint_pt)

#
# Example: convert pos to a function with a single argument, stacked
#   [q;r;r_joint]
#
[func_pos, string_pos] = drake_pytorch.sym_to_pytorch(pos, np.hstack((q, r, r_joint)))
func_pos(torch.hstack((q_pt, r_pt, r_joint_pt)))

forces = MultibodyForces_[Expression](sym_plant)
sym_plant.CalcForceElementsContribution(context, forces)


#
# Example: convert M to a function with multiple arguments (q, r_joint, m, I)
#
[func_M, string_M] = drake_pytorch.sym_to_pytorch(M, q, r_joint, m, I)
M_pt = func_M(q_pt, r_joint_pt, m_pt, I_pt)

#
# Example: foward dynamics (the right hand side, where M(q)*vdot = right_hand_side)
#
tau = Variable('tau')
right_hand_side = -sym_plant.CalcBiasTerm(context) + sym_plant.MakeActuationMatrix() * tau + sym_plant.CalcGravityGeneralizedForces(context)
[func_rhs, string_rhs] = drake_pytorch.sym_to_pytorch(right_hand_side, q, v, tau, r_joint, m, I)
rhs_pt = func_rhs(q_pt, v_pt, tau_pt, r_joint_pt, m_pt, I_pt)

import pdb; pdb.set_trace()
