# libraries
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data
import numpy as np
import pandas as pd
import random
import sklearn
import time
import math
import matplotlib.pyplot as plt
import os
import sys
import argparse

from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn import decomposition
from sklearn import manifold
from tqdm import tqdm
from scipy import stats

# get DH matrix based on robot choice
def get_robot_choice(robot_name):
    if robot_name == "panda":
        robot_choice = "7DoF-7R-Panda"

    return robot_choice



def get_DH(robot_choice, t):
    if robot_choice == "2DoF-2R":
        DH = torch.tensor([[t[0], 0, 1, 0],
                           [t[1], 0, 1, 0]])
    elif robot_choice == "3DoF-3R":
        DH = torch.tensor([[t[0], 0, 1, 0],
                           [t[1], 0, 1, torch.pi/2],
                           [t[2], 0, 0, 0]])
    elif robot_choice == "3DoF-3R-2":
        DH = torch.tensor([[t[0], 380/1000, 0, 0],
                           [t[1],        0, 280/1000, -torch.pi/2],
                           [t[2],        0, 280/1000, 0]])
    elif robot_choice == "4DoF-2RPR":
        DH = torch.tensor([[t[0], 400/1000, 250/1000, 0],
                           [t[1],        0, 150/1000, torch.pi],
                           [   0,     t[2],        0, 0],
                           [t[3], 150/1000,        0, 0]])
    elif robot_choice == "6DoF-6R-Jaco":
        DH = torch.tensor([[                                 -t[0],      0.2755,          0,         torch.pi/2],
                           [                     t[1]-(torch.pi/2),         0.0,       0.41,           torch.pi],
                           [                     t[2]+(torch.pi/2),     -0.0098,          0,         torch.pi/2],
                           [                                  t[3],     -0.2502,          0,         torch.pi/3],
                           [                       t[4]-(torch.pi),    -0.08579,          0,         torch.pi/3],
                           [t[5]+torch.deg2rad(torch.tensor(-100)),     -0.2116,          0,           torch.pi]])
    elif robot_choice == "6DoF-6R-Puma560":
        DH = torch.tensor([[t[0],           0,          0,        -torch.pi/2],
                           [t[1],     0.14909,     0.4318,                  0],
                           [t[2],           0,    -0.0203,         torch.pi/2],
                           [t[3],     0.43307,          0,        -torch.pi/2],
                           [t[4],           0,          0,         torch.pi/2],
                           [t[5],     0.05625,          0,                  0]])
    elif robot_choice == "6DoF-6R-Mico":
        DH = torch.tensor([[                                 -t[0],      0.2755,          0,         torch.pi/2],
                           [                     t[1]-(torch.pi/2),         0.0,       0.29,           torch.pi],
                           [                     t[2]+(torch.pi/2),      -0.007,          0,         torch.pi/2],
                           [                                  t[3],     -0.1661,          0,         torch.pi/3],
                           [                       t[4]-(torch.pi),    -0.08556,          0,         torch.pi/3],
                           [t[5]+torch.deg2rad(torch.tensor(-100)),     -0.2028,          0,           torch.pi]])
    elif robot_choice == "6DoF-6R-IRB140":
        DH = torch.tensor([[t[0],       0.352,       0.07,        -torch.pi/2],
                           [t[1],           0,       0.36,                  0],
                           [t[2],           0,          0,        -torch.pi/2],
                           [t[3],        0.38,          0,         torch.pi/2],
                           [t[4],           0,          0,        -torch.pi/2],
                           [t[5],       0.065,          0,                  0]])
    elif robot_choice == "6DoF-6R-KR5":
        DH = torch.tensor([[t[0],         0.4,       0.18,        -torch.pi/2],
                           [t[1],           0,        0.6,                  0],
                           [t[2],           0,       0.12,         torch.pi/2],
                           [t[3],       -0.62,          0,        -torch.pi/2],
                           [t[4],           0,          0,         torch.pi/2],
                           [t[5],      -0.115,          0,           torch.pi]])
    elif robot_choice == "6DoF-6R-UR10":
        DH = torch.tensor([[t[0],      0.1273,          0,         torch.pi/2],
                           [t[1],           0,     -0.612,                  0],
                           [t[2],           0,    -0.5723,                  0],
                           [t[3],      0.1639,          0,         torch.pi/2],
                           [t[4],      0.1157,          0,        -torch.pi/2],
                           [t[5],      0.0922,          0,                  0]])
    elif robot_choice == "6DoF-6R-UR3":
        DH = torch.tensor([[t[0],      0.1519,          0,         torch.pi/2],
                           [t[1],           0,    -0.2437,                  0],
                           [t[2],           0,    -0.2132,                  0],
                           [t[3],      0.1124,          0,         torch.pi/2],
                           [t[4],     0.08535,          0,        -torch.pi/2],
                           [t[5],      0.0819,          0,                  0]])
    elif robot_choice == "6DoF-6R-UR5":
        DH = torch.tensor([[t[0],     0.08946,          0,         torch.pi/2],
                           [t[1],           0,      0.425,                  0],
                           [t[2],           0,    -0.3922,                  0],
                           [t[3],      0.1091,          0,         torch.pi/2],
                           [t[4],     0.09465,          0,        -torch.pi/2],
                           [t[5],      0.0823,          0,                  0]])
    elif robot_choice == "6DoF-6R-Puma260":
        DH = torch.tensor([[t[0],           0,          0,        -torch.pi/2],
                           [t[1],      0.1254,     0.2032,                  0],
                           [t[2],           0,    -0.0079,         torch.pi/2],
                           [t[3],      0.2032,          0,        -torch.pi/2],
                           [t[4],           0,          0,         torch.pi/2],
                           [t[5],      0.0635,          0,                  0]])
    elif robot_choice == "6DoF-2RP3R-Stanford":
        DH = torch.tensor([[ t[0],  0.412,      0.0, -torch.pi/2],
                           [ t[1],  0.154,      0.0,  torch.pi/2],
                           [-90.0,   t[2],   0.0203,         0.0],
                           [ t[3],    0.0,      0.0, -torch.pi/2],
                           [ t[4],    0.0,      0.0,  torch.pi/2],
                           [ t[5],    0.0,      0.0,         0.0]])
    elif robot_choice == "7DoF-7R-Jaco2":
        DH = torch.tensor([[   t[0]+torch.pi,     -0.2755,          0,         torch.pi/2],
                            [           t[1],         0.0,          0,         torch.pi/2],
                            [           t[2],       -0.41,          0,         torch.pi/2],
                            [           t[3],     -0.0098,          0,         torch.pi/2],
                            [           t[4],     -0.3111,          0,         torch.pi/2],
                            [           t[5],         0.0,          0,         torch.pi/2],
                            [t[6]+torch.pi/2,     -0.2638,          0,           torch.pi]])
    elif robot_choice == "7DoF-7R-Panda":
        DH = torch.tensor([[t[0],    0.333,      0.0,           0],
                           [t[1],      0.0,      0.0, -torch.pi/2],
                           [t[2],    0.316,      0.0,  torch.pi/2],
                           [t[3],      0.0,   0.0825,  torch.pi/2],
                           [t[4],    0.384,  -0.0825, -torch.pi/2],
                           [t[5],      0.0,      0.0,  torch.pi/2],
                           [t[6],    0.107,    0.088,  torch.pi/2]])
    elif robot_choice == "7DoF-7R-WAM":
        DH = torch.tensor([[t[0],      0.0,      0.0, -torch.pi/2],
                           [t[1],      0.0,      0.0,  torch.pi/2],
                           [t[2],    0.550,      0.0, -torch.pi/2],
                           [t[3],      0.0,    0.045,  torch.pi/2],
                           [t[4],    0.300,   -0.045, -torch.pi/2],
                           [t[5],      0.0,      0.0,  torch.pi/2],
                           [t[6],    0.060,      0.0,         0.0]])
    elif robot_choice == "7DoF-7R-Baxter":
        DH = torch.tensor([[           t[0],     0.27,    0.069, -torch.pi/2],
                           [t[1]+torch.pi/2,      0.0,      0.0,  torch.pi/2],
                           [           t[2],    0.364,    0.069, -torch.pi/2],
                           [           t[3],      0.0,      0.0,  torch.pi/2],
                           [           t[4],    0.374,     0.01, -torch.pi/2],
                           [           t[5],      0.0,      0.0,  torch.pi/2],
                           [           t[6],     0.28,      0.0,         0.0]])
    elif robot_choice == "7DoF-7R-Sawyer":
        DH = torch.tensor([[t[0],    0.317,    0.081, -torch.pi/2],
                           [t[1],   0.1925,      0.0, -torch.pi/2],
                           [t[2],      0.4,      0.0, -torch.pi/2],
                           [t[3],   0.1685,      0.0, -torch.pi/2],
                           [t[4],      0.4,      0.0, -torch.pi/2],
                           [t[5],   0.1363,      0.0, -torch.pi/2],
                           [t[6],   0.1338,      0.0,         0.0]])
    elif robot_choice == "7DoF-7R-KukaLWR4+":
        # https://www.researchgate.net/publication/351149270_Exoscarne_Assistive_Strategies_for_an_Industrial_Meat_Cutting_System_Based_on_Physical_Human-Robot_Interaction
        DH = torch.tensor([[t[0],   0.3105,      0.0,  torch.pi/2],
                           [t[1],      0.0,      0.0, -torch.pi/2],
                           [t[2],      0.4,      0.0, -torch.pi/2],
                           [t[3],      0.0,      0.0,  torch.pi/2],
                           [t[4],     0.39,      0.0,  torch.pi/2],
                           [t[5],      0.0,      0.0, -torch.pi/2],
                           [t[6],    0.078,      0.0,         0.0]])
    elif robot_choice == "7DoF-7R-PR2Arm":
        DH = torch.tensor([[t[0],    0.333,      0.0, -torch.pi/2],
                           [t[1],      0.0,    0.350,  torch.pi/2],
                           [t[2],      0.0,    0.400,  torch.pi/2],
                           [t[3],      0.4,      0.0, -torch.pi/2],
                           [t[4],      0.0,      0.0,  torch.pi/2],
                           [t[5],      0.0,      0.0, -torch.pi/2],
                           [t[6],    0.082,      0.0,         0.0]])
    elif robot_choice == "7DoF-7R-PA10":
        DH = torch.tensor([[t[0],    0.317,      0.0, -torch.pi/2],
                           [t[1],      0.0,      0.0,  torch.pi/2],
                           [t[2],    0.450,      0.0, -torch.pi/2],
                           [t[3],      0.0,      0.0,  torch.pi/2],
                           [t[4],    0.480,      0.0, -torch.pi/2],
                           [t[5],      0.0,      0.0,  torch.pi/2],
                           [t[6],    0.070,      0.0,         0.0]])
    elif robot_choice == "7DoF-7R-Gen3":
        DH = torch.tensor([[         t[0],  -0.2848,      0.0,  torch.pi/2],
                           [t[1]+torch.pi,  -0.0118,      0.0,  torch.pi/2],
                           [t[2]+torch.pi,  -0.4208,      0.0,  torch.pi/2],
                           [t[3]+torch.pi,  -0.0128,      0.0,  torch.pi/2],
                           [t[4]+torch.pi,  -0.3143,      0.0,  torch.pi/2],
                           [t[5]+torch.pi,      0.0,      0.0,  torch.pi/2],
                           [t[6]+torch.pi,  -0.1674,      0.0,    torch.pi]])
    elif robot_choice == "7DoF-2RP4R-GP66+1":
        DH = torch.tensor([[t[0],    0.0,      0.0,  torch.pi/2],
                           [t[1],    0.0,     0.25,  torch.pi/2],
                           [ 0.0,   t[2],      0.0,         0.0],
                           [t[3],    0.0,      0.0,  torch.pi/2],
                           [t[4],   0.14,      0.0,  torch.pi/2],
                           [t[5],    0.0,      0.0,  torch.pi/2],
                           [t[6],    0.0,      0.0,  torch.pi/2]])
    elif robot_choice == "8DoF-P8":
        DH = torch.tensor([[        0.0,     t[0],      0.0, -torch.pi/2],
                           [-torch.pi/2,     t[1],      0.0,  torch.pi/2],
                           [       t[2],   0.6718,      0.0,  torch.pi/2],
                           [       t[3],      0.0,   0.4318,         0.0],
                           [       t[4],     0.15,   0.0203, -torch.pi/2],
                           [       t[5],   0.4318,      0.0,  torch.pi/2],
                           [       t[6],      0.0,      0.0, -torch.pi/2],
                           [       t[7],      0.0,      0.0,         0.0]])
    return DH



# Function to check orthogonality of rotation matrix
def is_orthogonal(matrix):
    """Check if a rotation matrix is orthogonal by verifying if R * R^T = I."""
    rotation_part = matrix[:3, :3]  # Extract rotation part
    return np.allclose(np.dot(rotation_part, rotation_part.T), np.eye(3), atol=1e-6)


# A matrix
def A_matrix(t,d,a,al):
    # the inputs of torch.sin and torch.cos are expressed in rad
    A = torch.tensor([[torch.cos(t), -torch.sin(t)*torch.cos(al),  torch.sin(t)*torch.sin(al), a*torch.cos(t)],
                      [torch.sin(t),  torch.cos(t)*torch.cos(al), -torch.cos(t)*torch.sin(al), a*torch.sin(t)],
                      [           0,               torch.sin(al),               torch.cos(al),              d],
                      [           0,                           0,                           0,              1]])
    return A

# Forward Kinematics
def forward_kinematics(DH):

    n_DoF = DH.shape[0]
    T = torch.eye(4,4)
    for i in range(n_DoF):
        A = A_matrix(*DH[i,:])
        T = torch.matmul(T, A)


        if not is_orthogonal(A):
            print(f"Warning: Transformation matrix from joint 0 to joint {i+1} is not orthogonal.")
        #else:
        #    print(f"Valid: Transformation matrix from joint 0 to joint {i+1} is orthogonal.")
    
    return T

# weight initialization
def weights_init_uniform_rule(m):
    classname = m.__class__.__name__
    # for every Linear layer in a model..
    if classname.find('Linear') != -1:
        # get the number of the inputs
        n = m.in_features
        y = 1.0/np.sqrt(n)
        m.weight.data.uniform_(-y, y)
        m.bias.data.fill_(0)


def weights_init_normal_rule(m):
    classname = m.__class__.__name__
    # for every Linear layer in a model..
    if classname.find('Linear') != -1:
        # get the number of the inputs
        n = m.in_features
        y = 1.0/np.sqrt(n)
        m.weight.data.normal_(0.0,y)
        m.bias.data.fill_(0)


def weights_init_xavier_uniform_rule(m):
    classname = m.__class__.__name__
    # for every Linear layer in a model..
    if classname.find('Linear') != -1:
        # get the number of the inputs
        n = m.in_features
        y = 1.0/np.sqrt(n)        
        nn.init.xavier_normal_(m.weight.data, gain=nn.init.calculate_gain('relu'))
        m.bias.data.fill_(0)

def weights_init_xavier_normal_rule(m):
    classname = m.__class__.__name__
    # for every Linear layer in a model..
    if classname.find('Linear') != -1:
        # get the number of the inputs
        n = m.in_features
        y = 1.0/np.sqrt(n)
        nn.init.xavier_normal_(m.weight.data)
        m.bias.data.fill_(0)


def weights_init_kaiming_uniform_rule(m):
    classname = m.__class__.__name__
    # for every Linear layer in a model..
    if classname.find('Linear') != -1:
        # get the number of the inputs
        n = m.in_features
        y = 1.0/np.sqrt(n)
        nn.init.kaiming_uniform_(m.weight.data)
        m.bias.data.fill_(0)

def weights_init_kaiming_normal_rule(m):
    classname = m.__class__.__name__
    # for every Linear layer in a model..
    if classname.find('Linear') != -1:
        # get the number of the inputs
        n = m.in_features
        y = 1.0/np.sqrt(n)
        nn.init.kaiming_normal_(m.weight.data)
        m.bias.data.fill_(0)




# classes and functions
## Conversion from PyTorch 3D GitHub repository
def _angle_from_tan(
    axis: str, other_axis: str, data, horizontal: bool, tait_bryan: bool
) -> torch.Tensor:
    """
    Extract the first or third Euler angle from the two members of
    the matrix which are positive constant times its sine and cosine.

    Args:
        axis: Axis label "X" or "Y or "Z" for the angle we are finding.
        other_axis: Axis label "X" or "Y or "Z" for the middle axis in the
            convention.
        data: Rotation matrices as tensor of shape (..., 3, 3).
        horizontal: Whether we are looking for the angle for the third axis,
            which means the relevant entries are in the same row of the
            rotation matrix. If not, they are in the same column.
        tait_bryan: Whether the first and third axes in the convention differ.

    Returns:
        Euler Angles in radians for each matrix in data as a tensor
        of shape (...).
    """

    i1, i2 = {"X": (2, 1), "Y": (0, 2), "Z": (1, 0)}[axis]
    if horizontal:
        i2, i1 = i1, i2
    even = (axis + other_axis) in ["XY", "YZ", "ZX"]
    if horizontal == even:
        return torch.atan2(data[..., i1], data[..., i2])
    if tait_bryan:
        return torch.atan2(-data[..., i2], data[..., i1])
    return torch.atan2(data[..., i2], -data[..., i1])

def _index_from_letter(letter: str) -> int:
    if letter == "X":
        return 0
    if letter == "Y":
        return 1
    if letter == "Z":
        return 2
    raise ValueError("letter must be either X, Y or Z.")

def matrix_to_euler_angles(matrix: torch.Tensor, convention: str) -> torch.Tensor:
    """
    Convert rotations given as rotation matrices to Euler angles in radians.

    Args:
        matrix: Rotation matrices as tensor of shape (..., 3, 3).
        convention: Convention string of three uppercase letters.

    Returns:
        Euler angles in radians as tensor of shape (..., 3).
    """
    if len(convention) != 3:
        raise ValueError("Convention must have 3 letters.")
    if convention[1] in (convention[0], convention[2]):
        raise ValueError(f"Invalid convention {convention}.")
    for letter in convention:
        if letter not in ("X", "Y", "Z"):
            raise ValueError(f"Invalid letter {letter} in convention string.")
    if matrix.size(-1) != 3 or matrix.size(-2) != 3:
        raise ValueError(f"Invalid rotation matrix shape {matrix.shape}.")
    i0 = _index_from_letter(convention[0])
    i2 = _index_from_letter(convention[2])
    tait_bryan = i0 != i2

    #print(tait_bryan)

    if tait_bryan:
        central_angle = torch.asin(
            matrix[..., i0, i2] * (-1.0 if i0 - i2 in [-1, 2] else 1.0)
        )
    else:
        central_angle = torch.acos(matrix[..., i0, i0])

    #print(central_angle.isnan().nonzero())
    #print(central_angle.shape)
    #sys.exit()

    o = (
        _angle_from_tan(
            convention[0], convention[1], matrix[..., i2], False, tait_bryan
        ),
        central_angle,
        _angle_from_tan(
            convention[2], convention[1], matrix[..., i0, :], True, tait_bryan
        ),
    )

    #print(_angle_from_tan(
    #        convention[0], convention[1], matrix[..., i2], False, tait_bryan
    #    ))
    #print(torch.stack(o, -1))
    #print("UpUp")
    return torch.stack(o, -1)



def euler_angles_to_matrix(euler_angles: torch.Tensor, convention: str) -> torch.Tensor:
    """
    Convert rotations given as Euler angles in radians to rotation matrices.

    Args:
        euler_angles: Euler angles in radians as tensor of shape (..., 3).
        convention: Convention string of three uppercase letters from
            {"X", "Y", and "Z"}.

    Returns:
        Rotation matrices as tensor of shape (..., 3, 3).
    """
    if euler_angles.dim() == 0 or euler_angles.shape[-1] != 3:
        raise ValueError("Invalid input euler angles.")
    if len(convention) != 3:
        raise ValueError("Convention must have 3 letters.")
    if convention[1] in (convention[0], convention[2]):
        raise ValueError(f"Invalid convention {convention}.")
    for letter in convention:
        if letter not in ("X", "Y", "Z"):
            raise ValueError(f"Invalid letter {letter} in convention string.")
    matrices = [
        _axis_angle_rotation(c, e)
        for c, e in zip(convention, torch.unbind(euler_angles, -1))
    ]
    # return functools.reduce(torch.matmul, matrices)
    return torch.matmul(torch.matmul(matrices[0], matrices[1]), matrices[2])



def _axis_angle_rotation(axis: str, angle: torch.Tensor) -> torch.Tensor:
    """
    Return the rotation matrices for one of the rotations about an axis
    of which Euler angles describe, for each value of the angle given.

    Args:
        axis: Axis label "X" or "Y or "Z".
        angle: any shape tensor of Euler angles in radians

    Returns:
        Rotation matrices as tensor of shape (..., 3, 3).
    """

    cos = torch.cos(angle)
    sin = torch.sin(angle)
    one = torch.ones_like(angle)
    zero = torch.zeros_like(angle)

    if axis == "X":
        R_flat = (one, zero, zero, zero, cos, -sin, zero, sin, cos)
    elif axis == "Y":
        R_flat = (cos, zero, sin, zero, one, zero, -sin, zero, cos)
    elif axis == "Z":
        R_flat = (cos, -sin, zero, sin, cos, zero, zero, zero, one)
    else:
        raise ValueError("letter must be either X, Y or Z.")

    return torch.stack(R_flat, -1).reshape(angle.shape + (3, 3))



# vanilla MLP architecture
class MLP_2(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()

        self.name = "MLP[]"
        self.input_fc = nn.Linear(input_dim, 250)
        self.hidden_fc = nn.Linear(250, 100)
        self.output_fc = nn.Linear(100, output_dim)

    def forward(self, x):

        # x = [batch size, height, width]
        batch_size = x.shape[0]
        
        x = x.view(batch_size, -1)
        # x = [batch size, height * width]
        
        h_1 = F.relu(self.input_fc(x))
        # h_1 = [batch size, 250]

        h_2 = F.relu(self.hidden_fc(h_1))
        # h_2 = [batch size, 100]

        y_pred = self.output_fc(h_2)
        # y_pred = [batch size, output dim]

        return y_pred, h_2

class MLP(nn.Module):
    def __init__(self, input_dim, h_sizes, output_dim):
        super().__init__()

        self.name = "MLP [{}, {}, {}]".format(str(input_dim), str(h_sizes).replace("[","").replace("]",""), str(output_dim))
        self.input_dim = input_dim
        self.in_size = input_dim
        self.h_sizes = h_sizes
        self.output_dim = output_dim
        self.out_size = output_dim
        
        self.input_fc = nn.Linear(self.input_dim, self.h_sizes[0])
        
        self.hidden_fc = nn.ModuleList()
        for i in range(len(self.h_sizes)-1):
            self.hidden_fc.append(nn.Linear(self.h_sizes[i], self.h_sizes[i+1]))
        
        self.output_fc = nn.Linear(self.h_sizes[len(self.h_sizes)-1], self.output_dim)

        self.selu_activation = nn.SELU()
        self.relu_activation = nn.ReLU()
        self.prelu_activation = nn.PReLU()
        self.lrelu_activation = nn.LeakyReLU()
        self.sigmoid_activation = nn.Sigmoid()
        self.batch_norm_fc = nn.BatchNorm1d(20000)

    def forward(self, x):

        x = self.input_fc(x)
        #x = self.batch_norm_fc(x)
        x = self.relu_activation(x)  # ReLU(), Sigmoid(), LeakyReLU(negative_slope=0.1)

        for i in range(len(self.h_sizes)-1):
            x = self.hidden_fc[i](x)
            #x = self.batch_norm_fc(x)
            x = self.relu_activation(x)

        x = self.output_fc(x)
        x_temp = x

        return x, x_temp 
        #return x 


class ResMLP_2(nn.Module):
    def __init__(self, input_dim, h_sizes, output_dim):
        super().__init__()

        self.name = "ResMLP [{}, {}, {}]".format(str(input_dim), str(h_sizes).replace("[","").replace("]",""), str(output_dim))
        self.input_dim = input_dim
        self.h_sizes = h_sizes
        self.output_dim = output_dim
        
        self.input_fc = nn.Linear(self.input_dim, self.h_sizes[0])
        self.hidden_fc_1 = nn.Linear(self.h_sizes[0], self.h_sizes[1])
        self.hidden_fc_2 = nn.Linear(self.h_sizes[1], self.h_sizes[2])
        self.hidden_fc_3 = nn.Linear(self.h_sizes[2], self.h_sizes[3])
        self.output_fc = nn.Linear(self.h_sizes[len(self.h_sizes)-1], self.output_dim)       

        self.selu_activation = nn.SELU()
        self.relu_activation = nn.ReLU()
        self.prelu_activation = nn.PReLU()
        self.lrelu_activation = nn.LeakyReLU()
        self.sigmoid_activation = nn.Sigmoid()
        self.batch_norm_fc = nn.BatchNorm1d(20000)

    def forward(self, x):

        x = self.input_fc(x)
        x = self.relu_activation(x)  # ReLU(), Sigmoid(), LeakyReLU(negative_slope=0.1)

        h1 = self.hidden_fc_1(x)
        h1 = self.relu_activation(h1)

        h2 = self.hidden_fc_2(h1)
        h2 = self.relu_activation(h2)

        h3 = self.hidden_fc_3(h2+h1)
        h3 = self.relu_activation(h3)

        o = self.output_fc(h3+h2+h1)
        x_temp = o

        return o, x_temp 


class DenseMLP_old(nn.Module):
    def __init__(self, input_dim, h_sizes, output_dim):
        super().__init__()

        self.name = "DenseMLP [{}, {}, {}]".format(str(input_dim), str(h_sizes).replace("[","").replace("]",""), str(output_dim))
        self.input_dim = input_dim
        self.h_sizes = h_sizes
        self.output_dim = output_dim
        
        self.input_fc = nn.Linear(self.input_dim, self.h_sizes[0])
        self.hidden_fc_1 = nn.Linear(self.h_sizes[0], self.h_sizes[1])
        self.hidden_fc_2 = nn.Linear(self.h_sizes[1], self.h_sizes[2])
        self.hidden_fc_3 = nn.Linear(self.h_sizes[2]*2, self.h_sizes[3])
        self.output_fc = nn.Linear(self.h_sizes[len(self.h_sizes)-1]*3, self.output_dim)       

        self.selu_activation = nn.SELU()
        self.relu_activation = nn.ReLU()
        self.prelu_activation = nn.PReLU()
        self.lrelu_activation = nn.LeakyReLU()
        self.sigmoid_activation = nn.Sigmoid()
        self.batch_norm_fc = nn.BatchNorm1d(20000)

    def forward(self, x):

        x = self.input_fc(x)
        x = self.relu_activation(x)  # ReLU(), Sigmoid(), LeakyReLU(negative_slope=0.1)

        h1 = self.hidden_fc_1(x)
        h1 = self.relu_activation(h1)

        h2 = self.hidden_fc_2(h1)
        h2 = self.relu_activation(h2)

        #print(h2.shape)
        #sys.exit(0)

        h3 = self.hidden_fc_3(torch.cat((h2,h1),1))
        h3 = self.relu_activation(h3)

        o = self.output_fc(torch.cat((h3,h2,h1),1))
        x_temp = o

        return o, x_temp 


# count network parameters
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# data loader
class LoadIKDataset(Dataset):
    def __init__(self, inputs_array, outputs_array, device):
        x_temp = inputs_array
        y_temp = outputs_array

        self.x_data = torch.tensor(x_temp, dtype=torch.float32) #.to(device) 
        self.y_data = torch.tensor(y_temp, dtype=torch.float32) #.to(device)  

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        current_input = self.x_data[idx, :]
        current_output = self.y_data[idx, :]
        
        sample = {'input': current_input,
                  'output': current_output}
        return sample
    

    def __len__(self):
        return len(self.x_data)


# function to load the dataset
def load_dataset(data, n_DoF, batch_size, robot_choice, dataset_type, device, input_dim):

    # file data_4DoF
    #X = data[:,:3]
    #y = data[:,6:]

    # file data_4DOF_2
    if robot_choice == "6DoF-6R-Puma260":
        X = data[:,:6]
        y = data[:,6:]
    if robot_choice == "7DoF-7R-Panda" or robot_choice == "7DoF-GP66" or robot_choice == "8DoF-P8":
        if dataset_type == "seq":
            print("==> Sequence dataset ...")
            X = data[:,:input_dim]
            y = data[:,input_dim:]
        elif dataset_type == "1_to_1": 
            print("==> 1 to 1 dataset ...")
            #X = data[:,:6]
            #X = data[:,:input_dim]
            #y = data[:,input_dim:] #13]

            
            input_dim = input_dim*2+n_DoF
            X = data[:,(input_dim-6):input_dim]
            y = data[:,input_dim:] #13]
            
    if robot_choice == "3DoF-3R":
        if dataset_type == "seq":
            print("==> Sequence dataset ...")
            X = data[:,:7]
            y = data[:,7:]
        elif dataset_type == "1_to_1": 
            print("==> 1 to 1 dataset ...")
            X = data[:,:2]
            y = data[:,2:5] #13]

        
    #y = data[:,:2]
    #X = data[:,2:]
        
    # split in train and test sets
    
    X_train, X_validate, y_train, y_validate = train_test_split(X, 
                                                                y, 
                                                                test_size = 0.1,
                                                                random_state = 1)

    X_train, X_test, y_train, y_test = train_test_split(X_train, 
                                                        y_train, 
                                                        test_size = 0.1,
                                                        random_state = 1)
    
    
    """
    n_samples = len(data[:,0])
    X_train = X[:int(0.8*n_samples),:]
    X_validate = X[int(0.8*n_samples):int(0.9*n_samples),:]
    X_test = X[int(0.9*n_samples):,:]

    y_train = y[:int(0.8*n_samples),:]
    y_validate = y[int(0.8*n_samples):int(0.9*n_samples),:]
    y_test = y[int(0.9*n_samples):,:]
    """
    
    
    sc_in = MinMaxScaler(copy=True, feature_range=(0, 1))
    sc_out = MinMaxScaler(copy=True, feature_range=(0, 1))
    
    X_train = sc_in.fit_transform(X_train)
    X_validate = sc_in.transform(X_validate) 
    X_test = sc_in.transform(X_test) 

    #print("Here")
   

    """
    min_value = np.min(X_train)
    max_value = np.max.max(X_train)
    range = max_value - min_value
    X_train = (X_train - min_value) / range
    X_validate = (X_validate - min_value) / range
    X_test = (X_test - min_value) / range

    print(X_train.min(), X_train.max())
    print(X_validate.min(), X_validate.max())
    print(X_test.min(), X_test.max())
    """

    
    
    #xx = torch.from_numpy(X_train)
    #xx = xx
    #print(xx)
    #print(B.to(torch.float64))
    #X_train = input_mapping(torch.from_numpy(X_train),B.to(torch.float64))
    #X_test = input_mapping(torch.from_numpy(X_test),B.to(torch.float64))
    #X_train = X_train.numpy()
    #X_test = X_test.numpy()
    
    #X_train = X_train_i
    #X_test = X_test_i

    #y_train = sc_out.fit_transform(y_train)
    #y_test = sc_out.transform(y_test) 

    print("==> Shape X_train: ", X_train.shape)
    print("==> Shape y_train: ", y_train.shape)

    train_data = LoadIKDataset(X_train, y_train, device)
    test_data = LoadIKDataset(X_validate, y_validate, device)

    train_data_loader = DataLoader(dataset=train_data,
                                   batch_size=batch_size,
                                   shuffle=True,
                                   drop_last=True,
                                   pin_memory=False,
                                   #num_workers=0,
                                   #persistent_workers=True
                                   )

    test_data_loader = DataLoader(dataset=test_data,
                                   batch_size=batch_size,
                                   drop_last=False,
                                   shuffle=False,
                                   pin_memory=False,
                                   #num_workers=8,
                                   #persistent_workers=True
                                   )

    return train_data_loader, test_data_loader, X_validate, y_validate, X_train, y_train, X_test, y_test




# function to load the dataset
def load_all_dataset(data, n_DoF, batch_size, robot_choice, dataset_type, device, input_dim, robot_list):


    X_train, y_train = [], []
    X_test, y_test = [], []
    X_validate, y_validate = [], []

    train_test_val_all = {}

    for i in range(len(robot_list)):
        if dataset_type == "combine-6DoF" or dataset_type == "combine-7DoF"  or dataset_type == "combine-up-to-7DoF":
            print("\n==> Sequence dataset for {}...".format(robot_list[i]))
            # get the input
            X = data[:,:input_dim,i]

            # get the output
            y = data[:,input_dim:,i]

            # get the train and validate sets
            X_train_each, X_validate_each, y_train_each, y_validate_each = train_test_split(X, 
                                                                                            y, 
                                                                                            test_size = 0.1,
                                                                                            random_state = 1)

            # get the train and test sets
            X_train_each, X_test_each, y_train_each, y_test_each = train_test_split(X_train_each, 
                                                                                    y_train_each, 
                                                                                    test_size = 0.1,
                                                                                    random_state = 1)


            X_train.append(X_train_each)
            y_train.append(y_train_each)

            X_validate.append(X_validate_each)
            y_validate.append(y_validate_each)
            
            X_test.append(X_test_each)
            y_test.append(y_test_each)

            train_test_val_all[robot_list[i]] = {"X_test": np.array(X_test_each),
                                                 "y_test": np.array(y_test_each)}


    

    # convert lists to arrays
    X_train, y_train = np.array(X_train), np.array(y_train)
    X_test, y_test = np.array(X_test), np.array(y_test)
    X_validate, y_validate = np.array(X_validate), np.array(y_validate)


    X_train = np.reshape(X_train, newshape=(X_train.shape[0]*X_train.shape[1], X_train.shape[2]) , order="F")
    X_validate = np.reshape(X_validate, newshape=(X_validate.shape[0]*X_validate.shape[1], X_validate.shape[2]) , order="F")
    X_test = np.reshape(X_test, newshape=(X_test.shape[0]*X_test.shape[1], X_test.shape[2]) , order="F")
   
    y_train = np.reshape(y_train, newshape=(y_train.shape[0]*y_train.shape[1], y_train.shape[2]) , order="F")
    y_validate = np.reshape(y_validate, newshape=(y_validate.shape[0]*y_validate.shape[1], y_validate.shape[2]) , order="F")
    y_test = np.reshape(y_test, newshape=(y_test.shape[0]*y_test.shape[1], y_test.shape[2]) , order="F")

   


    """
    print(X_train.shape)

    print(np.reshape(X_train, newshape=(X_train.shape[0]*X_train.shape[1], X_train.shape[2]) , order="F"))
    
    
    print(X_train[0,0,:])
    print(X_train[0,1,:])
    print(np.reshape(X_train[0,:2,:], newshape=(2, X_train.shape[2]) , order="F"))
    

    sys.exit()
    """



    sc_in = MinMaxScaler(copy=True, feature_range=(0, 1))
    sc_out = MinMaxScaler(copy=True, feature_range=(0, 1))
    
    X_train = sc_in.fit_transform(X_train)
    X_validate = sc_in.transform(X_validate) 
    X_test = sc_in.transform(X_test) 

    #print("Here")
   

    print("==> Shape X_train: ", X_train.shape)
    print("==> Shape y_train: ", y_train.shape)

    print("==> Shape X_validate: ", X_validate.shape)
    print("==> Shape y_validate: ", y_validate.shape)

    print("==> Shape X_test: ", X_test.shape)
    print("==> Shape y_test: ", y_test.shape)

    train_data = LoadIKDataset(X_train, y_train, device)
    test_data = LoadIKDataset(X_validate, y_validate, device)

    train_data_loader = DataLoader(dataset=train_data,
                                   batch_size=batch_size,
                                   shuffle=True,
                                   drop_last=True,
                                   pin_memory=True,
                                   num_workers=2,
                                   persistent_workers=True)

    test_data_loader = DataLoader(dataset=test_data,
                                   batch_size=batch_size,
                                   drop_last=False,
                                   shuffle=False,
                                   pin_memory=True,
                                   num_workers=2,
                                   persistent_workers=True)

    return train_data_loader, test_data_loader, train_test_val_all, sc_in




# function to load the dataset
def load_all_dataset_2(data, n_DoF, batch_size, robot_choice, dataset_type, device, input_dim, robot_list, robot_list_test):


    X_train, y_train = [], []
    X_test, y_test = [], []
    X_validate, y_validate = [], []

    train_test_val_all = {}

    for i in range(len(robot_list)):
        if dataset_type == "combine-6DoF" or dataset_type == "combine-7DoF"  or dataset_type == "combine-up-to-7DoF":

            if robot_list[i] in robot_list_test:
                
                print("\n==> Sequence dataset for {}... (test set)".format(robot_list[i]))

                # get the input
                X = data[:,:input_dim,i]

                # get the output
                y = data[:,input_dim:,i]

                X_test.append(X)
                y_test.append(y)

                train_test_val_all[robot_list[i]] = {"X_test": np.array(X),
                                                    "y_test": np.array(y)}
            else: 
                print("\n==> Sequence dataset for {}... (train set)".format(robot_list[i]))
                # get the input
                X = data[:,:input_dim,i]

                # get the output
                y = data[:,input_dim:,i]

                # get the train and validate sets
                X_train_each, X_validate_each, y_train_each, y_validate_each = train_test_split(X, 
                                                                                                y, 
                                                                                                test_size = 0.1,
                                                                                                random_state = 1)
                
                X_train.append(X_train_each)
                y_train.append(y_train_each)

                X_validate.append(X_validate_each)
                y_validate.append(y_validate_each)
            



    # convert lists to arrays
    X_train, y_train = np.array(X_train), np.array(y_train)
    X_test, y_test = np.array(X_test), np.array(y_test)
    X_validate, y_validate = np.array(X_validate), np.array(y_validate)


    X_train = np.reshape(X_train, newshape=(X_train.shape[0]*X_train.shape[1], X_train.shape[2]) , order="F")
    X_validate = np.reshape(X_validate, newshape=(X_validate.shape[0]*X_validate.shape[1], X_validate.shape[2]) , order="F")
    X_test = np.reshape(X_test, newshape=(X_test.shape[0]*X_test.shape[1], X_test.shape[2]) , order="F")
   
    y_train = np.reshape(y_train, newshape=(y_train.shape[0]*y_train.shape[1], y_train.shape[2]) , order="F")
    y_validate = np.reshape(y_validate, newshape=(y_validate.shape[0]*y_validate.shape[1], y_validate.shape[2]) , order="F")
    y_test = np.reshape(y_test, newshape=(y_test.shape[0]*y_test.shape[1], y_test.shape[2]) , order="F")



    sc_in = MinMaxScaler(copy=True, feature_range=(0, 1))
    sc_out = MinMaxScaler(copy=True, feature_range=(0, 1))
    
    X_train = sc_in.fit_transform(X_train)
    X_validate = sc_in.transform(X_validate) 
    X_test = sc_in.transform(X_test) 


    print("==> Shape X_train: ", X_train.shape)
    print("==> Shape y_train: ", y_train.shape)

    print("==> Shape X_validate: ", X_validate.shape)
    print("==> Shape y_validate: ", y_validate.shape)

    print("==> Shape X_test: ", X_test.shape)
    print("==> Shape y_test: ", y_test.shape)

    train_data = LoadIKDataset(X_train, y_train, device)
    test_data = LoadIKDataset(X_validate, y_validate, device)

    train_data_loader = DataLoader(dataset=train_data,
                                   batch_size=batch_size,
                                   shuffle=True,
                                   drop_last=True,
                                   pin_memory=True,
                                   num_workers=3,
                                   persistent_workers=True)

    test_data_loader = DataLoader(dataset=test_data,
                                   batch_size=batch_size,
                                   drop_last=False,
                                   shuffle=False,
                                   pin_memory=True,
                                   num_workers=3,
                                   persistent_workers=True)

    return train_data_loader, test_data_loader, train_test_val_all, sc_in




# function to load the dataset
def load_all_dataset_2_newloss(data, n_DoF, batch_size, robot_choice, dataset_type, device, input_dim, robot_list, robot_list_test):


    X_train, y_train = [], []
    X_test, y_test = [], []
    X_validate, y_validate = [], []

    train_test_val_all = {}

    for i in range(len(robot_list)):
        if dataset_type == "combine-6DoF" or dataset_type == "combine-7DoF"  or dataset_type == "combine-up-to-7DoF":

            if robot_list[i] in robot_list_test:
                
                print("\n==> Sequence dataset for {}... (test set)".format(robot_list[i]))

                # get the input
                X = data[:,:input_dim,i]

                # get the output
                y = data[:,input_dim:,i]

                X_test.append(X)
                y_test.append(y)

                train_test_val_all[robot_list[i]] = {"X_test": np.array(X),
                                                    "y_test": np.array(y)}
            else: 
                print("\n==> Sequence dataset for {}... (train set)".format(robot_list[i]))
                # get the input
                X = data[:,:input_dim,i]

                # get the output
                y = data[:,input_dim:,i]

                # get the train and validate sets
                X_train_each, X_validate_each, y_train_each, y_validate_each = train_test_split(X, 
                                                                                                y, 
                                                                                                test_size = 0.1,
                                                                                                random_state = 1)
                
                X_train.append(X_train_each)
                y_train.append(y_train_each)

                X_validate.append(X_validate_each)
                y_validate.append(y_validate_each)
            



    # convert lists to arrays
    X_train, y_train = np.array(X_train), np.array(y_train)
    X_test, y_test = np.array(X_test), np.array(y_test)
    X_validate, y_validate = np.array(X_validate), np.array(y_validate)


    X_train = np.reshape(X_train, newshape=(X_train.shape[0]*X_train.shape[1], X_train.shape[2]) , order="F")
    X_validate = np.reshape(X_validate, newshape=(X_validate.shape[0]*X_validate.shape[1], X_validate.shape[2]) , order="F")
    X_test = np.reshape(X_test, newshape=(X_test.shape[0]*X_test.shape[1], X_test.shape[2]) , order="F")
   
    y_train = np.reshape(y_train, newshape=(y_train.shape[0]*y_train.shape[1], y_train.shape[2]) , order="F")
    y_validate = np.reshape(y_validate, newshape=(y_validate.shape[0]*y_validate.shape[1], y_validate.shape[2]) , order="F")
    y_test = np.reshape(y_test, newshape=(y_test.shape[0]*y_test.shape[1], y_test.shape[2]) , order="F")



    sc_in = MinMaxScaler(copy=True, feature_range=(0, 1))
    sc_out = MinMaxScaler(copy=True, feature_range=(0, 1))
    
    X_train = sc_in.fit_transform(X_train)
    X_validate = sc_in.transform(X_validate) 
    X_test = sc_in.transform(X_test) 


    print("==> Shape X_train: ", X_train.shape)
    print("==> Shape y_train: ", y_train.shape)

    print("==> Shape X_validate: ", X_validate.shape)
    print("==> Shape y_validate: ", y_validate.shape)

    print("==> Shape X_test: ", X_test.shape)
    print("==> Shape y_test: ", y_test.shape)

    train_data = LoadIKDataset(X_train, y_train, device)
    test_data = LoadIKDataset(X_validate, y_validate, device)

    train_data_loader = DataLoader(dataset=train_data,
                                   batch_size=batch_size,
                                   shuffle=True,
                                   drop_last=True,
                                   pin_memory=False,
                                   num_workers=2,
                                   persistent_workers=True)

    test_data_loader = DataLoader(dataset=test_data,
                                   batch_size=batch_size,
                                   drop_last=False,
                                   shuffle=False,
                                   pin_memory=False,
                                   num_workers=2,
                                   persistent_workers=True)

    return train_data_loader, test_data_loader, train_test_val_all, sc_in




# function to load the dataset
def load_all_dataset_3(data_train, data_test, n_DoF, batch_size, robot_choice, dataset_type, device, input_dim, robot_list, robot_list_test):


    X_train, y_train = [], []
    X_test, y_test = [], []
    X_validate, y_validate = [], []

    train_test_val_all = {}

    # Load test dataset 
    for i in range(data_test.shape[2]):

        if robot_list[i] in robot_list_test:
            
            print("==> Sequence dataset for {}... (test set)".format(robot_list[i]))

            # get the input
            X = data_test[:,:input_dim,i]

            # get the output
            y = data_test[:,input_dim:,i]

            X_test.append(X)
            y_test.append(y)

            train_test_val_all[robot_list[i]] = {"X_test": np.array(X),
                                                 "y_test": np.array(y)}

    # Load train dataset
    for i in range(data_train.shape[2]):
        
        print("==> Sequence dataset for random robot {}... (train set)".format(i))
        # get the input
        X = data_train[:,:input_dim,i]

        # get the output
        y = data_train[:,input_dim:,i]

        # get the train and validate sets
        X_train_each, X_validate_each, y_train_each, y_validate_each = train_test_split(X, 
                                                                                        y, 
                                                                                        test_size = 0.1,
                                                                                        random_state = 1)
        
        X_train.append(X_train_each)
        y_train.append(y_train_each)

        X_validate.append(X_validate_each)
        y_validate.append(y_validate_each)
        



    # convert lists to arrays
    X_train, y_train = np.array(X_train), np.array(y_train)
    X_test, y_test = np.array(X_test), np.array(y_test)
    X_validate, y_validate = np.array(X_validate), np.array(y_validate)


    X_train = np.reshape(X_train, newshape=(X_train.shape[0]*X_train.shape[1], X_train.shape[2]) , order="F")
    X_validate = np.reshape(X_validate, newshape=(X_validate.shape[0]*X_validate.shape[1], X_validate.shape[2]) , order="F")
    X_test = np.reshape(X_test, newshape=(X_test.shape[0]*X_test.shape[1], X_test.shape[2]) , order="F")
   
    y_train = np.reshape(y_train, newshape=(y_train.shape[0]*y_train.shape[1], y_train.shape[2]) , order="F")
    y_validate = np.reshape(y_validate, newshape=(y_validate.shape[0]*y_validate.shape[1], y_validate.shape[2]) , order="F")
    y_test = np.reshape(y_test, newshape=(y_test.shape[0]*y_test.shape[1], y_test.shape[2]) , order="F")



    sc_in = MinMaxScaler(copy=True, feature_range=(0, 1))
    sc_out = MinMaxScaler(copy=True, feature_range=(0, 1))
    
    X_train = sc_in.fit_transform(X_train)
    X_validate = sc_in.transform(X_validate) 
    X_test = sc_in.transform(X_test) 


    print("==> Shape X_train: ", X_train.shape)
    print("==> Shape y_train: ", y_train.shape)

    print("==> Shape X_validate: ", X_validate.shape)
    print("==> Shape y_validate: ", y_validate.shape)

    print("==> Shape X_test: ", X_test.shape)
    print("==> Shape y_test: ", y_test.shape)

    train_data = LoadIKDataset(X_train, y_train, device)
    test_data = LoadIKDataset(X_validate, y_validate, device)

    train_data_loader = DataLoader(dataset=train_data,
                                   batch_size=batch_size,
                                   shuffle=True,
                                   drop_last=True,
                                   pin_memory=False,
                                   num_workers=2,
                                   persistent_workers=True)

    test_data_loader = DataLoader(dataset=test_data,
                                   batch_size=batch_size,
                                   drop_last=False,
                                   shuffle=False,
                                   pin_memory=False,
                                   num_workers=2,
                                   persistent_workers=True)

    return train_data_loader, test_data_loader, train_test_val_all, sc_in







def load_dataset_2(data, n_DoF, batch_size, robot_choice, dataset_type, device, input_dim):

    # file data_4DoF
    #X = data[:,:3]
    #y = data[:,6:]

    # file data_4DOF_2
    if robot_choice == "6DoF-6R-Puma260":
        X = data[:,:6]
        y = data[:,6:]
    if robot_choice == "7DoF-7R-Panda" or robot_choice == "7DoF-GP66" or robot_choice == "8DoF-P8":
        if dataset_type == "seq":
            print("\n==> Sequence dataset ...")
            X = data[:,:input_dim]
            y = data[:,input_dim:]
        elif dataset_type == "1_to_1": 
            print("\n==> 1 to 1 dataset ...")
            #X = data[:,:6]
            #X = data[:,:input_dim]
            #y = data[:,input_dim:] #13]

            
            input_dim = input_dim*2+n_DoF
            X = data[:,(input_dim-6):input_dim]
            y = data[:,input_dim:] #13]
            
    if robot_choice == "3DoF-3R":
        if dataset_type == "seq":
            print("==> Sequence dataset ...")
            X = data[:,:7]
            y = data[:,7:]
        elif dataset_type == "1_to_1": 
            print("==> 1 to 1 dataset ...")
            X = data[:,:2]
            y = data[:,2:5] #13]

        
    #y = data[:,:2]
    #X = data[:,2:]
        
    # split in train and test sets
    
    X_train, X_validate, y_train, y_validate = train_test_split(X, 
                                                                y, 
                                                                test_size = 0.1,
                                                                random_state = 1)

    X_train, X_test, y_train, y_test = train_test_split(X_train, 
                                                        y_train, 
                                                        test_size = 0.1,
                                                        random_state = 1)
    
    
    """
    n_samples = len(data[:,0])
    X_train = X[:int(0.8*n_samples),:]
    X_validate = X[int(0.8*n_samples):int(0.9*n_samples),:]
    X_test = X[int(0.9*n_samples):,:]

    y_train = y[:int(0.8*n_samples),:]
    y_validate = y[int(0.8*n_samples):int(0.9*n_samples),:]
    y_test = y[int(0.9*n_samples):,:]
    """
    
    
    sc_in = MinMaxScaler(copy=True, feature_range=(0, 1))
    sc_out = MinMaxScaler(copy=True, feature_range=(0, 1))
    
    X_train = sc_in.fit_transform(X_train)
    X_validate = sc_in.transform(X_validate) 
    X_test = sc_in.transform(X_test) 

    #print("Here")
   

    """
    min_value = np.min(X_train)
    max_value = np.max.max(X_train)
    range = max_value - min_value
    X_train = (X_train - min_value) / range
    X_validate = (X_validate - min_value) / range
    X_test = (X_test - min_value) / range

    print(X_train.min(), X_train.max())
    print(X_validate.min(), X_validate.max())
    print(X_test.min(), X_test.max())
    """

    
    
    #xx = torch.from_numpy(X_train)
    #xx = xx
    #print(xx)
    #print(B.to(torch.float64))
    #X_train = input_mapping(torch.from_numpy(X_train),B.to(torch.float64))
    #X_test = input_mapping(torch.from_numpy(X_test),B.to(torch.float64))
    #X_train = X_train.numpy()
    #X_test = X_test.numpy()
    
    #X_train = X_train_i
    #X_test = X_test_i

    #y_train = sc_out.fit_transform(y_train)
    #y_test = sc_out.transform(y_test) 

    print("==> Shape X_train: ", X_train.shape)
    print("==> Shape y_train: ", y_train.shape)

    train_data = LoadIKDataset(X_train, y_train, device)
    test_data = LoadIKDataset(X_validate, y_validate, device)

    train_data_loader = DataLoader(dataset=train_data,
                                   batch_size=batch_size,
                                   shuffle=True,
                                   drop_last=True,
                                   pin_memory=False,
                                   num_workers=8,
                                   persistent_workers=True)

    test_data_loader = DataLoader(dataset=test_data,
                                   batch_size=batch_size,
                                   drop_last=False,
                                   shuffle=False,
                                   pin_memory=False,
                                   num_workers=8,
                                   persistent_workers=True)

    return train_data_loader, test_data_loader, X_validate, y_validate, X_train, y_train, X_test, y_test, sc_in



# function to load the dataset
def load_dataset_forward(data, n_DoF, batch_size, robot_choice, dataset_type, device, output_dim):

    # file data_4DoF
    #X = data[:,:3]
    #y = data[:,6:]

    # file data_4DOF_2
    if robot_choice == "6DoF-6R-Puma260":
        X = data[:,:6]
        y = data[:,6:]
    if robot_choice == "7DoF-7R-Panda" or robot_choice == "7DoF-GP66" or robot_choice == "8DoF-P8":
        if dataset_type == "seq":
            print("\n==> Sequence dataset ...")
            X = data[:,:input_dim]
            y = data[:,input_dim:]
        elif dataset_type == "1_to_1": 
            print("\n==> 1 to 1 dataset ...")
            #X = data[:,:6]
            #X = data[:,:input_dim]
            #y = data[:,input_dim:] #13]

            
            input_dim = output_dim*2+n_DoF
            y = data[:,(input_dim-6):input_dim]
            X = data[:,input_dim:] #13]
            
    if robot_choice == "3DoF-3R":
        if dataset_type == "seq":
            print("==> Sequence dataset ...")
            X = data[:,:7]
            y = data[:,7:]
        elif dataset_type == "1_to_1": 
            print("==> 1 to 1 dataset ...")
            X = data[:,:2]
            y = data[:,2:5] #13]

 
        
    # split in train and test sets    
    X_train, X_validate, y_train, y_validate = train_test_split(X, 
                                                                y, 
                                                                test_size = 0.1,
                                                                random_state = 1)

    X_train, X_test, y_train, y_test = train_test_split(X_train, 
                                                        y_train, 
                                                        test_size = 0.1,
                                                        random_state = 1)
    

    
    """
    sc_in = MinMaxScaler(copy=True, feature_range=(0, 1))
    sc_out = MinMaxScaler(copy=True, feature_range=(0, 1))
    
    X_train = sc_in.fit_transform(X_train)
    X_validate = sc_in.transform(X_validate) 
    X_test = sc_in.transform(X_test) 
    """

    print("==> Shape X_train: ", X_train.shape)
    print("==> Shape y_train: ", y_train.shape)

    train_data = LoadIKDataset(X_train, y_train, device)
    test_data = LoadIKDataset(X_validate, y_validate, device)

    train_data_loader = DataLoader(dataset=train_data,
                                   batch_size=batch_size,
                                   shuffle=True,
                                   drop_last=True,
                                   pin_memory=False,
                                   num_workers=8,
                                   persistent_workers=True)

    test_data_loader = DataLoader(dataset=test_data,
                                   batch_size=batch_size,
                                   drop_last=False,
                                   shuffle=False,
                                   pin_memory=False,
                                   num_workers=8,
                                   persistent_workers=True)

    return train_data_loader, test_data_loader, X_validate, y_validate, X_train, y_train, X_test, y_test





# function to load the dataset
def load_dataset_sobolev(data, n_DoF, batch_size, robot_choice, dataset_type, device):

    if robot_choice == "7DoF-7R-Panda":
        #X = data[:,:6]
        #y = data[:,6:]
        
        X = data[:,6:13]
        y = np.concatenate((data[:,:6], data[:,13:]), axis=1)

    #print(X.shape)
    #print(y.shape)

          
    X_train, X_validate, y_train, y_validate = train_test_split(X, 
                                                                y, 
                                                                test_size = 0.1,
                                                                random_state = 1)

    X_train, X_test, y_train, y_test = train_test_split(X_train, 
                                                        y_train, 
                                                        test_size = 0.1,
                                                        random_state = 1)    
    
    
    sc_in = MinMaxScaler(copy=True, feature_range=(0, 1))
    sc_out = MinMaxScaler(copy=True, feature_range=(0, 1))
    
    X_train = sc_in.fit_transform(X_train)
    X_validate = sc_in.transform(X_validate) 
    X_test = sc_in.transform(X_test) 
       
    print("==> Shape X_train: ", X_train.shape)
    print("==> Shape y_train: ", y_train.shape)
    #print(batch_size)
    #batch_size = 10

    train_data = LoadIKDataset(X_train, y_train, device)
    test_data = LoadIKDataset(X_validate, y_validate, device)

    train_data_loader = DataLoader(dataset=train_data,
                                   batch_size=batch_size,
                                   shuffle=True,
                                   drop_last=True,
                                   pin_memory=False,
                                   num_workers=8,
                                   persistent_workers=True)

    test_data_loader = DataLoader(dataset=test_data,
                                   batch_size=batch_size,
                                   drop_last=False,
                                   shuffle=False,
                                   pin_memory=False,
                                   num_workers=8,
                                   persistent_workers=True)
    

    return train_data_loader, test_data_loader, X_validate, y_validate, X_train, y_train, X_test, y_test


# function to load the dataset
def load_dataset_dist(data, n_DoF, batch_size, robot_choice, dataset_type, device):

    # file data_4DoF
    #X = data[:,:3]
    #y = data[:,6:]

    # file data_4DOF_2
    if robot_choice == "6DoF-6R-Puma260":
        X = data[:,:6]
        y = data[:,6:]
    if robot_choice == "7DoF-7R-Panda":
        if dataset_type == "seq":
            print("==> Sequence dataset ...")
            X = data[:,:19]
            y = data[:,19:]
        elif dataset_type == "1_to_1": 
            print("==> 1 to 1 dataset ...")
            X = data[:,:6]
            y = data[:,6:] #13]
    if robot_choice == "3DoF-3R":
        if dataset_type == "seq":
            print("==> Sequence dataset ...")
            X = data[:,:7]
            y = data[:,7:]
        elif dataset_type == "1_to_1": 
            print("==> 1 to 1 dataset ...")
            X = data[:,:2]
            y = data[:,2:5] #13]


    # split in train and test sets    
    X_train_priv, X_validate_priv, y_train, y_validate = train_test_split(X, 
                                                                        y, 
                                                                        test_size = 0.1,
                                                                        random_state = 1)

    X_train_priv, X_test_priv, y_train, y_test = train_test_split(X_train_priv, 
                                                                y_train, 
                                                                test_size = 0.1,
                                                                random_state = 1)
        
    
    sc_in = MinMaxScaler(copy=True, feature_range=(0, 1))
    sc_out = MinMaxScaler(copy=True, feature_range=(0, 1))
    
    X_train_priv = sc_in.fit_transform(X_train_priv)
    X_validate_priv = sc_in.transform(X_validate_priv) 
    X_test_priv = sc_in.transform(X_test_priv) 
    
    print("==> Shape X_train: ", X_train_priv.shape)
    print("==> Shape y_train: ", y_train.shape)

    train_data_priv = LoadIKDataset(X_train_priv, y_train, device)
    test_data_priv = LoadIKDataset(X_validate_priv, y_validate, device)

    train_data_loader_priv = DataLoader(dataset=train_data_priv,
                                   batch_size=batch_size,
                                   shuffle=True,
                                   drop_last=True,
                                   pin_memory=False,
                                   num_workers=8,
                                   persistent_workers=True)

    test_data_loader_priv = DataLoader(dataset=test_data_priv,
                                   batch_size=batch_size,
                                   drop_last=False,
                                   shuffle=False,
                                   pin_memory=False,
                                   num_workers=8,
                                   persistent_workers=True)
    


    X_train_dist = X_train_priv[:,13:]
    X_validate_dist = X_validate_priv[:,13:]
    X_test_dist = X_test_priv[:,13:]
    train_data_dist = LoadIKDataset(X_train_dist, y_train, device)
    test_data_dist = LoadIKDataset(X_validate_dist, y_validate, device)

    train_data_loader_dist = DataLoader(dataset=train_data_dist,
                                   batch_size=batch_size,
                                   shuffle=True,
                                   drop_last=True,
                                   pin_memory=False,
                                   num_workers=8,
                                   persistent_workers=True)

    test_data_loader_dist = DataLoader(dataset=test_data_dist,
                                   batch_size=batch_size,
                                   drop_last=False,
                                   shuffle=False,
                                   pin_memory=False,
                                   num_workers=8,
                                   persistent_workers=True)
    


    output = {
        "train_data_loader_priv": train_data_loader_priv,
        "test_data_loader_priv": test_data_loader_priv,
        "X_validate_priv": X_validate_priv,
        "y_validate": y_validate,
        "X_train_priv": X_train_priv,
        "y_train": y_train,
        "X_test_priv": X_test_priv,
        "y_test": y_test,
        "train_data_loader_dist": train_data_loader_dist,
        "test_data_loader_dist": test_data_loader_dist,
        "X_validate_dist": X_validate_dist,
        "X_train_dist": X_train_dist,
        "X_test_dist": X_test_dist
    }


    return output




# function to load the dataset
def load_test_dataset(X_test, y_test, device):

    print("==> Shape X_test: ", X_test.shape)
    print("==> Shape y_test: ", y_test.shape)

    test_data = LoadIKDataset(X_test, y_test, device)

    test_data_loader = DataLoader(dataset=test_data,
                                   batch_size=1,
                                   drop_last=False,
                                   shuffle=False,
                                   pin_memory=False,
                                   num_workers=8,
                                   persistent_workers=True)

    return test_data_loader

# function to load the dataset
def load_test_dataset_2(X_test, y_test, device, sc_in):

    print("==> Shape X_test: ", X_test.shape)
    print("==> Shape y_test: ", y_test.shape)

    X_test = sc_in.transform(X_test) 

    test_data = LoadIKDataset(X_test, y_test, device)

    test_data_loader = DataLoader(dataset=test_data,
                                   batch_size=1,
                                   drop_last=False,
                                   shuffle=False,
                                   pin_memory=False,
                                   num_workers=8,
                                   persistent_workers=True)

    return test_data_loader



def load_test_all_dataset(X_test, y_test, device, sc_in):

    print("==> Shape X_test: ", X_test.shape)
    print("==> Shape y_test: ", y_test.shape)

    #sc_in = MinMaxScaler(copy=True, feature_range=(0, 1))

    X_test = sc_in.transform(X_test) 

    test_data = LoadIKDataset(X_test, y_test, device)

    test_data_loader = DataLoader(dataset=test_data,
                                   batch_size=1,
                                   drop_last=False,
                                   shuffle=False,
                                   pin_memory=False,
                                   num_workers=3,
                                   persistent_workers=True)

    return test_data_loader





# train function
def train(model, iterator, optimizer, criterion, criterion_type, batch_size, device, epoch, EPOCHS, scheduler, scaler):
    epoch_loss = 0
    count = 0
    count_batch = 0
    model.train()    
    #print("... FKloss Minimization ...")
   
    with tqdm(total=len(iterator), desc='Epoch: [{}/{}]'.format(epoch+1, EPOCHS), disable=True) as t:
        for data in iterator:
            optimizer.zero_grad()
            x, y = data['input'], data['output']
            #x.requires_grad = True
            

            x = x.to(device)
            y = y.to(device)   

            if criterion_type == "mdn":
                y_pred, mdn_params = model(x)
                y_pred = torch.nan_to_num(y_pred, nan=0.0)
                loss = criterion(mdn_params, y)
            else:        
                y_pred, _ = model(x)
                y_pred = torch.nan_to_num(y_pred, nan=0.0)
                loss = criterion(y_pred, y)

            """
            print("Epoch ->: ", epoch)
            print("x: ", torch.isnan(x).any())
            print("y: ", torch.isnan(y).any())
            print("y_pred: ", torch.isnan(y_pred).any())
            print("loss: ", torch.isnan(loss).any())
            print(loss)
            print("\n")
            """

            
            if torch.isnan(loss).any():
                count += 1
                if count == 5:
                    break
            
            

            """
            if criterion_type == "ld":
                criterion_2 = nn.MSELoss(reduction="mean")
                loss_2 = criterion_2(y_pred, y)
                loss = loss + loss_2
            """


            loss.backward()
            
            # Gradient clipping
            #max_norm = 1.0  # Set the maximum gradient norm
            #torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)

            optimizer.step()
            # scheduler.step()  # ReduceLROnPlateau is stepped in the main loop with val loss.           
           
            epoch_loss += loss.item()
            t.set_postfix_str('Train loss: {:.6f}'.format(epoch_loss/len(iterator)))
            t.update()   


            #print(y_pred.isnan().nonzero())
            #print(y.isnan().nonzero())
            #print(loss)
            #print(epoch_loss)

        #print(len(iterator))
        #sys.exit()

            
            
            #print("Here")
            #sys.exit()   
            # 
            count_batch += 1
            #print("Current batch: ", count_batch)  

    return epoch_loss/len(iterator)




# train function
def train_two_stages(fk_model, ik_model, iterator, optimizer, fk_criterion, ik_criterion, batch_size, device, epoch, EPOCHS, scheduler, scaler, robot_choice):
    epoch_loss = 0
    epoch_loss_ik = 0
    epoch_loss_fk = 0
    fk_model.train()    
    ik_model.train()

    #print("... FKloss Minimization ...")
   
    with tqdm(total=len(iterator), desc='Epoch: [{}/{}]'.format(epoch+1, EPOCHS), disable=True) as t:
        for data in iterator:
            optimizer.zero_grad()
            x, y = data['input'], data['output']
            #x.requires_grad = True            

            x = x.to(device)
            y = y.to(device)           
 
            
            y_pred, _ = ik_model(x)
            x_pred, _ = fk_model(y_pred)

            #r_x_pred = reconstruct_FK_pose(y_pred, robot_choice, device)
            
            loss_ik = ik_criterion(y_pred, y)
            loss_fk = fk_criterion(x_pred, x)
            loss = loss_ik + loss_fk  
            #loss = loss_fk  
            loss.backward()
            optimizer.step()
            #scheduler.step()           
           
            epoch_loss += loss.item()
            epoch_loss_ik += loss_ik.item()
            epoch_loss_fk += loss_fk.item()
            t.set_postfix_str('Train loss: {:.6f}'.format(epoch_loss/len(iterator)))
            t.update()   


            #print(y_pred.isnan().nonzero())
            #print(y.isnan().nonzero())
            #print(loss)
            #print(epoch_loss)

        #print(len(iterator))
        #sys.exit()

            
            
            #print("Here")
            #sys.exit()     

    return epoch_loss/len(iterator), epoch_loss_fk/len(iterator), epoch_loss_ik/len(iterator)


def train_keep(model, iterator, optimizer, criterion, criterion_type, batch_size, device, epoch, EPOCHS, scheduler, scaler):
    epoch_loss = 0
    epoch_loss_2 = 0
    model.train()
    i = 0



    if criterion_type == "ld":
        criterion_2 = nn.MSELoss(reduction="mean")

    #B_dict = {}
    #B_dict['basic'] = torch.eye(32,3)
    #B = B_dict['basic'].to(device)
    
    #with tqdm(total=(len(iterator) - len(iterator) % batch_size)) as t:
    with tqdm(total=len(iterator), desc='Epoch: [{}/{}]'.format(epoch+1, EPOCHS), disable=True) as t:
        for data in iterator:
        #for data in tqdm(iterator, desc="Training", leave=False):
            optimizer.zero_grad()
            x, y = data['input'], data['output']
            x.requires_grad = True

            x = x.to(device)
            y = y.to(device)
            
            #x = input_mapping(x,B)

            """
            y_pred, _ = model(x)
            loss = criterion(y_pred, y)
            
            loss.backward()
            #torch.nn.utils.clip_grad_norm_(model.parameters(), 10)
            optimizer.step()
            #scheduler.step()
            """
            
            
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                y_pred, _ = model(x)

                loss = criterion(y_pred, y)

                #loss = criterion(torch.sin(y_pred), torch.sin(y)) + criterion(torch.cos(y_pred), torch.cos(y))

                
                #if criterion_type == "ld":
                #    #loss = criterion(y_pred, x)
                #    wp = 1
                #    wq = 0
                #    loss = wp*criterion(y_pred, y)+wq*criterion_2(y_pred, y)
                #else:
                #    loss = criterion(y_pred, y)
                
                

            #print("y_pred\n:", y_pred)

            #print(x.shape)
            #print(y.shape)
            #print(y_pred.shape)
            #sys.exit()

            #print("\nTrain Epoch {} at batch {}".format(epoch, i))
            
            #if i == 1:
            #    print("\nTrain Epoch {} at batch {}".format(epoch, i))
            #    print(y_pred[:5,:])
            #    print(y[:5,:])
            #    #sys.exit()
            
            
            # optimizer.zero_grad()
            #loss = criterion(y_pred, y)

            #print(x)
            #print(y_pred) 
            

            #make_dot(loss, params=dict(list(model.named_parameters()))).render("loss", format="png")
            
            #loss.backward()
            #torch.nn.utils.clip_grad_norm_(model.parameters(), 10)
            #optimizer.step()
            #scheduler.step()

            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            #scheduler.step()
            scaler.update()
            
            

            epoch_loss += loss.item()
            #if criterion_type == "ld":
                #epoch_loss_2 += loss2.item()
            t.set_postfix_str('Train loss: {:.6f}'.format(epoch_loss/len(iterator)))
            t.update()

            i += 1

            #sys.exit()
    
    
    
    #print("Total batches {}".format(i))
        #if criterion_type == "ld":
            #print('\n\tTrain FK Loss: {}'.format(epoch_loss/len(iterator)))
            #print('\tTrain L2 Loss: {}'.format(epoch_loss_2/len(iterator)))
    return epoch_loss/len(iterator)



def train_sobolev(model, iterator, optimizer, criterion, criterion_type, batch_size, device, epoch, EPOCHS, scheduler, scaler):
    epoch_loss_q = 0
    epoch_loss_J = 0
    epoch_loss = 0
    model.train()
    i = 0
    
    
    with tqdm(total=len(iterator), desc='Epoch: [{}/{}]'.format(epoch+1, EPOCHS), disable=True) as t:
        for data in iterator:
            optimizer.zero_grad()
            x, y = data['input'], data['output']
            x.requires_grad = True

            x = x.to(device)
            y = y.to(device)
            
            #x = input_mapping(x,B)
            
            #with torch.autocast(device_type='cuda', dtype=torch.float16):
            y_pred = model(x)

            #print("Here 1:")
        

            # compute the Network jacobian
            """
            y_pred_J = get_network_jacobian(x, y_pred, device)
            #print(y_pred_J.shape)
            #sys.exit()
            y_pred_J = torch.flatten(y_pred_J, start_dim = 1).to(device)

            #y_pred_J = torch.autograd.functional.jacobian(model, x[0,:])
            #y_pred_J = y_pred_J.permute(1, 0)
            #print(y_pred_J)
            #print(y_pred_J.shape)
            """
            reshape_size = x.shape[0]
            #print(reshape_size)

            """
            # Method 2 to compute the Jacobian
            y_pred_J = torch.autograd.functional.jacobian(model, x)                
            y_pred_J = y_pred_J[y_pred_J.sum(dim=3) != 0]                
            y_pred_J = torch.reshape(y_pred_J, (reshape_size, 7, 6))
            y_pred_J = y_pred_J.permute(0,2,1)             
            print(y_pred_J[0])
            print(y_pred_J.shape)
            y_pred_J = torch.flatten(y_pred_J, start_dim = 1).to(device)
            """
            
            
            
            """
            # Method 3 to compute the Jacobian
            y_pred_J = torch.autograd.functional.jacobian(model, x)
            y_pred_J_True = torch.zeros(reshape_size, 7, 6)
            for i in range(reshape_size):
                y_pred_J_True[i,:,:] = y_pred_J[i,:,i,:].permute(1,0)

            #print(y_pred_J_True[0])
            #print(y_pred_J_True.shape)
            #sys.exit()  
                

            y_pred_J = torch.flatten(y_pred_J_True, start_dim = 1).to(device)
            

            #print("Here 2:")
            
            
            
            # compare the joints    
            loss_q = criterion(y_pred, y[:,:6])

            # compare the jacobians
            loss_J = criterion(y_pred_J, y[:,6:])

            # total loss
            loss = loss_q + loss_J
            """

            loss_q = criterion(y_pred, y[:,:6])
            loss_J = loss_q
            loss = loss_q
            

                
           
            #make_dot(loss, params=dict(list(model.named_parameters()))).render("loss", format="png")
            
            loss.backward()
            #torch.nn.utils.clip_grad_norm_(model.parameters(), 10)
            optimizer.step()
            #scheduler.step()


            #scaler.scale(loss).backward()
            #scaler.step(optimizer)
            #scheduler.step()
            #scaler.update()

            epoch_loss += loss.item()
            epoch_loss_q += loss_q.item()
            epoch_loss_J += loss_J.item()
            #if criterion_type == "ld":
                #epoch_loss_2 += loss2.item()
            t.set_postfix_str('Train loss: {:.6f} - Train loss q: {:.6f} - Train loss J: {:.6f}'.format(epoch_loss/len(iterator), 
                                                                                                        epoch_loss_q/len(iterator),
                                                                                                        epoch_loss_J/len(iterator)))
            t.update()

            i += 1

            #sys.exit()
    
    
    
    #print("Total batches {}".format(i))
        #if criterion_type == "ld":
            #print('\n\tTrain FK Loss: {}'.format(epoch_loss/len(iterator)))
            #print('\tTrain L2 Loss: {}'.format(epoch_loss_2/len(iterator)))
    return epoch_loss/len(iterator), epoch_loss_q/len(iterator), epoch_loss_J/len(iterator)






def get_network_jacobian(inputs, output_poses, device):

            # compute the Jacobian
            batch = inputs.shape[0]
            input_size = inputs.shape[1]
            output_size = output_poses.shape[1] #* self.dim_position
            

            # initialize a tensor to hold the Jacobian
            J = torch.zeros(batch, 1 , input_size, output_size)
            #print('J: ', J.shape)
            #print('output_size: ', output_size)
            #print('output_poses: ', output_poses.shape)
            #print('inputs: ', inputs.shape)

            t = time.time()
            for j in range(output_size):
                g = torch.autograd.grad(output_poses[:,j], 
                                        inputs, 
                                        grad_outputs=torch.ones_like(output_poses[:,j]).to(device),
                                        retain_graph=True)
                g = g[0].permute(1,0)
                g = torch.reshape(g, (batch, 1, input_size))
                J[:,:,:,j] = g
                #print('g{}: {}'.format(j, g))


            #print('g: ', g.shape)
            #print('J: ', J.shape)

            J_reshape = torch.reshape(J, (batch, -1, input_size))
            #print(J[0,:,:,0])
            #print(J[0,:,:,1])
            #print(J_reshape[0,:,:])
            #print('J_reshape: ', J_reshape.shape)

            J_reshape = J_reshape.permute(0, 2, 1) 
            #print('J_reshape: ', J_reshape.shape)
            #print(J_reshape[0,:,:])
            
            return J_reshape


def train_back(model, iterator, optimizer, criterion, criterion_type, batch_size, device, epoch, EPOCHS, scheduler, scaler):
    epoch_loss = 0
    epoch_loss_2 = 0
    model.train()
    i = 0



    if criterion_type == "ld":
        criterion_2 = nn.MSELoss(reduction="mean")

    #B_dict = {}
    #B_dict['basic'] = torch.eye(32,3)
    #B = B_dict['basic'].to(device)
    
    #with tqdm(total=(len(iterator) - len(iterator) % batch_size)) as t:
    with tqdm(total=len(iterator), desc='Epoch: [{}/{}]'.format(epoch+1, EPOCHS), disable=True) as t:
        for data in iterator:
        #for data in tqdm(iterator, desc="Training", leave=False):
            optimizer.zero_grad()
            x, y = data['input'], data['output']
            x.requires_grad = True

            x = x.to(device)
            y = y.to(device)
            
            #x = input_mapping(x,B)
            
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                y_pred, _ = model(x)

                if criterion_type == "ld":
                    #loss = criterion(y_pred, x)
                    wp = 1
                    wq = 0
                    loss = wp*criterion(y_pred, y)+wq*criterion_2(y_pred, y)
                else:
                    loss = criterion(y_pred, y)

            #print("y_pred\n:", y_pred)

            #print(x.shape)
            #print(y.shape)
            #print(y_pred.shape)
            #sys.exit()

            #print("\nTrain Epoch {} at batch {}".format(epoch, i))
            """
            if i == 1:
                print("\nTrain Epoch {} at batch {}".format(epoch, i))
                print(y_pred[:5,:])
                print(y[:5,:])
                #sys.exit()
            """
            
            # optimizer.zero_grad()
            #loss = criterion(y_pred, y)

            #print(x)
            #print(y_pred) 
            

            #make_dot(loss, params=dict(list(model.named_parameters()))).render("loss", format="png")
            
            #loss.backward()
            #torch.nn.utils.clip_grad_norm_(model.parameters(), 10)
            #optimizer.step()
            #scheduler.step()


            scaler.scale(loss).backward()
            scaler.step(optimizer)
            #scheduler.step()
            scaler.update()

            epoch_loss += loss.item()
            #if criterion_type == "ld":
                #epoch_loss_2 += loss2.item()
            t.set_postfix_str('Train loss: {:.6f}'.format(epoch_loss/len(iterator)))
            t.update()

            i += 1

            #sys.exit()
    
    
    
    #print("Total batches {}".format(i))
        #if criterion_type == "ld":
            #print('\n\tTrain FK Loss: {}'.format(epoch_loss/len(iterator)))
            #print('\tTrain L2 Loss: {}'.format(epoch_loss_2/len(iterator)))
    return epoch_loss/len(iterator)


def train_dist(model_priv, model_dist, iterator_priv, iterator_dist, optimizer_dist, criterion, criterion_type, batch_size, device, epoch, EPOCHS, scheduler, scaler, alpha):
    epoch_loss = 0
    epoch_loss_2 = 0
    model_priv.eval()
    model_dist.train()
    i = 0
    



    if criterion_type == "ld":
        criterion_2 = nn.MSELoss(reduction="mean")

    #B_dict = {}
    #B_dict['basic'] = torch.eye(32,3)
    #B = B_dict['basic'].to(device)

    
    #with tqdm(total=(len(iterator) - len(iterator) % batch_size)) as t:
    with tqdm(total=len(iterator_dist), desc='Epoch: [{}/{}]'.format(epoch+1, EPOCHS), disable=True) as t:
        for (data_dist, data_priv) in zip (iterator_dist, iterator_priv):
        #for data in tqdm(iterator, desc="Training", leave=False):
            optimizer_dist.zero_grad()

            # for distillation phase
            x_dist, y_dist = data_dist['input'], data_dist['output']
            x_dist.requires_grad = True
            x_dist = x_dist.to(device)
            y_dist = y_dist.to(device)


            # from privilege phase
            x_priv, y_priv = data_priv['input'], data_priv['output']
            x_priv = x_priv.to(device)
            y_priv = y_priv.to(device)

            #print(x_dist[0,:],y_dist[0,:])
            #print(x_priv[0,:],y_priv[0,:])
            #sys.exit()
            
            with torch.no_grad():
                y_pred_priv, _ = model_priv(x_priv)
            
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                y_pred_dist, _ = model_dist(x_dist)

                loss_dist = criterion(y_pred_dist, y_dist)
                loss_imit = criterion(y_pred_dist, y_pred_priv)
                loss = (1-alpha)*loss_dist + alpha*loss_imit           
           

            #make_dot(loss, params=dict(list(model.named_parameters()))).render("loss", format="png")
            
            #loss.backward()
            #torch.nn.utils.clip_grad_norm_(model.parameters(), 10)
            #optimizer.step()
            #scheduler.step()


            scaler.scale(loss).backward()
            scaler.step(optimizer_dist)
            #scheduler.step()
            scaler.update()

            epoch_loss += loss.item()
            #if criterion_type == "ld":
                #epoch_loss_2 += loss2.item()
            t.set_postfix_str('Train loss: {:.6f}'.format(epoch_loss/len(iterator_dist)))
            t.update()

            i += 1

            #sys.exit()
    
    
    
    #print("Total batches {}".format(i))
        #if criterion_type == "ld":
            #print('\n\tTrain FK Loss: {}'.format(epoch_loss/len(iterator)))
            #print('\tTrain L2 Loss: {}'.format(epoch_loss_2/len(iterator)))
    return epoch_loss/len(iterator_dist)

# evaluation function 
def evaluate(model, iterator, criterion, criterion_type, device, epoch, EPOCHS):
    epoch_loss = 0
    model.eval()

    if criterion_type == "ld":
        criterion_2 = nn.MSELoss(reduction="mean")

    #B_dict = {}
    #B_dict['basic'] = torch.eye(32,3)
    #B = B_dict['basic'].to(device)
    
    with torch.no_grad():
        #for data in tqdm(iterator, desc="Evaluating", leave=False):        
        with tqdm(total=len(iterator), desc='Epoch: [{}/{}]'.format(epoch+1, EPOCHS), disable=True) as t:
            for data in iterator:
                x = data['input'].to(device)
                y = data['output'].to(device)
                #x.requires_grad = True

                #x = input_mapping(x,B)
                
                if criterion_type == "mdn":
                    y_pred, mdn_params = model(x)
                    y_pred = torch.nan_to_num(y_pred, nan=0.0)
                    loss = criterion(mdn_params, y)
                else:
                    y_pred, _ = model(x)
                    y_pred = torch.nan_to_num(y_pred, nan=0.0)
                    loss = criterion(y_pred, y)

                """
                if criterion_type == "ld":
                    criterion_2 = nn.MSELoss(reduction="mean")
                    loss_2 = criterion_2(y_pred, y)
                    loss = loss + loss_2
                """
                
                

                #loss = criterion(y_pred, x)  
                
                #loss = criterion(torch.sin(y_pred), torch.sin(y)) + criterion(torch.cos(y_pred), torch.cos(y))

                """
                if criterion_type == "ld":
                    #loss = criterion(y_pred, x)
                    wp = 1
                    wq = 0
                    loss = wp*criterion(y_pred, y)+wq*criterion_2(y_pred, y)
                else:
                    loss = criterion(y_pred, y)
                """
                
                
                epoch_loss += loss.item()
    
                t.set_postfix_str('Valid loss: {:.6f}'.format(epoch_loss/len(iterator)))
                t.update()

    return epoch_loss/len(iterator)



def evaluate_two_stages(fk_model, ik_model, iterator, fk_criterion, ik_criterion, device, epoch, EPOCHS, robot_choice):
    epoch_loss = 0
    epoch_loss_ik = 0
    epoch_loss_fk = 0
    fk_model.eval()
    ik_model.eval()

    
    with torch.no_grad():
        #for data in tqdm(iterator, desc="Evaluating", leave=False):        
        with tqdm(total=len(iterator), desc='Epoch: [{}/{}]'.format(epoch+1, EPOCHS), disable=True) as t:
            for data in iterator:
                x = data['input'].to(device)
                y = data['output'].to(device)          
 
            
                y_pred, _ = ik_model(x)
                x_pred, _ = fk_model(y_pred)

                #r_x_pred = reconstruct_FK_pose(y_pred, robot_choice, device)
                
                loss_ik = ik_criterion(y_pred, y)
                loss_fk = fk_criterion(x_pred, x)
                loss = loss_ik + loss_fk  
                #loss = loss_fk 
            
                epoch_loss += loss.item()
                epoch_loss_ik += loss_ik.item()
                epoch_loss_fk += loss_fk.item()
                   
                t.set_postfix_str('Valid loss: {:.6f}'.format(epoch_loss/len(iterator)))
                t.update()

    return epoch_loss/len(iterator), epoch_loss_fk/len(iterator), epoch_loss_ik/len(iterator)



def evaluate_sobolev(model, iterator, criterion, criterion_type, device, epoch, EPOCHS):
    epoch_loss = 0
    epoch_loss_q = 0
    epoch_loss_J = 0
    model.eval()
    
    with torch.no_grad():
        #for data in tqdm(iterator, desc="Evaluating", leave=False):        
        with tqdm(total=len(iterator), desc='Epoch: [{}/{}]'.format(epoch+1, EPOCHS), disable=True) as t:
            for data in iterator:
                x = data['input'].to(device)
                y = data['output'].to(device)
                
                y_pred = model(x)   
                
                loss = criterion(y_pred, y[:,:6])
                
                epoch_loss += loss.item()
                #if criterion_type == "ld":
                    #epoch_loss_2 += loss2.item()
                t.set_postfix_str('Valid loss: {:.6f}'.format(epoch_loss/len(iterator)))
                t.update()
                t.update()

    return epoch_loss/len(iterator)

# make predictions
def inference(model, iterator, criterion, device, robot_choice):
    model.eval()
    y_preds = []
    y_desireds = []
    X_desireds = []
    
    for data in iterator:
        x = data['input'].to(device)
        y = data['output'].to(device)

        #x = input_mapping(x,B)
        
        y_pred, _ = model(x)
        y_preds.append(y_pred.detach().cpu().numpy().squeeze())
        y_desireds.append(y.detach().cpu().numpy().squeeze())
        #X_desireds.append(x.detach().cpu().numpy().squeeze())


    y_desireds = np.array(y_desireds)
    #X_desireds = np.array(X_desireds)
    X_desireds = reconstruct_pose(y_desireds, robot_choice)
    y_preds = np.array(y_preds)
    X_preds = reconstruct_pose(y_preds, robot_choice)

    #print(X_preds.shape)
    #print(X_desireds.shape)
    
    X_errors = np.abs(X_preds - X_desireds)
    y_errors = np.abs(y_preds - y_desireds)

    #print(X_errors.shape)

    X_errors_report = np.array([[X_errors.min(axis=0)],
                                [X_errors.mean(axis=0)],
                                [X_errors.max(axis=0)],
                                [X_errors.std(axis=0)]]).squeeze()
    
    results = {
        "y_preds": y_preds,
        "X_preds": X_preds,
        "y_desireds": y_desireds,
        "X_desireds": X_desireds,
        "X_errors": X_errors_report
    }
    return results


def inference_modified(model, iterator, criterion, device, robot_choice):
    model.eval()
    y_preds = []
    y_desireds = []
    X_desireds = []
    
    for data in iterator:
        x = data['input'].to(device)
        y = data['output'].to(device)

        #x = input_mapping(x,B)
        
        y_pred, _ = model(x)
        y_preds.append(y_pred.detach().cpu().numpy().squeeze())
        y_desireds.append(y.detach().cpu().numpy().squeeze())
        #X_desireds.append(x.detach().cpu().numpy().squeeze())


    y_desireds = np.array(y_desireds)
    y_preds = np.array(y_preds)
    #X_desireds = np.array(X_desireds)
    X_desireds, X_preds, X_errors = reconstruct_pose_modified(y_desireds, y_preds, robot_choice)
    
    X_errors_report = np.array([[X_errors.min(axis=0)],
                                [X_errors.mean(axis=0)],
                                [X_errors.max(axis=0)],
                                [X_errors.std(axis=0)]]).squeeze()
    
    results = {
        "y_preds": y_preds,
        "X_preds": X_preds,
        "y_desireds": y_desireds,
        "X_desireds": X_desireds,
        "X_errors": X_errors,
        "X_errors_report": X_errors_report
    }
    return results





def inference_modified_best_of_k(model, iterator, criterion, device, robot_choice,
                                pose_slice=(6, 12),  # indices of target pose in x for LBE seq: [pose_e(0:6), pose(6:12), q_e(12:)]
                                use_position_only=False):
    """
    Best-of-K inference for MDN models:
    - Uses mu (K anchors) as candidates
    - Selects the candidate minimizing FK pose error to the desired pose

    pose_slice: tuple(start, end) giving the target pose indices in x (CPU numpy)
    use_position_only: if True, select by xyz only; else use full (xyz+rpy) error norm
    """
    model.eval()
    y_preds = []
    y_desireds = []

    count_false = 0
    count_true = 0

    with torch.no_grad():
        for data in tqdm(iterator):
            x = data['input'].to(device)    # [B, input_dim]
            y = data['output'].to(device)   # [B, D]

            # Forward: MDN returns (y_pred_dummy, mdn_params)
            _, mdn_params = model(x)
            logits_pi, mu, log_sigma = mdn_params  # mu: [B, K, D]

            B, K, D = mu.shape

            # ----- Build candidates: [B*K, D] -----
            cand_y = mu.reshape(B * K, D)  # joints candidates

            # ----- Desired joints replicated K times: [B*K, D] -----
            y_rep = y.unsqueeze(1).expand(B, K, D).reshape(B * K, D)

            # Convert to numpy for your existing reconstruct_pose_modified
            cand_y_np = cand_y.detach().cpu().numpy()
            y_rep_np = y_rep.detach().cpu().numpy()

            # Compute FK errors for all candidates using your existing routine
            # X_errors shape should be [B*K, 6] (xyz + rpy in your code)
            _, _, X_errors_all = reconstruct_pose_modified(y_rep_np, cand_y_np, robot_choice)

            

            # ----- Reduce candidate errors to a scalar per candidate -----
            # If X_errors_all is [N,6], make a scalar error per row
            if use_position_only:
                # position only (xyz)
                err_scalar = np.linalg.norm(X_errors_all[:, :3], axis=1)  # [B*K]
            else:
                """
                # full pose error (xyz + rpy)
                err_scalar = np.linalg.norm(X_errors_all, axis=1)         # [B*K]
                #print(X_errors_all)
                #print(np.linalg.norm(X_errors_all, axis=1))
                #print(np.mean(X_errors_all, axis=1))
                
                best_pi = torch.argmax(logits_pi, dim=1)
                best_pi_np = best_pi.detach().cpu().numpy()
                #print(best_pi_np)
                """
                # X_errors_all: [B*K, 6]  -> [dx, dy, dz, droll, dpitch, dyaw]
                pos = X_errors_all[:, :3]   # meters
                rot = X_errors_all[:, 3:]   # radians

                # Paper strategy: 2 rad == 1 m  =>  1 rad == 0.5 m
                rot_m = 0.5 * rot

                # scalar metric in "meters-equivalent"
                err_scalar = np.sqrt(np.sum(pos**2, axis=1) + np.sum(rot_m**2, axis=1))  # [B*K]
                
                


            # ----- Pick best candidate per original sample -----
            err_scalar = err_scalar.reshape(B, K)         # [B, K]
            best_k = np.argmin(err_scalar, axis=1)        # [B]
            #print(best_k)
            #same = (best_pi_np == best_k)
            #print(f"same: {same}\n")
            #if same:
            #    count_true = count_true + 1
            #else:
            #    count_false = count_false + 1



            # Gather best joints from mu
            mu_np = mu.detach().cpu().numpy()             # [B, K, D]
            y_best_np = mu_np[np.arange(B), best_k, :]    # [B, D]

            # Append results
            y_preds.append(y_best_np)
            y_desireds.append(y.detach().cpu().numpy())

    # Stack batches
    y_preds = np.concatenate(y_preds, axis=0)
    y_desireds = np.concatenate(y_desireds, axis=0)

    # Recompute pose errors for the selected best predictions (your normal pipeline)
    X_desireds, X_preds, X_errors = reconstruct_pose_modified(y_desireds, y_preds, robot_choice)
    print(X_preds.shape)
    print(y_desireds.shape)
    print(y_preds.shape)
    sys.exit()

    X_errors_report = np.array([[X_errors.min(axis=0)],
                                [X_errors.mean(axis=0)],
                                [X_errors.max(axis=0)],
                                [X_errors.std(axis=0)]]).squeeze()

    results = {
        "y_preds": y_preds,
        "X_preds": X_preds,
        "y_desireds": y_desireds,
        "X_desireds": X_desireds,
        "X_errors": X_errors,
        "X_errors_report": X_errors_report,
        "best_of_k": True,
        "K": int(K),
    }

    #print(f"same: {count_true}\n")
    #print(f"not same: {count_false}\n")
    #sys.exit()
    return results



def inference_modified_return_all_k(
    model,
    iterator,
    criterion,
    device,
    robot_choice,
    use_position_only=False,
):
    """
    MDN inference returning ALL K IK solutions (mu) per sample, plus best-of-K selection.

    IMPORTANT: reconstruct_pose_modified(...) returns X_des_flat and X_preds_flat as (B*K, D+6)
    because it concatenates [X,Y,Z,R,P,Y] + [q1..qD] per row. We therefore reshape using (D+6)
    and slice the first 6 columns as pose.

    Returns:
      y_preds_all        : (N, K, D)   all IK joint solutions (anchors)
      y_desireds         : (N, D)      ground-truth joints
      y_desireds_all     : (N, K, D)   ground-truth joints replicated

      X_preds_all        : (N, K, 6)   FK poses (xyz+rpy) for each IK solution
      X_desireds         : (N, 6)      desired poses
      X_desireds_all     : (N, K, 6)   desired poses replicated

      X_errors_all       : (N, K, 6)   pose errors per IK solution (m/rad)
      err_scalar_all     : (N, K)      scalar pose error used for selection
      best_k             : (N,)        index of best candidate per sample

      y_preds            : (N, D)      best-of-K joint solution
      X_preds            : (N, 6)      FK pose of best-of-K
      X_errors           : (N, 6)      pose errors of best-of-K
      X_errors_report    : (4, 6)      min/mean/max/std over test set (best-of-K)
    """
    model.eval()

    # ---- Accumulators ----
    y_desireds_list = []

    y_preds_all_list = []        # (B,K,D)
    X_preds_all_list = []        # (B,K,6)
    X_errors_all_list = []       # (B,K,6)
    err_scalar_all_list = []     # (B,K)
    best_k_list = []             # (B,)
    y_best_list = []             # (B,D)

    with torch.no_grad():
        for data in tqdm(iterator):
            x = data["input"].to(device)      # (B, input_dim)
            y = data["output"].to(device)     # (B, D)

            # Forward pass (MDN)
            _, mdn_params = model(x)
            logits_pi, mu, log_sigma = mdn_params   # mu: (B,K,D)
            B, K, D = mu.shape

            # ---- Candidate joints ----
            cand_y = mu.reshape(B * K, D)            # (B*K, D)
            y_rep = y.unsqueeze(1).expand(B, K, D).reshape(B * K, D)  # (B*K, D)

            cand_y_np = cand_y.detach().cpu().numpy()
            y_rep_np = y_rep.detach().cpu().numpy()

            # ---- FK + pose error for all candidates ----
            # X_des_flat and X_preds_flat are (B*K, D+6) (pose + joints)
            X_des_flat, X_preds_flat, X_errors_flat = reconstruct_pose_modified(
                y_rep_np, cand_y_np, robot_choice
            )

            X_des_flat = np.asarray(X_des_flat)
            X_preds_flat = np.asarray(X_preds_flat)
            X_errors_flat = np.asarray(X_errors_flat)

            # last_dim should be D+6
            last_dim = X_preds_flat.shape[-1]

            # Reshape pose+joints to (B,K,D+6)
            X_des_k_full = X_des_flat.reshape(B, K, last_dim)
            X_preds_k_full = X_preds_flat.reshape(B, K, last_dim)

            # Slice pose only (first 6)
            X_des_k = X_des_k_full[:, :, :6]     # (B,K,6)
            X_preds_k = X_preds_k_full[:, :, :6] # (B,K,6)

            # Errors might be (B*K,6) or (B*K,D+6); handle both robustly then slice pose part
            if X_errors_flat.ndim == 2 and X_errors_flat.shape[1] == 6:
                X_errors_k = X_errors_flat.reshape(B, K, 6)
            else:
                X_errors_k = X_errors_flat.reshape(B, K, last_dim)[:, :, :6]

            # ---- Scalar error (for best-of-K) ----
            if use_position_only:
                # Position only (meters)
                pos = X_errors_k[:, :, :3]
                err_scalar = np.linalg.norm(pos, axis=2)  # (B,K)
            else:
                # Position + orientation with unit matching: 2 rad == 1 m
                pos = X_errors_k[:, :, :3]          # meters
                rot = X_errors_k[:, :, 3:]          # radians
                rot_m = 0.5 * rot                   # meters-equivalent
                err_scalar = np.sqrt(
                    np.sum(pos**2, axis=2) + np.sum(rot_m**2, axis=2)
                )                                   # (B,K)

            # ---- Best-of-K selection ----
            best_k = np.argmin(err_scalar, axis=1)   # (B,)

            # Gather best joints from mu
            mu_np = mu.detach().cpu().numpy()        # (B,K,D)
            y_best = mu_np[np.arange(B), best_k, :]  # (B,D)

            # ---- Accumulate ----
            y_desireds_list.append(y.detach().cpu().numpy())

            y_preds_all_list.append(mu_np)       # (B,K,D)
            X_preds_all_list.append(X_preds_k)   # (B,K,6)
            X_errors_all_list.append(X_errors_k) # (B,K,6)
            err_scalar_all_list.append(err_scalar)
            best_k_list.append(best_k)
            y_best_list.append(y_best)

    # ---- Stack across batches ----
    y_desireds = np.concatenate(y_desireds_list, axis=0)           # (N,D)
    y_preds_all = np.concatenate(y_preds_all_list, axis=0)         # (N,K,D)
    X_preds_all = np.concatenate(X_preds_all_list, axis=0)         # (N,K,6)
    X_errors_all = np.concatenate(X_errors_all_list, axis=0)       # (N,K,6)
    err_scalar_all = np.concatenate(err_scalar_all_list, axis=0)   # (N,K)
    best_k = np.concatenate(best_k_list, axis=0)                   # (N,)
    y_preds = np.concatenate(y_best_list, axis=0)                  # (N,D)

    # ---- Desired replicated ----
    K = y_preds_all.shape[1]
    y_desireds_all = np.repeat(y_desireds[:, None, :], K, axis=1)  # (N,K,D)

    # ---- Recompute FK for best-of-K (standard reporting) ----
    X_desireds, X_preds, X_errors = reconstruct_pose_modified(
        y_desireds, y_preds, robot_choice
    )

    # X_desireds might also be (N, D+6) depending on your reconstruct_pose_modified; slice pose
    X_desireds = np.asarray(X_desireds)
    X_preds = np.asarray(X_preds)
    X_errors = np.asarray(X_errors)

    # If X_desireds includes joints too, slice first 6
    if X_desireds.ndim == 2 and X_desireds.shape[1] >= 6:
        X_desireds_pose = X_desireds[:, :6]
    else:
        X_desireds_pose = X_desireds

    if X_preds.ndim == 2 and X_preds.shape[1] >= 6:
        X_preds_pose = X_preds[:, :6]
    else:
        X_preds_pose = X_preds

    if X_errors.ndim == 2 and X_errors.shape[1] >= 6:
        X_errors_pose = X_errors[:, :6]
    else:
        X_errors_pose = X_errors

    X_desireds_all = np.repeat(X_desireds_pose[:, None, :], K, axis=1)  # (N,K,6)

    X_errors_report = np.array([
        X_errors_pose.min(axis=0),
        X_errors_pose.mean(axis=0),
        X_errors_pose.max(axis=0),
        X_errors_pose.std(axis=0),
    ])

    # ---- Final results ----
    results = {
        # Joint space
        "y_preds_all": y_preds_all,              # (N,K,D)
        "y_desireds": y_desireds,                # (N,D)
        "y_desireds_all": y_desireds_all,        # (N,K,D)

        # Task space (pose only: 6D)
        "X_preds_all": X_preds_all,              # (N,K,6)
        "X_desireds": X_desireds_pose,           # (N,6)
        "X_desireds_all": X_desireds_all,        # (N,K,6)
        "X_errors_all": X_errors_all,            # (N,K,6)
        "err_scalar_all": err_scalar_all,        # (N,K)

        # Best-of-K
        "best_k": best_k,                        # (N,)
        "y_preds": y_preds,                      # (N,D)
        "X_preds": X_preds_pose,                 # (N,6)
        "X_errors": X_errors_pose,               # (N,6)
        "X_errors_report": X_errors_report,      # (4,6)

        "best_of_k": True,
        "K": int(K),
    }

    return results



def print_inference_results(results, max_k_show=5, precision=4):
    np.set_printoptions(precision=precision, suppress=True)

    print("\n" + "="*80)
    print("INFERENCE RESULTS SUMMARY")
    print("="*80)

    print(f"Best-of-K enabled: {results.get('best_of_k', False)}")
    print(f"K (number of IK candidates): {results.get('K', 'N/A')}")
    print("-"*80)

    # --------------------
    # Joint space
    # --------------------
    print("\n[JOINT SPACE]")
    print("-"*40)

    y_preds_all = results["y_preds_all"]
    y_desireds = results["y_desireds"]
    y_desireds_all = results["y_desireds_all"]
    y_preds = results["y_preds"]

    print(f"y_preds_all      shape: {y_preds_all.shape}  (N,K,D)")
    print(f"y_desireds       shape: {y_desireds.shape}   (N,D)")
    print(f"y_desireds_all   shape: {y_desireds_all.shape} (N,K,D)")
    print(f"y_preds (best)   shape: {y_preds.shape}      (N,D)")

    print("\nExample (sample 0):")
    print("  y_desireds[0]:")
    print("   ", y_desireds[0])

    print(f"\n  y_preds_all[0, :{max_k_show}]:")
    for k in range(min(max_k_show, y_preds_all.shape[1])):
        print(f"    k={k}: {y_preds_all[0, k]}")

    # --------------------
    # Task space
    # --------------------
    print("\n[TASK SPACE]")
    print("-"*40)

    X_preds_all = results["X_preds_all"]
    X_desireds = results["X_desireds"]
    X_desireds_all = results["X_desireds_all"]
    X_errors_all = results["X_errors_all"]
    err_scalar_all = results["err_scalar_all"]
    X_preds = results["X_preds"]
    X_errors = results["X_errors"]

    print(f"X_preds_all      shape: {X_preds_all.shape}   (N,K,6)")
    print(f"X_desireds       shape: {X_desireds.shape}    (N,6)")
    print(f"X_desireds_all   shape: {X_desireds_all.shape} (N,K,6)")
    print(f"X_errors_all     shape: {X_errors_all.shape}  (N,K,6)")
    print(f"err_scalar_all   shape: {err_scalar_all.shape} (N,K)")
    print(f"X_preds (best)   shape: {X_preds.shape}       (N,6)")
    print(f"X_errors (best)  shape: {X_errors.shape}      (N,6)")

    print("\nExample (sample 0):")
    print("  X_desireds[0] (xyz rpy):")
    print("   ", X_desireds[0])

    print(f"\n  X_preds_all[0, :{max_k_show}] (xyz rpy):")
    for k in range(min(max_k_show, X_preds_all.shape[1])):
        print(f"    k={k}: {X_preds_all[0, k]}")

    print(f"\n  X_errors_all[0, :{max_k_show}] (dx dy dz dR dP dY):")
    for k in range(min(max_k_show, X_errors_all.shape[1])):
        print(f"    k={k}: {X_errors_all[0, k]}")

    # --------------------
    # Best-of-K
    # --------------------
    print("\n[BEST-OF-K SELECTION]")
    print("-"*40)

    best_k = results["best_k"]

    print(f"best_k shape: {best_k.shape}")
    print(f"best_k[0]: {best_k[0]}")

    print("\nBest solution (sample 0):")
    print("  y_preds[0]:")
    print("   ", y_preds[0])
    print("  X_preds[0]:")
    print("   ", X_preds[0])
    print("  X_errors[0]:")
    print("   ", X_errors[0])

    # --------------------
    # Global error stats
    # --------------------
    print("\n[GLOBAL ERROR STATISTICS] (best-of-K)")
    print("-"*40)

    X_errors_report = results["X_errors_report"]
    labels = ["min", "mean", "max", "std"]
    dof_labels = ["x", "y", "z", "roll", "pitch", "yaw"]

    for i, stat in enumerate(labels):
        vals = X_errors_report[i]
        s = ", ".join([f"{d}:{vals[j]:.4f}" for j, d in enumerate(dof_labels)])
        print(f"{stat:>4}: {s}")

    print("\n" + "="*80 + "\n")

    
        # --------------------
    # Rank K solutions by pose error (sample 0)
    # --------------------
    print("\n[RANKING OF K IK SOLUTIONS — SAMPLE 0]")
    print("-"*40)

    X_errors_all = results["X_errors_all"]   # (N,K,6)
    K = results["K"]

    # Select sample 0
    X_err_k = X_errors_all[0]                # (K,6)

    # Convert units
    pos_mm = X_err_k[:, :3] * 1000.0          # meters -> mm
    ori_deg = np.rad2deg(X_err_k[:, 3:])      # radians -> degrees

    # Mean errors
    mean_pos_err = pos_mm.mean(axis=1)        # (K,)
    mean_ori_err = ori_deg.mean(axis=1)       # (K,)

    # Combined score (simple additive metric)
    lambda_ori = 1.0  # weight for orientation (adjust if needed)
    combined_score = mean_pos_err + lambda_ori * mean_ori_err

    # Sort by combined score (lower is better)
    order = np.argsort(combined_score)

    print(f"{'Rank':>4} | {'k':>2} | {'Pos err (mm)':>12} | {'Ori err (deg)':>14} | {'Score':>10}")
    print("-"*60)

    for r, k in enumerate(order):
        print(f"{r+1:>4} | {k:>2} | "
              f"{mean_pos_err[k]:>12.4f} | "
              f"{mean_ori_err[k]:>14.4f} | "
              f"{combined_score[k]:>10.4f}")

    # Highlight best-of-K choice
    best_k = results["best_k"][0]
    print("\nBest-of-K selected index:", best_k)
    print("Best-of-K position error (mm):", mean_pos_err[best_k])
    print("Best-of-K orientation error (deg):", mean_ori_err[best_k])




def inference_modified_all(model, iterator, criterion, device, robot_choice):
    model.eval()
    y_preds = []
    y_desireds = []
    X_desireds = []
    
    for data in iterator:
        x = data['input'].to(device)
        y = data['output'].to(device)

        #x = input_mapping(x,B)
        
        y_pred, _ = model(x)
        y_pred = torch.nan_to_num(y_pred, nan=0.0)

        if '6DoF' in robot_choice:
            y_preds.append(y_pred[:,:6].detach().cpu().numpy().squeeze())
            y_desireds.append(y[:,:6].detach().cpu().numpy().squeeze())

        elif '7DoF' in robot_choice:
            y_preds.append(y_pred[:,:7].detach().cpu().numpy().squeeze())
            y_desireds.append(y[:,:7].detach().cpu().numpy().squeeze())

        #X_desireds.append(x.detach().cpu().numpy().squeeze())


    y_desireds = np.array(y_desireds)
    y_preds = np.array(y_preds)
    #X_desireds = np.array(X_desireds)
    X_desireds, X_preds, X_errors = reconstruct_pose_modified(y_desireds, y_preds, robot_choice)
    
    X_errors_report = np.array([[X_errors.min(axis=0)],
                                [X_errors.mean(axis=0)],
                                [X_errors.max(axis=0)],
                                [X_errors.std(axis=0)]]).squeeze()
    
    results = {
        "y_preds": y_preds,
        "X_preds": X_preds,
        "y_desireds": y_desireds,
        "X_desireds": X_desireds,
        "X_errors": X_errors,
        "X_errors_report": X_errors_report
    }
    return results



def forward_inference_modified(model, iterator, criterion, device, robot_choice):
    model.eval()
    y_preds = []
    y_desireds = []
    X_desireds = []
    
    for data in iterator:
        x = data['input'].to(device)
        y = data['output'].to(device)

        #x = input_mapping(x,B)
        
        y_pred, _ = model(x)
        y_preds.append(y_pred.detach().cpu().numpy().squeeze())
        y_desireds.append(y.detach().cpu().numpy().squeeze())
        #X_desireds.append(x.detach().cpu().numpy().squeeze())


    y_desireds = np.array(y_desireds)
    y_preds = np.array(y_preds)
    #X_desireds = np.array(X_desireds)
    y_errors = forward_reconstruct_pose_modified(y_desireds, y_preds, robot_choice)
    
    y_errors_report = np.array([[y_errors.min(axis=0)],
                                [y_errors.mean(axis=0)],
                                [y_errors.max(axis=0)],
                                [y_errors.std(axis=0)]]).squeeze()
    
    results = {
        "y_preds": y_preds,
        "y_desireds": y_desireds,
        "y_errors": y_errors,
        "y_errors_report": y_errors_report
    }
    return results


def inference_sobolev(model, iterator, criterion, device, robot_choice):
    model.eval()
    y_preds = []
    y_desireds = []
    X_desireds = []
    
    for data in iterator:
        x = data['input'].to(device)
        yt = data['output'].to(device)
        y = yt[:,:7]

        #x = input_mapping(x,B)
        
        y_pred = model(x)
        y_preds.append(y_pred.detach().cpu().numpy().squeeze())
        y_desireds.append(y.detach().cpu().numpy().squeeze())
        #X_desireds.append(x.detach().cpu().numpy().squeeze())


    y_desireds = np.array(y_desireds)
    y_preds = np.array(y_preds)
    #X_desireds = np.array(X_desireds)
    X_desireds, X_preds, X_errors = reconstruct_pose_modified(y_desireds, y_preds, robot_choice)
    
    X_errors_report = np.array([[X_errors.min(axis=0)],
                                [X_errors.mean(axis=0)],
                                [X_errors.max(axis=0)],
                                [X_errors.std(axis=0)]]).squeeze()
    
    results = {
        "y_preds": y_preds,
        "X_preds": X_preds,
        "y_desireds": y_desireds,
        "X_desireds": X_desireds,
        "X_errors": X_errors,
        "X_errors_report": X_errors_report
    }
    return results


def inference_FK(model, iterator, criterion, device):
    model.eval()
    y_preds = []
    y_desireds = []
    X_desireds = []
    for data in iterator:
        x = data['input'].to(device)
        y = data['output'].to(device)
        y_pred, _ = model(x)
        y_preds.append(y_pred.detach().cpu().numpy().squeeze())
        y_desireds.append(y.detach().cpu().numpy().squeeze())
        X_desireds.append(x.detach().cpu().numpy().squeeze())

    y_desireds = np.array(y_desireds)
    X_desireds = np.array(X_desireds)
    #X_desireds = reconstruct_pose(y_desireds, robot_choice)
    y_preds = np.array(y_preds)
    #X_preds = reconstruct_pose(y_preds, robot_choice)

    #X_errors = np.abs(X_preds - X_desireds)
    y_errors = np.abs(y_preds - y_desireds)

    y_errors_report = np.array([[y_errors.min(axis=0)],
                                [y_errors.mean(axis=0)],
                                [y_errors.max(dim=0)]]).squeeze()
    
    results = {
        "y_preds": y_preds,
        #"X_preds": X_preds,
        "y_desireds": y_desireds,
        #"X_desireds": X_desireds,
        "y_errors": y_errors_report
    }
    return results


def inference_FK_sobolev(model, iterator, criterion, device, robot_choice):
    model.eval()
    y_preds = []
    y_desireds = []
    X_desireds = []
    for data in iterator:
        x = data['input'].to(device)
        yt = data['output'].to(device)
        y = yt[:,:6]
        y_pred = model(x)
        y_preds.append(y_pred.detach().cpu().numpy().squeeze())
        y_desireds.append(y.detach().cpu().numpy().squeeze())
        X_desireds.append(x.detach().cpu().numpy().squeeze())

    y_desireds = np.array(y_desireds)
    X_desireds = np.array(X_desireds)
    y_preds = np.array(y_preds)

    y_errors = forward_reconstruct_pose_modified(y_desireds, y_preds, robot_choice)
    #y_errors = np.abs(y_preds - y_desireds)

    y_errors_report = np.array([[y_errors.min(axis=0)],
                                [y_errors.mean(axis=0)],
                                [y_errors.max(axis=0)],
                                [y_errors.std(axis=0)]]).squeeze()
    
    results = {
        "y_preds": y_preds,
        "y_desireds": y_desireds,
        "y_errors": y_errors,
        "y_errors_report": y_errors_report
    }
    return results




# reconstruct positions in cartesian space from predictions
def reconstruct_pose(y_preds, robot_choice):
    y_preds = torch.from_numpy(y_preds)
    n_samples = y_preds.shape[0]
    pose = []
    for i in range(n_samples):
        t = y_preds[i,:]
        DH = get_DH(robot_choice, t)
        T = forward_kinematics(DH)
        if robot_choice == "4DoF-2RPR":
            # x,y,t1,t2,t3 where x,y (m) and t (rad)
            pose.append(T[:3,-1].numpy())
        
        elif robot_choice == "6DoF-6R-Puma260":
            R = T[:3,:3] 
            rpy = matrix_to_euler_angles(R, "XYZ")
            # x,y,z,R,P,Y,t1,t2,t3,t4,t5,t6 where x,y,z (m) and t (rad)
            #print(T[:3,-1])
            #print(rpy)
            pose.append(torch.cat([T[:3,-1], rpy, t]).numpy())
        
        elif robot_choice == "7DoF-7R-Panda" or robot_choice == "7DoF-GP66":
            R = T[:3,:3] 
            rpy = matrix_to_euler_angles(R, "XYZ")
            # x,y,z,R,P,Y,t1,t2,t3,t4,t5,t6,t7 where x,y,z (m) and t (rad)
            #print(T[:3,-1])
            #print(rpy)
            pose.append(torch.cat([T[:3,-1], rpy, t]).numpy())

    X_pred = np.array(pose)
    return X_pred



def reconstruct_FK_pose(joints_pred, robot_choice, device):
    #y_preds = torch.from_numpy(y_preds)
    n_samples = joints_pred.shape[0]

    DH = get_DH_2(robot_choice).to(device)
    DH_preds = get_DH_batch(robot_choice, joints_pred, DH, device)      
    _, T_preds = forward_kinematics_batch(DH_preds, device)
    position = T_preds[:,:3,3]
    orientation = matrix_to_euler_angles(T_preds[:,:3,:3], "XYZ")
    X_preds = torch.cat([position, orientation], dim=1)

    #print(X_preds.shape)
    #sys.exit()

    return X_preds



def reconstruct_pose_modified(y_desireds, y_preds, robot_choice):
    y_desireds = torch.from_numpy(y_desireds)
    y_preds = torch.from_numpy(y_preds)
    n_samples = y_preds.shape[0]

    pose_desireds = []
    pose_preds = []
    pose_errors = []
    
    for i in range(n_samples):

        # set the joints
        t_desireds = y_desireds[i,:]
        t_preds = y_preds[i,:]

        # compute the forward kinematics
        DH_desireds = get_DH(robot_choice, t_desireds)
        T_desireds = forward_kinematics(DH_desireds)

        DH_preds = get_DH(robot_choice, t_preds)
        T_preds = forward_kinematics(DH_preds)       
        
        #if robot_choice == "7DoF-7R-Panda" or robot_choice == "7DoF-GP66":
        R_desireds = T_desireds[:3,:3]
        R_preds = T_preds[:3,:3] 

        rpy_desireds = matrix_to_euler_angles(R_desireds, "XYZ")
        rpy_preds = matrix_to_euler_angles(R_preds, "XYZ")
        #rpy_desireds = matrix_to_rpy_angles(R_desireds)
        #rpy_preds = matrix_to_rpy_angles(R_preds)

        R_errors = torch.matmul(R_desireds, torch.inverse(R_preds))
        #T_errors = torch.matmul(T_desireds, torch.inverse(T_preds))
        #R_errors = T_errors[:3,:3] 
        rpy_errors = matrix_to_euler_angles(R_errors, "XYZ")
        #rpy_errors = matrix_to_rpy_angles(R_errors)  
        rpy_errors = torch.abs(rpy_errors)         
        #position_errors = T_errors[:3,-1]
        position_errors = torch.abs(T_desireds[:3,-1]-T_preds[:3,-1])
        
        # x,y,z,R,P,Y,t1,t2,t3,t4,t5,t6,t7 where x,y,z (m) and t (rad)
        #print(T[:3,-1])
        #print(rpy)
        pose_desireds.append(torch.cat([T_desireds[:3,-1], rpy_desireds, t_desireds]).numpy())
        pose_preds.append(torch.cat([T_preds[:3,-1], rpy_preds, t_preds]).numpy())
        pose_errors.append(torch.cat([position_errors, rpy_errors]).numpy())



    X_desireds = np.array(pose_desireds)
    X_preds = np.array(pose_preds)
    X_errors = np.array(pose_errors)
    return X_desireds, X_preds, X_errors
    



def forward_reconstruct_pose_modified(y_desireds, y_preds, robot_choice):
    y_desireds = torch.from_numpy(y_desireds)
    y_preds = torch.from_numpy(y_preds)
    n_samples = y_preds.shape[0]

    pose_errors = []
    
    for i in range(n_samples):

        # set the joints
        d_desireds = y_desireds[i,:]
        d_preds = y_preds[i,:]

        # compute the forward kinematics
        R_desireds = euler_angles_to_matrix(d_desireds[3:], "XYZ")
        R_preds = euler_angles_to_matrix(d_preds[3:], "XYZ")

        R_errors = torch.matmul(R_desireds, torch.inverse(R_preds))
        
        rpy_errors = torch.abs(matrix_to_euler_angles(R_errors, "XYZ")) 
        position_errors = torch.abs(d_desireds[:3]-d_preds[:3])

        pose_errors.append(torch.cat([position_errors, rpy_errors]).numpy())

    X_errors = np.array(pose_errors)
    return X_errors



# compute epoch time
def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


# def FFT embedding from this paper: Fourier Features Let Networks Learn High Frequency Functions in Low Dimensional Domains
def input_mapping(x, B):
    if B is None:
        return x
    else:
        x_proj = torch.matmul((2*torch.pi*x), B.T)
        #print(x.shape)
        #print(B.shape)
        #print(x_proj.shape)
        #print(torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1).shape)
    
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)
        

class FourierLayer(nn.Module):
    def __init__(self, in_features, out_features, scale):
        super().__init__()
        B = torch.randn(in_features, out_features)*scale
        self.register_buffer("B", B)

    def forward(self, x):
        x_proj = torch.matmul(2*math.pi*x, self.B)
        out = torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)
        return out

class FourierMLP(nn.Module):
    def __init__(self, input_dim, fourier_dim, h_sizes, output_dim, scale):
        super().__init__()
        
        self.name = "FourierMLP [{}, {}, {}]".format(str(input_dim), str(h_sizes).replace("[","").replace("]",""), str(output_dim))
        self.input_dim = input_dim
        self.fourier_dim = fourier_dim
        self.h_sizes = h_sizes
        self.output_dim = output_dim

        self.fourier_fc = FourierLayer(self.input_dim, self.fourier_dim, scale)
        self.input_fc = nn.Linear(2*self.fourier_dim, self.h_sizes[0])
        
        self.hidden_fc = nn.ModuleList()
        for i in range(len(self.h_sizes)-1):
            self.hidden_fc.append(nn.Linear(self.h_sizes[i], self.h_sizes[i+1]))
        
        self.output_fc = nn.Linear(self.h_sizes[len(self.h_sizes)-1], self.output_dim)

        self.selu_activation = nn.SELU()
        self.relu_activation = nn.ReLU()
        self.prelu_activation = nn.PReLU()
        self.lrelu_activation = nn.LeakyReLU()
        self.sigmoid_activation = nn.Sigmoid()
        self.batch_norm_fc = nn.BatchNorm1d(20000)

    def forward(self, x):

        x = self.fourier_fc(x)
        x = self.input_fc(x)
        x = self.relu_activation(x)  # ReLU(), Sigmoid(), LeakyReLU(negative_slope=0.1)

        for i in range(len(self.h_sizes)-1):
            x = self.hidden_fc[i](x)
            #x = self.batch_norm_fc(x)
            x = self.relu_activation(x)

        x = self.output_fc(x)
        x_temp = x

        return x, x_temp 
        




def get_DH_2(robot_choice):
    # columns: t, d, a, alpha

    if robot_choice == "7DoF-7R-Panda":
        DH = torch.tensor([[0.0,    0.333,      0.0,           0],
                           [0.0,      0.0,      0.0, -torch.pi/2],
                           [0.0,    0.316,      0.0,  torch.pi/2],
                           [0.0,      0.0,   0.0825,  torch.pi/2],
                           [0.0,    0.384,  -0.0825, -torch.pi/2],
                           [0.0,      0.0,      0.0,  torch.pi/2],
                           [0.0,    0.107,    0.088,  torch.pi/2]])
    elif robot_choice == "7DoF-GP66":
        DH = torch.tensor([[t[0],    0.0,      0.0,  torch.pi/2],
                           [t[1],    0.0,     0.25,  torch.pi/2],
                           [ 0.0,   t[2],      0.0,         0.0],
                           [t[3],    0.0,      0.0,  torch.pi/2],
                           [t[4],   0.14,      0.0,  torch.pi/2],
                           [t[5],    0.0,      0.0,  torch.pi/2],
                           [t[6],    0.0,      0.0,  torch.pi/2]])

    return DH



def joint_angle_to_transformation_matrix(theta_ndh, DH, device):
        
        #print("theta.shape: {}".format(theta_ndh.shape))
        #print(theta.shape[0])
        #print(theta.shape[1])

        
        #print("theta: {}".format(theta_ndh))
        #print("theta: {}".format(theta_ndh[:,:,0]))
       
        batch = theta_ndh.shape[0]
        joint_number = theta_ndh.shape[1]

        # populate the DH with the thetas and have as many as the batch size
        #print("DH.shape: {}".format(self.DH.shape))
        DH = DH.to(device)
        #print(DH)
        DH = DH.repeat(batch, 1).view(batch, joint_number, 4)
        #print(DH[0,:,0])
        
        DH[:,:,0] = theta_ndh
        #print(theta_ndh[0,:])
        #print("Inside JFK ||")
        #print(DH[0,:,0])
        #sys.exit()


        #DH[:,2,2] = theta_ndh[:,0,2]
        #DH[:,:2,3] = theta_ndh[:,0,:2]
        #print("DH.shape: {}".format(DH.shape))
        #print("DH.shape: {}".format(DH))
        
        #theta = theta_ndh.clone()
        #print("theta.shape 2", theta.shape)

        #print(DH)
        theta = DH[:,:,0]
        d = DH[:,:,1]
        a = DH[:,:,2]
        alpha = DH[:,:,3]
        
        #print("theta: {}".format(theta))
        #print("d: {}".format(d))
        #print("alpha: {}".format(alpha))
        #print("a: {}".format(a))

        theta = theta.view(-1,1)
        d = d.view(-1, 1)        
        a = a.view(-1, 1)
        alpha = alpha.view(-1, 1)
        
        #print("theta:\n",theta)
        #print("d:\n",d)
        #print("a:\n",a)
        #print("alpha:\n",alpha)
        

        row_1 = torch.cat( (torch.cos(theta), -torch.sin(theta)*torch.cos(alpha),  torch.sin(theta)*torch.sin(alpha), a*torch.cos(theta)), 1 )    
        row_2 = torch.cat( (torch.sin(theta),  torch.cos(theta)*torch.cos(alpha), -torch.cos(theta)*torch.sin(alpha), a*torch.sin(theta)), 1 )   
            
        #print("row_1: ", row_1.shape)

        zeros = torch.autograd.Variable(torch.zeros(joint_number,1).to(device))
        zeros = zeros.repeat(batch,1).view(-1, 1)         
        ones = torch.autograd.Variable(torch.ones(joint_number,1).to(device))
        ones = ones.repeat(batch,1).view(-1, 1)

        #print(joint_number)
        #print(zeros.shape)
        #print(alpha.shape)
        #print(d.shape)
        
        row_3 = torch.cat( (zeros, torch.sin(alpha), torch.cos(alpha), d), 1 )
        row_4 = torch.cat( (zeros, zeros, zeros, ones), 1 )
        T_successive = torch.cat((row_1, row_2, row_3, row_4), 1).view(batch, joint_number, 4, 4)  

        #print(T_successive.shape)
        #print(T_successive[0,:,:])
        #print()

        T_total = T_successive[:,0,:,:].view(batch,1,4,4)
        #print("T_successive.shape): {}".format(T_successive.shape))
        #print("T_total.shape): {}".format(T_total.shape))  
            

        for i in range(1, joint_number):
            temp_total_transformation = torch.matmul(T_total, T_successive[:,i,:,:].view(batch,1,4,4))
            T_total = temp_total_transformation    

        return T_successive, T_total.view(batch,4,4)






def get_DH_batch(robot_choice, t, DH, device):
    batch_size = t.shape[0]  # Get the batch size
    zeros = torch.zeros(batch_size, device=device)
    ones = torch.ones(batch_size, device=device)
    #print("Inside")
    #print(DH)
    #print(t[0,:])
    
    if robot_choice == "7DoF-7R-Panda":
        """
        DH = torch.stack([
            torch.stack([t[:, 0],   torch.ones(batch_size)*0.333,        torch.zeros(batch_size),                 torch.zeros(batch_size)], dim=1),
            torch.stack([t[:, 1],        torch.zeros(batch_size),        torch.zeros(batch_size),  -torch.ones(batch_size) * torch.pi / 2], dim=1),
            torch.stack([t[:, 2],   torch.ones(batch_size)*0.316,        torch.zeros(batch_size),    torch.ones(batch_size)* torch.pi / 2], dim=1),
            torch.stack([t[:, 3],        torch.zeros(batch_size),  torch.ones(batch_size)*0.0825,   torch.ones(batch_size) * torch.pi / 2], dim=1),
            torch.stack([t[:, 4], torch.ones(batch_size) * 0.384, -torch.ones(batch_size)*0.0825,  -torch.ones(batch_size) * torch.pi / 2], dim=1),
            torch.stack([t[:, 5],        torch.zeros(batch_size),        torch.zeros(batch_size),   torch.ones(batch_size) * torch.pi / 2], dim=1),
            torch.stack([t[:, 6],   torch.ones(batch_size)*0.107,   torch.ones(batch_size)*0.088,   torch.ones(batch_size) * torch.pi / 2], dim=1)
        ], dim=1) 
        """
        DH = torch.stack([
            torch.stack([t[:, 0],    ones*DH[0,1],    zeros*DH[0,2],    zeros*DH[0,3]], dim=1),
            torch.stack([t[:, 1],   zeros*DH[1,1],    zeros*DH[1,2],     ones*DH[1,3]], dim=1),
            torch.stack([t[:, 2],    ones*DH[2,1],    zeros*DH[2,2],     ones*DH[2,3]], dim=1),
            torch.stack([t[:, 3],   zeros*DH[3,1],     ones*DH[3,2],     ones*DH[3,3]], dim=1),
            torch.stack([t[:, 4],    ones*DH[4,1],     ones*DH[4,2],     ones*DH[4,3]], dim=1),
            torch.stack([t[:, 5],   zeros*DH[5,1],    zeros*DH[5,2],     ones*DH[5,3]], dim=1),
            torch.stack([t[:, 6],    ones*DH[6,1],     ones*DH[6,2],     ones*DH[6,3]], dim=1)
        ], dim=1)

    elif robot_choice == "7DoF-GP66":
        DH = torch.stack([
            torch.stack([t[:, 0], torch.zeros(batch_size), torch.zeros(batch_size), torch.ones(batch_size) * torch.pi / 2], dim=1),
            torch.stack([t[:, 1], torch.zeros(batch_size), torch.ones(batch_size) * 0.25, torch.ones(batch_size) * torch.pi / 2], dim=1),
            torch.stack([torch.zeros(batch_size), t[:, 2], torch.zeros(batch_size), torch.zeros(batch_size)], dim=1),
            torch.stack([t[:, 3], torch.zeros(batch_size), torch.zeros(batch_size), torch.ones(batch_size) * torch.pi / 2], dim=1),
            torch.stack([t[:, 4], torch.ones(batch_size) * 0.14, torch.zeros(batch_size), torch.ones(batch_size) * torch.pi / 2], dim=1),
            torch.stack([t[:, 5], torch.zeros(batch_size), torch.zeros(batch_size), torch.ones(batch_size) * torch.pi / 2], dim=1),
            torch.stack([t[:, 6], torch.zeros(batch_size), torch.zeros(batch_size), torch.ones(batch_size) * torch.pi / 2], dim=1)
        ], dim=1)
    else:
        raise ValueError("robot DH not yet implemented, current possible choices are: 7DoF-7R-Panda, 7DoF-GP66")
    
    return DH


# A matrix using DH parameters with batch support
def A_matrix_batch(t, d, a, al):
    # the inputs of torch.sin and torch.cos are expressed in rad
    cos_t = torch.cos(t)
    sin_t = torch.sin(t)
    cos_al = torch.cos(al)
    sin_al = torch.sin(al)

    zeros = torch.zeros_like(t)
    ones = torch.ones_like(t)

    A = torch.stack([
        torch.stack([cos_t, -sin_t * cos_al,  sin_t * sin_al,     a * cos_t], dim=1),
        torch.stack([sin_t,  cos_t * cos_al, -cos_t * sin_al,     a * sin_t], dim=1),
        torch.stack([zeros,          sin_al,          cos_al,             d], dim=1),
        torch.stack([zeros,           zeros,           zeros,          ones], dim=1)
    ], dim=1)

    return A
                         
# Forward Kinematics with DH parameters and batch support
def forward_kinematics_batch(DH, device):
    n_DoF = DH.shape[1]
    batch_size = DH.shape[0]
    #print(DH[0,:,0])

    T = torch.eye(4, device=device).unsqueeze(0).repeat(batch_size, 1, 1)  # Initialize as identity matrix for batch operations

    for i in range(n_DoF):
        A = A_matrix_batch(*DH[:, i].unbind(dim=1))  # Extract DH parameters for each joint in the batch
        T = torch.matmul(T, A)  # Batch matrix multiplication
        #print(A[0,:,:])

    #print("Inside FK batch ||")
    T_successive = T

    return T_successive, T #.squeeze()  # Remove the batch dimension if not needed



def matrix_to_axis_angle(matrix: torch.Tensor) -> torch.Tensor:
    """
    Convert rotations given as rotation matrices to axis/angle.

    Args:
        matrix: Rotation matrices as tensor of shape (..., 3, 3).

    Returns:
        Rotations given as a vector in axis angle form, as a tensor
            of shape (..., 3), where the magnitude is the angle
            turned anticlockwise in radians around the vector's
            direction.
    """
    return quaternion_to_axis_angle(matrix_to_quaternion(matrix))


def standardize_quaternion(quaternions: torch.Tensor) -> torch.Tensor:
    """
    Convert a unit quaternion to a standard form: one in which the real
    part is non negative.

    Args:
        quaternions: Quaternions with real part first,
            as tensor of shape (..., 4).

    Returns:
        Standardized quaternions as tensor of shape (..., 4).
    """
    return torch.where(quaternions[..., 0:1] < 0, -quaternions, quaternions)


def _sqrt_positive_part(x: torch.Tensor) -> torch.Tensor:
    """
    Returns torch.sqrt(torch.max(0, x))
    but with a zero subgradient where x is 0.
    """
    ret = torch.zeros_like(x)
    positive_mask = x > 0
    ret[positive_mask] = torch.sqrt(x[positive_mask])
    return ret


def matrix_to_quaternion(matrix: torch.Tensor) -> torch.Tensor:
    """
    Convert rotations given as rotation matrices to quaternions.

    Args:
        matrix: Rotation matrices as tensor of shape (..., 3, 3).

    Returns:
        quaternions with real part first, as tensor of shape (..., 4).
    """
    if matrix.size(-1) != 3 or matrix.size(-2) != 3:
        raise ValueError(f"Invalid rotation matrix shape {matrix.shape}.")

    batch_dim = matrix.shape[:-2]
    m00, m01, m02, m10, m11, m12, m20, m21, m22 = torch.unbind(
        matrix.reshape(batch_dim + (9,)), dim=-1
    )

    q_abs = _sqrt_positive_part(
        torch.stack(
            [
                1.0 + m00 + m11 + m22,
                1.0 + m00 - m11 - m22,
                1.0 - m00 + m11 - m22,
                1.0 - m00 - m11 + m22,
            ],
            dim=-1,
        )
    )

    # we produce the desired quaternion multiplied by each of r, i, j, k
    quat_by_rijk = torch.stack(
        [
            # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
            #  `int`.
            torch.stack([q_abs[..., 0] ** 2, m21 - m12, m02 - m20, m10 - m01], dim=-1),
            # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
            #  `int`.
            torch.stack([m21 - m12, q_abs[..., 1] ** 2, m10 + m01, m02 + m20], dim=-1),
            # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
            #  `int`.
            torch.stack([m02 - m20, m10 + m01, q_abs[..., 2] ** 2, m12 + m21], dim=-1),
            # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
            #  `int`.
            torch.stack([m10 - m01, m20 + m02, m21 + m12, q_abs[..., 3] ** 2], dim=-1),
        ],
        dim=-2,
    )

    # We floor here at 0.1 but the exact level is not important; if q_abs is small,
    # the candidate won't be picked.
    flr = torch.tensor(0.1).to(dtype=q_abs.dtype, device=q_abs.device)
    quat_candidates = quat_by_rijk / (2.0 * q_abs[..., None].max(flr))

    # if not for numerical problems, quat_candidates[i] should be same (up to a sign),
    # forall i; we pick the best-conditioned one (with the largest denominator)
    out = quat_candidates[
        F.one_hot(q_abs.argmax(dim=-1), num_classes=4) > 0.5, :
    ].reshape(batch_dim + (4,))
    return standardize_quaternion(out)


def quaternion_to_axis_angle(quaternions: torch.Tensor) -> torch.Tensor:
    """
    Convert rotations given as quaternions to axis/angle.

    Args:
        quaternions: quaternions with real part first,
            as tensor of shape (..., 4).

    Returns:
        Rotations given as a vector in axis angle form, as a tensor
            of shape (..., 3), where the magnitude is the angle
            turned anticlockwise in radians around the vector's
            direction.
    """
    norms = torch.norm(quaternions[..., 1:], p=2, dim=-1, keepdim=True)
    half_angles = torch.atan2(norms, quaternions[..., :1])
    angles = 2 * half_angles
    eps = 1e-6
    small_angles = angles.abs() < eps
    sin_half_angles_over_angles = torch.empty_like(angles)
    sin_half_angles_over_angles[~small_angles] = (
        torch.sin(half_angles[~small_angles]) / angles[~small_angles]
    )
    # for x small, sin(x/2) is about x/2 - (x/2)^3/6
    # so sin(x/2)/x is about 1/2 - (x*x)/48
    sin_half_angles_over_angles[small_angles] = (
        0.5 - (angles[small_angles] * angles[small_angles]) / 48
    )
    return quaternions[..., 1:] / sin_half_angles_over_angles




# compute loss function by employing the FK 
class IKLoss(nn.Module):
    def __init__(self, robot_choice, device):
        #super(FKLoss, self).__init__()
        super().__init__()
        self.criterion = nn.MSELoss(reduction="mean")
        #self.criterion = nn.L1Loss(reduction="mean")
        self.robot_choice = robot_choice
        self.device = device

    def forward(self, joints_pred, joints_des):
        
        # get the DH
        DH = get_DH_2(self.robot_choice).to(self.device)

        # get the FK from the predicted joints
        DH_preds = get_DH_batch(self.robot_choice, joints_pred, DH, self.device)      
        _, T_preds = forward_kinematics_batch(DH_preds, self.device)

        # get the FK from the desired joints
        DH_des = get_DH_batch(self.robot_choice, joints_des, DH, self.device)
        _, T_des = forward_kinematics_batch(DH_des, self.device)

        # get the inverse of the desired HTMs
        T_des_inv = T_des.clone()
        T_des_inv[:,:3,:3] = torch.transpose(T_des_inv[:,:3,:3], 1, 2)
        T_des_inv[:,:3,3] = torch.matmul(-T_des_inv[:,:3,:3], T_des_inv[:,:3,3].unsqueeze(-1)).squeeze(-1)

        # get the errors
        T_errors = torch.matmul(T_preds, T_des_inv)

        R_errors = T_errors[:,:3,:3]
        axis_angle_errors = matrix_to_axis_angle(R_errors)
        axis_angle_errors = torch.norm(axis_angle_errors, dim=1)
        
        #loss = self.criterion(T_total_pred[:,:3,-1], T_total_des[:,:3,-1]) + torch.mean((rpy_errors)**2)
        loss = self.criterion(T_preds[:,:3,-1], T_des[:,:3,-1]) + torch.mean((axis_angle_errors)**2)
        

        return loss



####################################################
# Custom Loss Functions 
####################################################

class MDNLoss(nn.Module):
    def __init__(self, eps=1e-8):
        super().__init__()
        self.eps = eps

    def forward(self, mdn_params, y):
        logits_pi, mu, log_sigma = mdn_params  # [B,K], [B,K,D], [B,K,D]
        y = y.unsqueeze(1)  # [B,1,D] to broadcast against K

        # log N(y | mu, sigma)
        # sigma = exp(log_sigma)
        log_sigma = torch.clamp(log_sigma, min=-10.0, max=10.0)
        sigma2 = torch.exp(2.0 * log_sigma)

        # per-dimension Gaussian log-prob, then sum over D
        log_prob = -0.5 * (((y - mu) ** 2) / (sigma2 + self.eps) + 2.0 * log_sigma + math.log(2.0 * math.pi))
        log_prob = log_prob.sum(dim=2)  # [B,K]

        log_pi = F.log_softmax(logits_pi, dim=1)  # [B,K]

        # log sum_k pi_k * N_k  using logsumexp for stability
        log_mix = torch.logsumexp(log_pi + log_prob, dim=1)  # [B]
        nll = -log_mix.mean()
        return nll



class MDNLossEntropy(nn.Module):
    """
    L = L_MDN - lambda_ent * H(pi)

    - L_MDN is the usual MDN negative log-likelihood.
    - H(pi) is the entropy of mixture weights, encouraging non-collapsed usage.
    """
    def __init__(self, eps=1e-8, lambda_ent=1e-3, clamp_log_sigma=(-10.0, 10.0)):
        super().__init__()
        self.eps = eps
        self.lambda_ent = float(lambda_ent)
        self.clamp_log_sigma = clamp_log_sigma

    def forward(self, mdn_params, y):
        logits_pi, mu, log_sigma = mdn_params  # [B,K], [B,K,D], [B,K,D]
        y = y.unsqueeze(1)                     # [B,1,D] broadcast against K

        # ---- Gaussian log-likelihood per component ----
        log_sigma = torch.clamp(log_sigma, min=self.clamp_log_sigma[0], max=self.clamp_log_sigma[1])
        sigma2 = torch.exp(2.0 * log_sigma)

        log_prob = -0.5 * (
            ((y - mu) ** 2) / (sigma2 + self.eps)
            + 2.0 * log_sigma
            + math.log(2.0 * math.pi)
        )
        log_prob = log_prob.sum(dim=2)         # [B,K]

        log_pi = F.log_softmax(logits_pi, dim=1)  # [B,K]

        # ---- Mixture log-likelihood via logsumexp ----
        log_mix = torch.logsumexp(log_pi + log_prob, dim=1)  # [B]
        nll = -log_mix.mean()

        # ---- Entropy regularization on pi ----
        # pi = softmax(logits_pi), H(pi) = -sum pi log(pi)
        pi = torch.softmax(logits_pi, dim=1)                 # [B,K]
        entropy = -(pi * (log_pi)).sum(dim=1).mean()         # scalar

        # Minimize: nll - lambda * entropy  (so entropy is maximized)
        loss = nll - self.lambda_ent * entropy

        # Optional: return diagnostics for logging
        return loss




class FKLoss(nn.Module):
    def __init__(self, robot_choice, device):
        #super(FKLoss, self).__init__()
        super().__init__()
        self.criterion = nn.MSELoss(reduction="mean")
        #self.criterion = nn.L1Loss(reduction="mean")
        self.robot_choice = robot_choice
        self.device = device

    def forward(self, pose_pred, pose_des):

        #print(pose_pred.shape)

        R_preds = euler_angles_to_matrix(pose_pred[:,3:], "XYZ")
        R_desireds = euler_angles_to_matrix(pose_des[:,3:], "XYZ")
        R_errors = torch.matmul(R_desireds, torch.inverse(R_preds))

        axis_angle_errors = matrix_to_axis_angle(R_errors)
        axis_angle_errors = torch.norm(axis_angle_errors, dim=1)
        
        #loss = self.criterion(T_total_pred[:,:3,-1], T_total_des[:,:3,-1]) + torch.mean((rpy_errors)**2)
        loss = self.criterion(pose_pred[:,:3], pose_des[:,:3]) + torch.mean((axis_angle_errors)**2)
        

        return loss




# compute loss function by employing the FK 
class FKLossOld(nn.Module):
    def __init__(self, robot_choice, device):
        #super(FKLoss, self).__init__()
        super().__init__()
        self.criterion = nn.MSELoss(reduction="mean")
        #self.criterion = nn.L1Loss(reduction="mean")
        self.robot_choice = robot_choice
        self.device = device

    def forward(self, joints_pred, joints_des):
        
        DH = get_DH_2(self.robot_choice)
        #print(joints_pred[0,:])
        #print("Inside FK Loss")
        T_successive_pred, T_total_pred = joint_angle_to_transformation_matrix(joints_pred, DH, self.device)
        R_pred = T_total_pred[:,:3,:3]
        #rpy_pred = matrix_to_euler_angles(R_pred, "XYZ")
        #pose_pred = torch.cat([T_total_pred[:,:3,-1], rpy_pred[:,:]], axis=1)

        """
        T_successive_des, T_total_des = joint_angle_to_transformation_matrix(joints_des, DH, self.device)
        R_des = T_total_des[:,:3,:3]
        #rpy_des = matrix_to_euler_angles(R_des, "XYZ")
        #pose_des = torch.cat([T_total_des[:,:3,-1], rpy_des[:,:]], axis=1)

        R_errors = torch.matmul(R_pred, torch.inverse(R_des))
        rpy_errors = matrix_to_euler_angles(R_errors, "XYZ")
        """

        ################################################
        # get the DH
        DH = get_DH_2(self.robot_choice).to(self.device)

        # get the FK from the predicted joints
        DH_preds = get_DH_batch(self.robot_choice, joints_pred, DH, self.device)     
        #print("DH_batch: ", DH_preds[0,:,0])  
        _, T_preds = forward_kinematics_batch(DH_preds, self.device)

        #print("End ||")
        #print(T_preds[0,:,:])
        #print(T_total_pred[0,:,:])
        #sys.exit()

        # get the FK from the desired joints
        DH_des = get_DH_batch(self.robot_choice, joints_des, DH, self.device)
        _, T_des = forward_kinematics_batch(DH_des, self.device)

        # get the inverse of the desired HTMs
        T_des_inv = T_des.clone()
        T_des_inv[:,:3,:3] = torch.transpose(T_des_inv[:,:3,:3], 1, 2)
        T_des_inv[:,:3,3] = torch.matmul(-T_des_inv[:,:3,:3], T_des_inv[:,:3,3].unsqueeze(-1)).squeeze(-1)

        # get the errors
        T_errors = torch.matmul(T_preds, T_des_inv)


        

        """
        print("T_errors:\n", T_errors.isnan().nonzero())

        # get the pose loss 
        xyz_errors = T_errors[:,:3,3]
        rpy_errors = matrix_to_euler_angles(T_errors[:,:3,:3].round(decimals=3), "ZYX")
        #rpy_errors = matrix_to_rpy_angles(T_errors[:,:3,:3])

        #print(xyz_errors)
        #print(rpy_errors)
        xyz_errors_index = xyz_errors.isnan().nonzero()
        rpy_errors_index = rpy_errors.isnan().nonzero()

        print("xyz_errors:\n", xyz_errors_index)
        print("rpy_errors:\n", rpy_errors_index)

        print()
        idx = rpy_errors_index[0,0]
        print(T_errors[idx,:3,:3])
        print(matrix_to_euler_angles(T_errors[idx,:3,:3], "ZYX"))


        axis_angle_errors = matrix_to_axis_angle(T_errors[idx,:3,:3])
        print(axis_angle_errors.shape)
        print(axis_angle_errors[0,])
        axis_angle_errors_index = axis_angle_errors.isnan().nonzero()
        print("axis_angle_errors:\n", axis_angle_errors_index)


        loss = torch.mean(torch.sum((xyz_errors**2), dim=1)) + torch.mean(torch.sum((rpy_errors**2), dim=1))
        print(loss)  
        sys.exit()    

        #T_errors = torch.matmul(T_desireds, torch.inverse(T_preds))
        #R_errors = T_errors[:3,:3] 
        #print(R_errors.shape)
        #print(rpy_errors.shape)


        #print(torch.mean(torch.square(rpy_errors)))
        #print(torch.mean((rpy_errors)**2))
        #print(self.criterion(T_total_pred[:,:3,-1], T_total_des[:,:3,-1]))

        #print("T_total:")
        #print(T_total)
        #print("Joints_fk:")
        #print(pose_pred)
        #print("poses")
        #print(pose_des)
        #print()
        
        #loss = self.criterion(T_total_pred[:,:3,-1], T_total_des[:,:3,-1]) + torch.mean((rpy_errors)**2)
        """
        R_errors = T_errors[:,:3,:3]
        axis_angle_errors = matrix_to_axis_angle(R_errors)
        axis_angle_errors = torch.norm(axis_angle_errors, dim=1)
        
        #loss = self.criterion(T_total_pred[:,:3,-1], T_total_des[:,:3,-1]) + torch.mean((rpy_errors)**2)
        loss = self.criterion(T_preds[:,:3,-1], T_des[:,:3,-1]) + torch.mean((axis_angle_errors)**2)
        

        return loss



def matrix_to_rpy_angles(rotation_matrices):
    # Ensure the input is a 3x3 matrix
    if rotation_matrices.shape[-2:] != torch.Size([3, 3]):
        raise ValueError("Input should be a 3x3 rotation matrix")

    # Extract individual elements for clarity
    nx, ox, ax = rotation_matrices[..., 0, 0], rotation_matrices[..., 0, 1], rotation_matrices[..., 0, 2]
    ny, oy, ay = rotation_matrices[..., 1, 0], rotation_matrices[..., 1, 1], rotation_matrices[..., 1, 2]
    nz, oz, az = rotation_matrices[..., 2, 0], rotation_matrices[..., 2, 1], rotation_matrices[..., 2, 2]
   

    # Calculate roll (x-axis rotation)
    roll = torch.atan2(ny, nx) 
    pitch = torch.atan2(-nz, (nx*torch.cos(roll) + ny*torch.sin(roll)))  
    yaw = torch.atan2((-ay*torch.cos(roll) + ax*torch.sin(roll)),(oy*torch.cos(roll) - ox*torch.sin(roll)))
    
    return torch.stack((roll, pitch, yaw),dim=-1)



# compute loss function by employing the FK 
class FKLossD(nn.Module):
    def __init__(self, robot_choice, device):
        #super(FKLoss, self).__init__()
        super().__init__()
        self.criterion = nn.MSELoss(reduction="mean")
        #self.criterion = nn.L1Loss(reduction="mean")
        self.robot_choice = robot_choice
        self.device = device

    def forward(self, joints_pred, joints_des):
        #inputs_fk = torch.zeros_like(targets)
        #joints_fk = torch.clone(poses)
        #joints_fk.retain_grad()

        
        DH = get_DH_2(self.robot_choice)
        T_successive_pred, T_total_pred = joint_angle_to_transformation_matrix(joints_pred, DH, self.device)
        R_pred = T_total_pred[:,:3,:3]
        #rpy_pred = matrix_to_euler_angles(R_pred, "XYZ")
        #pose_pred = torch.cat([T_total_pred[:,:3,-1], rpy_pred[:,:]], axis=1)


        T_successive_des, T_total_des = joint_angle_to_transformation_matrix(joints_des, DH, self.device)
        R_des = T_total_des[:,:3,:3]
        #rpy_des = matrix_to_euler_angles(R_des, "XYZ")
        #pose_des = torch.cat([T_total_des[:,:3,-1], rpy_des[:,:]], axis=1)

        R_errors = torch.matmul(R_pred, torch.inverse(R_des))
        #rpy_errors = matrix_to_euler_angles(R_errors, "XYZ")
        axis_angle_errors = matrix_to_axis_angle(R_errors)
        #print(axis_angle_errors.shape)
        #print(axis_angle_errors[0,])

        axis_angle_errors = torch.norm(axis_angle_errors, dim=1)
        #print(axis_angle_errors)
        #print(axis_angle_errors.shape)
        #print()
        #axis_angle_errors_index = axis_angle_errors.isnan().nonzero()
        #print("axis_angle_errors:\n", axis_angle_errors_index)
        #sys.exit()

        #T_errors = torch.matmul(T_desireds, torch.inverse(T_preds))
        #R_errors = T_errors[:3,:3] 
        #print(R_errors.shape)
        #print(rpy_errors.shape)


        #print(torch.mean(torch.square(rpy_errors)))
        #print(torch.mean((rpy_errors)**2))
        #print(self.criterion(T_total_pred[:,:3,-1], T_total_des[:,:3,-1]))

        #print("T_total:")
        #print(T_total)
        #print("Joints_fk:")
        #print(pose_pred)
        #print("poses")
        #print(pose_des)
        #print()
        
        #loss = self.criterion(T_total_pred[:,:3,-1], T_total_des[:,:3,-1]) + torch.mean((rpy_errors)**2)
        loss = self.criterion(T_total_pred[:,:3,-1], T_total_des[:,:3,-1]) + torch.mean((axis_angle_errors)**2)
        

        return loss
        

class FKLossC(nn.Module):
    def __init__(self, robot_choice, device):
        #super(FKLoss, self).__init__()
        super().__init__()
        self.criterion = nn.MSELoss(reduction="mean")
        #self.criterion = nn.L1Loss(reduction="mean")
        self.robot_choice = robot_choice
        self.device = device

    def forward(self, joints_pred, joints_des):
        #inputs_fk = torch.zeros_like(targets)
        #joints_fk = torch.clone(poses)
        #joints_fk.retain_grad()

        
        DH = get_DH_2(self.robot_choice)
        T_successive_pred, T_total_pred = joint_angle_to_transformation_matrix(joints_pred, DH, self.device)
        R_pred = T_total_pred[:,:3,:3]
        rpy_pred = matrix_to_euler_angles(R_pred, "XYZ")
        pose_pred = torch.cat([T_total_pred[:,:3,-1], rpy_pred[:,:]], axis=1)


        T_successive_des, T_total_des = joint_angle_to_transformation_matrix(joints_des, DH, self.device)
        R_des = T_total_des[:,:3,:3]
        rpy_des = matrix_to_euler_angles(R_des, "XYZ")
        pose_des = torch.cat([T_total_des[:,:3,-1], rpy_des[:,:]], axis=1)

        #print()
        #print("T_total:")
        #print(T_total)
        #print("Joints_fk:")
        #print(pose_pred)
        #print("poses")
        #print(pose_des)
        #print()
        
        loss = self.criterion(pose_pred, pose_des)
        
        return loss



class FKLossB(nn.Module):
    def __init__(self, robot_choice):
        #super(FKLoss, self).__init__()
        super().__init__()
        self.criterion = nn.MSELoss(reduction="mean")
        #self.criterion = nn.L1Loss(reduction="mean")
        self.robot_choice = robot_choice

    def forward(self, joints, poses):
        #inputs_fk = torch.zeros_like(targets)
        joints_fk = torch.clone(poses)
        joints_fk.retain_grad()


        #DH = torch.vmap(get_DH, in)(self.robot_choice, joints)
        #print(DH.shape)

        #sys.exit()



        #print(targets)
        #print(inputs_fk)
        #sys.exit()
        for i in range(joints.shape[0]):
            #print()
            t = joints[i,:]
            DH = get_DH(self.robot_choice, t)
            #print(DH)
            T = forward_kinematics(DH)
            #print(T.type)
            if self.robot_choice == "7DoF-7R-Panda":
                R = T[:3,:3]
                rpy = matrix_to_euler_angles(R, "XYZ")
                
                #inputs_fk[i,:] = T[:3,-1]   
                joints_fk[i,:] = torch.cat([T[:3,-1], rpy])

        #inputs_fk = inputs_fk
        #print(joints_fk)
        #print(poses)
        #print("here")
        #sys.exit()
        loss = self.criterion(joints_fk, poses)
        #print(loss)
        return loss




#################################################
# Custom Loss for IROS 2025
#################################################
# Position Loss Component
class PositionLoss(nn.Module):
    def __init__(self, beta=1.0):
        super(PositionLoss, self).__init__()
        self.smooth_l1 = nn.SmoothL1Loss(beta=beta)

    def forward(self, p_pred, p_true):
        return self.smooth_l1(p_pred, p_true)
    

# Rotation Loss Component
class RotationLoss(nn.Module):
    def __init__(self, beta=1.0):
        super(RotationLoss, self).__init__()
        self.smooth_l1 = nn.SmoothL1Loss(beta=beta)

    def forward(self, q_pred, q_true):
        loss_q = self.smooth_l1(q_pred, q_true)
        loss_neg_q = self.smooth_l1(q_pred, -q_true)
        return torch.min(loss_q, loss_neg_q)  # Take the minimum loss
    

# Joint Limit Loss
class JointLimitLoss(nn.Module):
    def __init__(self, min_limits, max_limits):
        super(JointLimitLoss, self).__init__()
        self.min_limits = torch.tensor(min_limits, dtype=torch.float32)
        self.max_limits = torch.tensor(max_limits, dtype=torch.float32)

    def forward(self, joint_angles):
        lower_violation = torch.relu(self.min_limits - joint_angles)  # If below min, add penalty
        upper_violation = torch.relu(joint_angles - self.max_limits)  # If above max, add penalty
        return torch.mean(lower_violation + upper_violation)  # Penalize joint limit violations


# Smoothness Loss
class SmoothnessLoss(nn.Module):
    def __init__(self):
        super(SmoothnessLoss, self).__init__()

    def forward(self, joint_angles, prev_joint_angles):
        return torch.mean(torch.abs(joint_angles - prev_joint_angles))  # Penalize sudden changes

# Singularity Loss
class SingularityLoss(nn.Module):
    def __init__(self):
        super(SingularityLoss, self).__init__()

    def forward(self, jacobian):
        det = torch.det(jacobian)
        return torch.mean(1.0 / (det + 1e-6))  # Penalize near-zero determinant


# Combined Inverse Kinematics Loss with Learnable weights
class InverseKinematicsLoss(nn.Module):
    def __init__(self, dh_params, min_limits, max_limits):
        super(InverseKinematicsLoss, self).__init__()

        # Learnable weights
        self.w_pos = nn.Parameter(torch.tensor(1.0).log())
        self.w_rot = nn.Parameter(torch.tensor(1.0).log())
        self.w_joint = nn.Parameter(torch.tensor(1.0).log())
        self.w_smooth = nn.Parameter(torch.tensor(1.0).log())
        self.w_sing = nn.Parameter(torch.tensor(1.0).log())

        # Loss functions
        self.position_loss = PositionLoss()
        self.rotation_loss = RotationLoss()
        self.joint_limit_loss = JointLimitLoss(min_limits, max_limits)
        self.smoothness_loss = SmoothnessLoss()
        self.singularity_loss = SingularityLoss()

    def forward(self, joint_angles_pred, prev_joint_angles, p_true, q_true, jacobian):
        """
        Computes the full IK loss function.

        Args:
            joint_angles_pred (torch.Tensor): Predicted joint angles (N, num_joints)
            prev_joint_angles (torch.Tensor): Previous joint angles (N, num_joints)
            p_true (torch.Tensor): Target positions (N, 3)
            q_true (torch.Tensor): Target quaternions (N, 4)
            jacobian (torch.Tensor): Jacobian matrix (N, 6, num_joints)

        Returns:
            torch.Tensor: Combined loss.
        """
        # Compute FK to get predicted pose
        p_pred, q_pred = forward_kinematics(joint_angles_pred, dh_params)

        # Compute losses
        l_pos = self.position_loss(p_pred, p_true)
        l_rot = self.rotation_loss(q_pred, q_true)
        l_joint = self.joint_limit_loss(joint_angles_pred)
        l_smooth = self.smoothness_loss(joint_angles_pred, prev_joint_angles)
        l_sing = self.singularity_loss(jacobian)

        # Convert learnable weights
        w_pos = torch.exp(self.w_pos)
        w_rot = torch.exp(self.w_rot)
        w_joint = torch.exp(self.w_joint)
        w_smooth = torch.exp(self.w_smooth)
        w_sing = torch.exp(self.w_sing)

        # Compute total loss
        total_loss = (w_pos * l_pos 
                      + w_rot * l_rot 
                      + w_joint * l_joint 
                      + w_smooth * l_smooth 
                      + w_sing * l_sing)

        return total_loss
