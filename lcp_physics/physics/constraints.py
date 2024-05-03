#
# Copyright 2024 Max-Planck-Gesellschaft
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# ========================================================================
# This file originates from
# https://github.com/locuslab/lcp-physics/tree/a85ecfe0fdc427ee016f3d1c2ddb0d0c0f98f21b/lcp_physics, licensed under the
# Apache License, version 2.0 (see LICENSE).
# This file was updated to transfer pytorch tensors to CPU before converting them to numpy arrays. The update was done
# by Michael Strecke, michael.strecke@tuebingen.mpg.de, Embodied Vision Group, Max Planck Institute for Intelligent Systems.
#
import pygame

import torch

from .utils import Indices, Defaults, cart_to_polar, polar_to_cart


X = Indices.X
Y = Indices.Y
DIM = Defaults.DIM


class Joint:
    """Revolute joint.
    """
    def __init__(self, body1, body2, pos):
        self.static = False
        self.num_constraints = 2
        self.body1 = body1
        self.body2 = body2
        self.pos = body1.p.new_tensor(pos)
        self.pos1 = self.pos - self.body1.pos
        self.r1, self.rot1 = cart_to_polar(self.pos1)
        self.rot2 = None
        if body2 is not None:
            self.pos2 = self.pos - self.body2.pos
            self.r2, self.rot2 = cart_to_polar(self.pos2)

    def J(self):
        J1 = torch.cat([torch.cat([-self.pos1[Y:Y+1], self.pos1[X:X+1]]).unsqueeze(1),
                        torch.eye(DIM).type_as(self.pos)], dim=1)
        J2 = None
        if self.body2 is not None:
            J2 = torch.cat([torch.cat([self.pos2[Y:Y+1], -self.pos2[X:X+1]]).unsqueeze(1),
                            -torch.eye(DIM).type_as(self.pos)], dim=1)
        return J1, J2

    def move(self, dt):
        self.rot1 = self.rot1 + self.body1.v[0] * dt
        if self.body2 is not None:
            self.rot2 = self.rot2 + self.body2.v[0] * dt
        self.update_pos()

    def update_pos(self):
        self.pos1 = polar_to_cart(self.r1, self.rot1)
        self.pos = self.body1.pos + self.pos1
        if self.body2 is not None:
            # keep position on body1 as reference
            self.pos2 = self.pos - self.body2.pos

    def draw(self, screen, pixels_per_meter=1):
        pos = (self.pos.detach().cpu().numpy() * pixels_per_meter).astype(int)
        return [pygame.draw.circle(screen, (0, 255, 0), pos, 2)]


class FixedJoint:
    """Fixed joint, fixes two bodies together."""
    def __init__(self, body1, body2):
        self.static = False
        self.num_constraints = 3
        self.body1 = body1
        self.body2 = body2
        self.pos = body1.pos
        self.pos1 = self.pos - self.body1.pos
        self.rot1 = self.pos.new_tensor(0)
        self.rot2 = None
        self.pos2 = self.pos - self.body2.pos
        self.rot2 = self.body2.p[0] - self.body1.p[0]  # inverted sign?

    def J(self):
        J1 = torch.cat([torch.cat([-self.pos1[Y:Y+1], self.pos1[X:X+1]]).unsqueeze(1),
                        torch.eye(DIM).type_as(self.pos)], dim=1)
        J1 = torch.cat([J1, J1.new_tensor([1, 0, 0]).unsqueeze(0)], dim=0)
        J2 = torch.cat([torch.cat([self.pos2[Y:Y+1], -self.pos2[X:X+1]]).unsqueeze(1),
                        -torch.eye(DIM).type_as(self.pos)], dim=1)
        J2 = torch.cat([J2, J2.new_tensor([-1, 0, 0]).unsqueeze(0)], dim=0)
        return J1, J2

    def move(self, dt):
        self.update_pos()

    def update_pos(self):
        self.pos = self.body1.pos
        self.pos1 = self.pos - self.body1.pos
        if self.body2 is not None:
            # keep position on body1 as reference
            self.pos2 = self.pos - self.body2.pos

    def draw(self, screen, pixels_per_meter=1):
        start = (self.body1.pos.detach().cpu().numpy() * pixels_per_meter).astype(int)
        end = (self.body2.pos.detach().cpu().numpy() * pixels_per_meter).astype(int)
        return [pygame.draw.line(screen, (0, 255, 0), start, end, 2)]


class YConstraint:
    """Prevents motion in the Y axis.
    """
    def __init__(self, body1):
        self.static = True
        self.num_constraints = 1
        self.body1 = body1
        self.pos = body1.pos
        self.rot1 = self.body1.p[0]
        self.body2 = self.rot2 = None

    def J(self):
        J = self.pos.new_tensor([0, 0, 1]).unsqueeze(0)
        return J, None

    def move(self, dt):
        self.update_pos()

    def update_pos(self):
        self.pos = self.body1.pos
        self.rot1 = self.body1.p[0]

    def draw(self, screen, pixels_per_meter=1):
        pos = (self.pos.detach().cpu().numpy() * pixels_per_meter).astype(int)
        return [pygame.draw.line(screen, (0, 255, 0), pos - [5, 0], pos + [5, 0], 2)]


class XConstraint:
    """Prevents motion in the X axis.
    """
    def __init__(self, body1):
        self.static = True
        self.num_constraints = 1
        self.body1 = body1
        self.pos = body1.pos
        self.rot1 = self.body1.p[0]
        self.body2 = self.rot2 = None

    def J(self):
        J = self.pos.new_tensor([0, 1, 0]).unsqueeze(0)
        return J, None

    def move(self, dt):
        self.update_pos()

    def update_pos(self):
        self.pos = self.body1.pos
        self.rot1 = self.body1.p[0]

    def draw(self, screen, pixels_per_meter=1):
        pos = (self.pos.detach().cpu().numpy() * pixels_per_meter).astype(int)
        return [pygame.draw.line(screen, (0, 255, 0), pos - [0, 5], pos + [0, 5], 2)]


class RotConstraint:
    """Prevents rotational motion.
    """
    def __init__(self, body1):
        self.static = True
        self.num_constraints = 1
        self.body1 = body1
        self.pos = body1.pos
        self.rot1 = self.body1.p[0]
        self.body2 = self.rot2 = None

    def J(self):
        J = self.pos.new_tensor([1, 0, 0]).unsqueeze(0)
        return J, None

    def move(self, dt):
        self.update_pos()

    def update_pos(self):
        self.pos = self.body1.pos
        self.rot1 = self.body1.p[0]

    def draw(self, screen, pixels_per_meter=1):
        pos = (self.pos.detach().cpu().numpy() * pixels_per_meter).astype(int)
        return [pygame.draw.circle(screen, (0, 255, 0), pos, 5, 1)]


class TotalConstraint:
    """Prevents all motion.
    """
    def __init__(self, body1):
        self.static = True
        self.num_constraints = 3
        self.body1 = body1
        self.pos = body1.pos
        self.pos1 = self.pos - self.body1.pos
        self.r1, self.rot1 = cart_to_polar(self.pos1)

        self.body2 = self.rot2 = None
        self.eye = torch.eye(self.num_constraints).type_as(self.pos)

    def J(self):
        J = self.eye
        return J, None

    def move(self, dt):
        self.rot1 = self.rot1 + self.body1.v[0] * dt
        self.update_pos()

    def update_pos(self):
        self.pos1 = polar_to_cart(self.r1, self.rot1)
        self.pos = self.body1.pos + self.pos1

    def draw(self, screen, pixels_per_meter=1):
        pos = (self.pos.detach().cpu().numpy() * pixels_per_meter).astype(int)
        return [pygame.draw.circle(screen, (0, 255, 0), pos + 1, 5, 1),
                pygame.draw.line(screen, (0, 255, 0), pos - [5, 0], pos + [5, 0], 2),
                pygame.draw.line(screen, (0, 255, 0), pos - [0, 5], pos + [0, 5], 2)]
