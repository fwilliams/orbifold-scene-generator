import copy

import numpy as np

from geometry import shapes, utils

EPSILON = 1e-8


class PlanarReflectionGroup(object):
    def __init__(self, height, *vertices):
        """
        Construct the reflection group corresponding to a configuration of mirrors specified by the closed, convex
        polygon determined by the parameter vertices. The group must either be a purely dihedral group of integer order
        or a wallpaper group. In both cases, we construct the transformations for a dihedral group.
        In the latter case, we also construct a set of translational basis vectors and any element in
        the group is parameterized by a linear combination of the translation vectors as well as one of the
        transformations in the dihedral subgroup.

        :param vertices: The vertices of the fundamental domain.
        """

        #
        # Convert the vertices to projective coordinates
        #
        vs = [np.array((v[0], 0.0, v[1])) if len(v) == 2 else v for v in vertices]
        vs = [utils.make_projective_point(v) for v in vs]

        if not utils.coplanar(*vs):
            raise ValueError("Cannot construct a kaleidoscope from non-coplanar vertices")
        self._vertices = vs

        #
        # Store the ground plane and height of the ceiling
        #
        self._height = height
        self._ground_plane = shapes.Plane(vs[0], vs[1], vs[2])

        #
        # Populate mirror edge index pairs and list of vertices
        #
        self._mirror_edges = []
        self._mirror_planes = []

        for e in range(len(vertices)):
            v1, v2 = vs[e], vs[(e + 1) % len(vs)]
            vup = vs[e] + self._ground_plane.normal
            self._mirror_edges.append((e, e + 1 % len(vs)))
            self._mirror_planes.append(shapes.Plane(v1, v2, vup))

        #
        # Find the two edges with the mimimal internal angle and store their planes as plane1 and plane2
        # min_angle_edge_index stores the index of the edge before the minimum angle
        # plane1 and plane2 correspond to the mirror planes forming the dihedral subgroup of this reflection group
        # ctr_vertex index corresponds to the index of the vertex at the center of the dihedral tile
        #
        min_angle = np.pi * 2.0
        min_angle_edge_index = 0
        for i in range(len(self._vertices)):
            prv_i = (i - 1) % len(self._vertices)
            nxt_i = (i + 1) % len(self._vertices)
            prv = self._vertices[prv_i]
            cur = self._vertices[i]
            nxt = self._vertices[nxt_i]

            cos_angle = np.dot(prv - cur, nxt - cur) / (np.linalg.norm(prv-cur) * np.linalg.norm(nxt-cur))
            min_angle = min(min_angle, np.arccos(cos_angle) % (2.0 * np.pi))
            min_angle_edge_index = i

        plane1_index, plane2_index = (min_angle_edge_index - 1) % len(self._vertices), min_angle_edge_index
        plane1, plane2 = self._mirror_planes[plane1_index], self._mirror_planes[plane2_index]
        ctr_vertex_index = plane2_index

        #
        # Compute the order of the dihedral subgroup
        #
        angle = np.abs(np.pi - np.arccos(np.dot(plane1.normal, plane2.normal)))
        self._two_n = (2.0 * np.pi) / angle
        if np.abs(np.round(self._two_n) - self._two_n) > EPSILON:
            raise ValueError(
                "Reflection planes in dihedral group must have internal angle which is an integer divisor of "
                "two pi. Got %f which divides two pi into %f." % (angle, self._two_n))
        self._two_n = int(self._two_n)

        #
        # Compute the transformations of each element in the dihedral subgroup
        #
        self._dihedral_transforms = []
        last_transform = np.identity(4)
        for i in range(self._two_n):
            self._dihedral_transforms.append(last_transform)
            p = copy.deepcopy([plane1, plane2][i % 2])
            p.transform(last_transform)
            last_transform = np.dot(utils.reflection_matrix(p), last_transform)

        #
        # Compute the normals of the outer edges of the polygon (which bound the dihedral tile) and the distance from
        # the center of the dihedral tile to each outer edge.
        # Use this information to construct the translational basis vectors for the group.
        #
        outer_edges = copy.deepcopy(self._mirror_edges)
        outer_edges.pop(min_angle_edge_index)
        outer_edges.pop((min_angle_edge_index - 1) % len(self._vertices))

        self._translational_fd_edges = []
        for tx in self._dihedral_transforms:
            for e in outer_edges:
                v1, v2 = np.dot(tx, self._vertices[e[0]]), np.dot(tx, self._vertices[e[1]])
                self._translational_fd_edges.append((v1, v2))

        i = 0
        while i < len(self._translational_fd_edges):
            v1, v2 = self._translational_fd_edges[i]
            v3, v4 = self._translational_fd_edges[(i+1) % len(self._translational_fd_edges)]
            if np.linalg.matrix_rank(np.column_stack((v2-v1, v4-v3)), tol=EPSILON) == 1:
                if np.allclose(v2, v4):
                    self._translational_fd_edges[i] = (v1, v3)
                    self._translational_fd_edges.pop((i+1) % len(self._translational_fd_edges))
                elif np.allclose(v1, v3):
                    self._translational_fd_edges[i] = (v2, v4)
                    self._translational_fd_edges.pop((i+1) % len(self._translational_fd_edges))
                else:
                    assert False, "Bad case!"
            i += 1

        # Due to the dihedral symmetry, we know half the outer edges are just reflected copies of the other half
        # so we can delete them to get the set of edges whose normals form the basis
        self._translational_fd_vertices = [e[0] for e in self._translational_fd_edges]

        basis_edges = self._translational_fd_edges[0:len(self._translational_fd_edges)/2]
        self._translational_basis = \
            [2.0*(0.5 * (e[0] + e[1]) - self._vertices[ctr_vertex_index]) for e in basis_edges]

    @property
    def n(self):
        return self._two_n / 2

    @property
    def dihedral_subgroup(self):
        return self._dihedral_transforms

    @property
    def translational_subgroup_basis(self):
        if (self._two_n / 2) not in (1, 2, 4, 3, 6):
            raise ValueError("Dihedral group of order %d does not have a translational subgroup" % self._two_n)
        return self._translational_basis

    @property
    def fd_vertices(self):
        return self._vertices

    @property
    def fd_edges(self):
        return self._mirror_edges

    @property
    def translational_fd_vertices(self):
        return self._translational_fd_vertices

    @property
    def translational_fd_edges(self):
        return self._translational_fd_edges
        
    @property
    def height(self):
        return self._height

    @property
    def ground_plane(self):
        return self._ground_plane

    @property
    def mirror_planes(self):
        return self._mirror_planes


class SquareKernel:
    def __init__(self, radius, center, group):
        if group.n not in [2, 4]:
            raise ValueError("Cannot construct a square kernel from planar group with dihedral "
                             "subgroup of order not 2 or 4")
        self._group = group
        self._radius = radius
        self._diameter = 2 * radius + 1
        self._center = np.array(center)

    def __str__(self):
        return "Square Kernel: %d by %d centered at %s" % (self._diameter, self._diameter, str(self._center))

    def adjacent_kernels(self, overlap):
        for direction in (1, 0), (1, 0), (-1, 0), (0, -1):
            new_ctr = self._center + np.array(direction) * (self._diameter - overlap)
            yield SquareKernel(self._radius, new_ctr, self._group)

    @property
    def fundamental_domains(self):
        for pos, translate, _ in self.translational_fundamental_domains:
            print translate
            for k in range(len(self._group.dihedral_subgroup)):
                reflect = self._group.dihedral_subgroup[k]
                translate = np.dot(translate, reflect)
                prism = shapes.Prism(self._group.height, *self._group.fd_vertices)
                prism.transform(translate)
                yield (pos, k), translate, prism

    @property
    def translational_fundamental_domains(self):
        for i in range(self._diameter):
            for j in range(self._diameter):
                pos = np.array((i - self._radius, j - self._radius)) + np.array(self._center)
                translate = utils.translation_matrix(
                    pos[0] * self._group.translational_subgroup_basis[0] +
                    pos[1] * self._group.translational_subgroup_basis[1])

                prism = shapes.Prism(self._group.height, *self._group.translational_fd_vertices)
                prism.transform(translate)
                yield pos, translate, prism

    @property
    def center(self):
        return self._center


# TODO: Port to new framework
class LineKernel:
    def __init__(self, radius, fd_scale=np.array((1, 1, 1)), fd_ctr=np.array((0, 0, 0))):
        self._radius = radius
        self._diameter = 2 * radius + 1
        self._fd_center = np.array(fd_ctr)
        self._fd_scale = np.array(fd_scale)
        self._directions = (np.array((0, 0, 1)), np.array((0, 0, -1)))

    def __str__(self):
        return "Line Kernel: size = %d with center at %s" % (self._diameter, str(self._fd_center))

    def adjacent_kernels(self, overlap):
        for direction in self._directions:
            new_ctr = self._fd_center + direction * (self._diameter - overlap) * self._fd_scale
            yield LineKernel(self._radius, fd_scale=self._fd_scale, fd_ctr=new_ctr)

    @property
    def fundamental_domains(self):
        for i in range(2*self._diameter):
            x = i - self.radius*2
            if i % 2 != 0:
                p1 = self._fd_center
                p2 = p1 + np.array((self._fd_scale[0], 0, 0))
                p3 = p1 + np.array((0, self._fd_scale[1], 0))
                plane = shapes.Plane(p1, p2, p3)

                tx = utils.translation_matrix(self._fd_center - x * self._directions[0] * self._fd_scale)
                tx = np.dot(tx, utils.reflection_matrix(plane))
            else:
                tx = utils.translation_matrix(self._fd_center - x * self._directions[0] * self._fd_scale)
            aabb = shapes.AABB(self._fd_center - self._fd_scale*0.5, self._fd_center + self._fd_scale*0.5)
            aabb.transform(tx)
            yield x, tx, aabb

    @property
    def radius(self):
        return self._radius

    @property
    def center(self):
        return self._fd_center


# TODO: Port to new framework
class HexKernel:
    """
    A hexagonal kernel of hexagonal tiles. The radius specifies the number of tiles away from the center in every
    direction. The ctr parameter specifies the barycentric hex co-ordinate of the center tile.
    """
    def __init__(self, radius, ctr):
        self.radius = radius
        self.diameter = 2 * radius + 1
        self.center = ctr
        # right, upright, upleft, left, downleft, downright
        self.directions = ((-1, 1, 0), (-1, 0, 1), (0, -1, 1), (1, -1, 0), (1, 0, -1), (0, 1, -1))

    def __str__(self):
        return "Hex Kernel: radius of %d centered at %s" % (self.radius, str(self.center))

    def adjacent_kernel(self, direction, overlap):
        if direction not in self.directions:
            raise RuntimeError("Invalid direction, %s, passed to Kernel.adjacent_kernel" % (str(direction)))
        else:
            dir_index = self.directions.index(direction)
            next_dir = np.array(self.directions[(dir_index + 1) % len(self.directions)])
            new_ctr = np.array(self.center) + np.array(direction) * (self.radius + 1 - overlap) + next_dir*self.radius
            return HexKernel(self.radius, (new_ctr[0], new_ctr[1], new_ctr[2]))

    def prism(self, scale_factor, trans_kernel_ctr):
        # Use the 1st and 3rd co-ordinates as eisenstein integers. (https://en.wikipedia.org/wiki/Eisenstein_integer)
        # Use these integers to compute the euclidean co-ordinates of the center of the tile
        ni = np.array([0, 0, 1])
        nj = np.array([-np.sqrt(3)/2, 0, 0.5])
        euclidean_ctr = ni * self.center[2] + nj * self.center[0] + np.array(trans_kernel_ctr)

        aabb_hw = 1 + 1.5 * self.radius
        aabb_hh = (np.sqrt(3.0)/2.0) + self.radius * np.sqrt(3.0)

        scale = np.array([scale_factor[0], scale_factor[1], scale_factor[0]])
        bl = np.array([euclidean_ctr[0] - aabb_hw, -0.5,  euclidean_ctr[2] - aabb_hh]) * scale
        tr = np.array([euclidean_ctr[0] + aabb_hw, 0.5, euclidean_ctr[2] + aabb_hh]) * scale

        return shapes.AABB(bl, tr)


class InvalidBaseKernelError(Exception):
    pass


class KernelTiling:
    """
    A tiling of kernels which have the same shape which are fully or partially contained within a frustum.
    base_kernel specifies the shape of
    """
    def __init__(self, base_kernel, frustum, overlap):
        self.visible_kernels = []
        self.frustum = frustum
        self.overlap = overlap

        found_isect = False
        for i, tx, fd in base_kernel.fundamental_domains:
            if shapes.intersects(self.frustum, fd):
                found_isect = True
                break

        if not found_isect:
            raise InvalidBaseKernelError("Error: Base kernel for KernelTiling does "
                                         "not intersect the specified frustum.")
        visited_dict = dict()
        visited_dict[str(base_kernel.center)] = True
        self._add_kernel_rec(base_kernel, visited_dict)

    def __str__(self):
        return str([str(k) for k in self.visible_kernels])

    def _add_kernel_rec(self, kernel, visited):
        self.visible_kernels.append(kernel)
        for next_kernel in kernel.adjacent_kernels(self.overlap):
            if str(next_kernel.center) in visited:
                continue
            else:
                visited[str(next_kernel.center)] = True

            for i, tx, fd in next_kernel.translational_fundamental_domains:
                if shapes.intersects(self.frustum, fd):
                    self._add_kernel_rec(next_kernel, visited)
                    break
