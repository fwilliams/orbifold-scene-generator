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
        :param height: The height of the fundamental domain
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
            self._mirror_edges.append((e, (e + 1) % len(vs)))
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
            angle = np.arccos(cos_angle) % (2.0 * np.pi)
            if angle < min_angle:
                min_angle = angle  # min(min_angle, np.arccos(cos_angle) % (2.0 * np.pi))
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
        self._two_n = int(np.round(self._two_n))
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
        outer_edges.pop((min_angle_edge_index - 1) % len(outer_edges))

        self._translational_fd_edges = []
        for i in range(len(self._dihedral_transforms)):
            tx = self._dihedral_transforms[i]
            for j in range(len(outer_edges)):
                # This handles the case where there are 2 outer edges
                index = ((i % len(outer_edges)) + j) % len(outer_edges)
                e = outer_edges[index]
                v1, v2 = np.dot(tx, self._vertices[e[0]]), np.dot(tx, self._vertices[e[1]])
                self._translational_fd_edges.append((v1, v2) if (i % 2) == 0 else (v2, v1))

        i = 0
        while i < len(self._translational_fd_edges):
            v1, v2 = self._translational_fd_edges[i]
            v3, v4 = self._translational_fd_edges[(i+1) % len(self._translational_fd_edges)]
            if np.linalg.matrix_rank(np.column_stack((v2-v1, v4-v3)), tol=EPSILON) == 1:
                assert np.allclose(v2, v3) and not np.allclose(v1, v4)
                self._translational_fd_edges[i] = (v1, v4)
                self._translational_fd_edges.pop((i+1) % len(self._translational_fd_edges))
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
        for direction in (1, 0), (0, 1), (-1, 0), (0, -1):
            new_ctr = self._center + np.array(direction) * (self._diameter - overlap)
            yield SquareKernel(self._radius, new_ctr, self._group)

    @property
    def fundamental_domains(self):
        for pos, translate, _ in self.translational_fundamental_domains:
            for k in range(len(self._group.dihedral_subgroup)):
                transform = np.dot(translate, self._group.dihedral_subgroup[k])
                prism = shapes.Prism(self._group.height, *self._group.fd_vertices)
                prism.transform(transform)
                yield (pos, k), transform, prism

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
    def translational_fundamental_domain_transforms(self):
        for _, tx, _ in self.translational_fundamental_domains:
            yield tx

    @property
    def fundamental_domain_transforms(self):
        for _, tx, _ in self.fundamental_domains:
            yield tx

    @property
    def center(self):
        return self._center


class HexKernel:
    def __init__(self, radius, center, group):
        if group.n not in [3, 6]:
            raise ValueError("Cannot construct a hex kernel from planar group with dihedral "
                             "subgroup of order not 3 or 6. Got group of order %d" % group.n)
        self._group = group
        self._radius = radius
        self._diameter = 2 * radius + 1
        self._center = np.array(center)

    def __str__(self):
        return "Square Kernel: %d by %d centered at %s" % (self._diameter, self._diameter, str(self._center))

    def adjacent_kernels(self, overlap):
        directions = (1, 1), (-1, 2), (-2, 1), (-1, -1), (1, -2), (2, -1)
        shifts = (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1), (1, 0)
        for i in range(len(directions)):
            new_ctr = self.center + np.array(directions[i])*self._radius + np.array(shifts[i])*(1-overlap)
            yield HexKernel(self._radius, new_ctr, self._group)

    @property
    def fundamental_domains(self):
        for pos, translate, _ in self.translational_fundamental_domains:
            for k in range(len(self._group.dihedral_subgroup)):
                transform = np.dot(translate, self._group.dihedral_subgroup[k])
                prism = shapes.Prism(self._group.height, *self._group.fd_vertices)
                prism.transform(transform)
                yield (pos, k), transform, prism

    @property
    def translational_fundamental_domains(self):
        for i in range(-self._radius, self._radius+1):
            start = -self._radius if i >= 0 else -self._radius + abs(i)
            end = self._radius if i <= 0 else self._radius - abs(i)
            for j in range(start, end+1):
                tx = (self._center[0] + i) * self._group.translational_subgroup_basis[0] + \
                     (self._center[1] + j) * self._group.translational_subgroup_basis[1]
                transform = utils.translation_matrix(tx)
                prism = shapes.Prism(self._group.height, *self._group.translational_fd_vertices)
                prism.transform(transform)

                yield (self.center[0] + i, self.center[1] + j), transform, prism

    @property
    def translational_fundamental_domain_transforms(self):
        for _, tx, _ in self.translational_fundamental_domains:
            yield tx

    @property
    def fundamental_domain_transforms(self):
        for _, tx, _ in self.fundamental_domains:
            yield tx

    @property
    def center(self):
        return self._center


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
            raise ValueError("Error: Base kernel for KernelTiling does not intersect the specified frustum.")
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
            for _, tx, prism in next_kernel.translational_fundamental_domains:
                if shapes.intersects(self.frustum, prism):
                    self._add_kernel_rec(next_kernel, visited)
                    break
