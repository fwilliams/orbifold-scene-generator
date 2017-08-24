import copy

import numpy as np

from geometry import shapes, utils

EPSILON = 1e-8


class FriezeReflectionGroup(object):
    def __init__(self, height, up, mirror1_ctr, mirror2_ctr):
        ctr1 = utils.make_projective_point(mirror1_ctr)
        ctr2 = utils.make_projective_point(mirror2_ctr)
        up = utils.make_projective_vector(up)

        width = np.linalg.norm(ctr2 - ctr1)
        m1n = ctr2 - ctr1
        m1n /= width
        m2n = -m1n
        up_proj = up - np.dot(up, m1n)*m1n
        up_proj /= np.linalg.norm(up_proj)
        right_proj = utils.make_projective_vector(np.cross(up_proj[0:3], m1n[0:3]))
        right_proj2 = utils.make_projective_vector(np.cross(up_proj[0:3], m2n[0:3]))

        self._mirror_planes = [
            shapes.Plane(ctr1, ctr1 + up_proj, ctr1 + right_proj),
            shapes.Plane(ctr2, ctr2 + up_proj, ctr2 + right_proj2)
        ]
        self._dihedral_transforms = [np.eye(4), utils.reflection_matrix(self._mirror_planes[0])]
        self._translational_basis = [2*m1n*width, 2*m2n*width]
        b = ctr1 - up_proj * height * 0.5

        self._vertices = [b - right_proj*width*0.5, b + right_proj*width*0.5,
                          b + right_proj*width*0.5 + m1n*width,
                          b - right_proj*width*0.5 + m1n*width]

        self._mirror_edges = [(0, 1), (2, 3)]
        # tfdv = [b - right_proj*width*0.5, b + right_proj*width*0.5,
        #         b + right_proj*width*0.5 + m1n*width*2.0,
        #         b - right_proj * width * 0.5 + m1n*width*2.0]
        tfdv = [b - right_proj*width*0.5 - m1n*width,
                b + right_proj*width*0.5 - m1n*width,
                b + right_proj*width*0.5 + m1n*width,
                b - right_proj * width * 0.5 + m1n*width]
        self._translational_fd_vertices = tfdv
        self._translational_fd_edges = [(tfdv[0], tfdv[1]),
                                        (tfdv[1], tfdv[2]),
                                        (tfdv[2], tfdv[3]),
                                        (tfdv[3], tfdv[0])]
        self._height = height
        self._ground_plane = shapes.Plane(b, b + m1n, b + right_proj)

    @property
    def n(self):
        return 1

    @property
    def dihedral_subgroup(self):
        return self._dihedral_transforms

    @property
    def translational_subgroup_basis(self):
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
        # Compute the nor_mirror_planesmals of the outer edges of the polygon
        # (which bound the dihedral tile) and the distance from
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


class SquareKernel(object):
    def __init__(self, hradius, vradius, center, hgroup, vgroup, flag_ceiling, flag_floor):
        if hgroup.n not in [2, 4]:
            raise ValueError("Cannot construct a horizontal square kernel from planar group with dihedral "
                             "subgroup of order not 2 or 4")

        if vgroup.n not in [1]:
            raise ValueError("Cannot construct a vertical line kernel from planar group with dihedral "
                             "subgroup of order not 1")

        self._vradius = vradius
        self._vdiameter = 0 if not flag_ceiling and not flag_floor else 2 * vradius + 1
        self._hradius = hradius
        self._hdiameter = 2 * hradius + 1
        self._center = np.array(center)
        self._hgroup = hgroup
        self._vgroup = vgroup
        self._flag_ceiling = flag_ceiling
        self._flag_floor = flag_floor
        self._fd_height = hgroup.height if not flag_ceiling and not flag_floor else vgroup.height
        self._tfd_height = self._fd_height if not flag_ceiling and not flag_floor else self._fd_height*2

    def __str__(self):
        return "Square Kernel: %d by %d centered at %s" % (self._diameter, self._diameter, str(self._center))

    def adjacent_kernels(self, overlap):

        neighbor_directions = [(1, 0, 0), (0, 1, 0), (-1, 0, 0), (0, -1, 0)]

        if self._flag_ceiling:
            neighbor_directions.append((0, 0, 1))
        if self._flag_floor:
            neighbor_directions.append((0, 0, -1))

        for neighbor in neighbor_directions:

            h_direction = np.array((neighbor[0], neighbor[1]))
            v_direction = neighbor[2]

            if np.size(self._center) is not 3:
                raise ValueError("The R3 tiling is not correct")

            h_cor = np.array((self._center[0], self._center[1]))+ h_direction * (self._hdiameter - overlap)
            v_Cor = self._center[2] + v_direction * (self._vdiameter - overlap)

            new_ctr = np.array((h_cor[0], h_cor[1], v_Cor))
            yield SquareKernel(self._hradius, self._vradius, new_ctr, self._hgroup, self._vgroup, self._flag_ceiling, self._flag_floor)

    @property
    def fundamental_domains(self):
        for pos, translate, _ in self.translational_fundamental_domains:

            #If no ceiling or floor mirrors, fds only in one layer
            if not self._flag_ceiling and not self._flag_floor:
                for k_h in range(len(self._hgroup.dihedral_subgroup)):
                    transform = self._hgroup.dihedral_subgroup[k_h]
                    transform = np.dot(translate, transform)

                    prism = shapes.Prism(self._fd_height, *self._hgroup.fd_vertices)
                    prism.transform(transform)
                    yield (pos, k_h, 0), transform, prism

            # If there is ceiling or floor mirrors, fds only in two layers
            else:
                for k_h in range(len(self._hgroup.dihedral_subgroup)):
                    for k_v in range(len(self._vgroup.dihedral_subgroup)):
                        transform = np.dot(self._hgroup.dihedral_subgroup[k_h], self._vgroup.dihedral_subgroup[k_v])
                        transform = np.dot(translate, transform)

                        prism = shapes.Prism(self._fd_height, *self._hgroup.fd_vertices)
                        prism.transform(transform)
                        yield (pos, k_h, k_v), transform, prism

    @property
    def translational_fundamental_domains(self):

        for j in range(self._hdiameter):
            for k in range(self._hdiameter):
                for i in range(1) if not self._flag_ceiling and not self._flag_floor else range(self._vdiameter):

                    h_pos = np.array((j - self._hradius, k - self._hradius)) + np.array((self.center[0],self.center[1]))
                    v_pos = np.array((i - self._vradius)) + np.array(self._center[2]) if self._vdiameter != 0 else 0

                    if self._flag_ceiling and not self._flag_floor and v_pos < 0:
                        continue;

                    if self._flag_floor and not self._flag_ceiling and v_pos > 0:
                        continue;

                    v_ctr = abs(v_pos) * self._vgroup.translational_subgroup_basis[1 if v_pos > 0 else 0]
                    if self._flag_floor and not self._flag_ceiling:
                        v_ctr = abs(v_pos) * self._vgroup.translational_subgroup_basis[0 if v_pos > 0 else 1]

                    h_ctr = h_pos[0] * self._hgroup.translational_subgroup_basis[0] + h_pos[1] * self._hgroup.translational_subgroup_basis[1]

                    translate = utils.translation_matrix(v_ctr + h_ctr)

                    prism = shapes.Prism(self._tfd_height, *self._hgroup.translational_fd_vertices)

                    prism.transform(translate)
                    yield np.array([h_pos[0],h_pos[1], v_pos]), translate, prism

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


class HexKernel(object):
    def __init__(self, hradius, vradius, center, hgroup, vgroup, flag_ceiling, flag_floor):
        if hgroup.n not in [3, 6]:
            raise ValueError("Cannot construct a hex kernel from planar group with dihedral "
                             "subgroup of order not 3 or 6. Got group of order %d" % hgroup.n)
        if vgroup.n not in [1]:
            raise ValueError("Cannot construct a vertical line kernel from planar group with dihedral "
                             "subgroup of order not 1")

        self._vradius = vradius
        self._vdiameter = 0 if not flag_ceiling and not flag_floor else 2 * vradius + 1
        self._hradius = hradius
        self._hdiameter = 2 * hradius + 1
        self._center = np.array(center)
        self._hgroup = hgroup
        self._vgroup = vgroup
        self._flag_ceiling = flag_ceiling
        self._flag_floor = flag_floor
        self._fd_height = hgroup.height if not flag_ceiling and not flag_floor else vgroup.height
        self._tfd_height = self._fd_height if not flag_ceiling and not flag_floor else self._fd_height*2

    def __str__(self):
        return "Hex Kernel: %d by %d centered at %s" % (self._diameter, self._diameter, str(self._center))

    def adjacent_kernels(self, overlap):

        neighbor_directions = [(1, 1, 0), (-1, 2, 0), (-2, 1, 0), (-1, -1, 0), (1, -2, 0), (2, -1, 0)]
        shifts = [(0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1), (1, 0)]

        if self._flag_ceiling:
            neighbor_directions.append((0, 0, 1))
            shifts.append((0, 0))

        if self._flag_floor:
            neighbor_directions.append((0, 0, -1))
            shifts.append((0, 0))

        for i in range(len(neighbor_directions)):

            neighbor = neighbor_directions[i]
            h_direction = np.array((neighbor[0], neighbor[1]))
            direction = neighbor[2]

            if np.size(self._center) is not 3:
                raise ValueError("The R3 tiling is not correct")

            h_cor = np.array((self._center[0], self._center[1])) + np.array(h_direction)*self._hradius + np.array(shifts[i])*(1-overlap)
            v_Cor = self._center[2] + direction * (self._vdiameter - overlap)

            new_ctr = np.array((h_cor[0], h_cor[1], v_Cor))
            yield HexKernel(self._hradius, self._vradius, new_ctr, self._hgroup, self._vgroup, self._flag_ceiling, self._flag_floor)

    @property
    def fundamental_domains(self):
        for pos, translate, _ in self.translational_fundamental_domains:
            if not self._flag_ceiling and not self._flag_floor:
                for k_h in range(len(self._hgroup.dihedral_subgroup)):
                    transform = self._hgroup.dihedral_subgroup[k_h]
                    transform = np.dot(translate, transform)

                    prism = shapes.Prism(self._fd_height, *self._hgroup.fd_vertices)
                    prism.transform(transform)
                    yield (pos, k_h, 0), transform, prism
            else:
                for k_h in range(len(self._hgroup.dihedral_subgroup)):
                    for k_v in range(len(self._vgroup.dihedral_subgroup)):
                        transform = np.dot(self._hgroup.dihedral_subgroup[k_h], self._vgroup.dihedral_subgroup[k_v])
                        transform = np.dot(translate, transform)

                        prism = shapes.Prism(self._fd_height, *self._hgroup.fd_vertices)
                        prism.transform(transform)
                        yield (pos, k_h, k_v), transform, prism

    @property
    def translational_fundamental_domains(self):

        for i in range(-self._hradius, self._hradius+1):
            start = -self._hradius if i >= 0 else -self._hradius + abs(i)
            end = self._hradius if i <= 0 else self._hradius - abs(i)
            for j in range(start, end+1):

                for i_r3 in range(1) if not self._flag_ceiling and not self._flag_floor else range(self._vdiameter):

                    v_pos = np.array((i_r3 - self._vradius)) + np.array(self._center[2])if self._vdiameter != 0 else 0

                    if self._flag_ceiling and not self._flag_floor and v_pos < 0:
                        continue;

                    if self._flag_floor and not self._flag_ceiling and v_pos > 0:
                        continue;

                    v_ctr = abs(v_pos) * self._vgroup.translational_subgroup_basis[1 if v_pos > 0 else 0]
                    if self._flag_floor and not self._flag_ceiling:
                        v_ctr = abs(v_pos) * self._vgroup.translational_subgroup_basis[0 if v_pos > 0 else 1]

                    h_ctr = (self._center[0] + i) * self._hgroup.translational_subgroup_basis[0] + \
                             (self._center[1] + j) * self._hgroup.translational_subgroup_basis[1]

                    transform = utils.translation_matrix(v_ctr + h_ctr)
                    prism = shapes.Prism(self._tfd_height, *self._hgroup.translational_fd_vertices)
                    prism.transform(transform)

                    yield (self.center[0] + i, self.center[1] + j, v_pos), transform, prism

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

class LineKernel(object):
    def __init__(self, hradius, vradius, center, hgroup, vgroup, flag_ceiling, flag_floor):
        if vgroup.n not in [1]:
            raise ValueError("Cannot construct a vertical line kernel from planar group with dihedral "
                             "subgroup of order not 1")

        if hgroup.n not in [1]:
            raise ValueError("Cannot construct a horizontal line kernel from planar group with dihedral "
                             "subgroup of order not 1")

        self._vradius = vradius
        self._vdiameter = 0 if not flag_ceiling and not flag_floor else 2 * vradius + 1
        self._hradius = hradius
        self._hdiameter = 2 * hradius + 1
        self._center = np.array(center)
        self._hgroup = hgroup
        self._vgroup = vgroup
        self._flag_ceiling = flag_ceiling
        self._flag_floor = flag_floor
        self._fd_height = hgroup.height if not flag_ceiling and not flag_floor else vgroup.height
        self._tfd_height = self._fd_height if not flag_ceiling and not flag_floor else self._fd_height*2

    def __str__(self):
        return "Line Kernel: %d by %d centered at %s" % (self._diameter, self._diameter, str(self._center))

    def adjacent_kernels(self, overlap):

        neighbor_directions = [(1, 0), (-1, 0)]

        if self._flag_ceiling:
            neighbor_directions.append((0, 1))

        if self._flag_floor:
            neighbor_directions.append((0, -1))

        for neighbor in neighbor_directions:

            h_direction = neighbor[0]
            direction = neighbor[1]

            if np.size(self._center) is not 2:
                raise ValueError("The R3 tiling is not correct")

            xCor = self._center[0] + h_direction * (self._hdiameter - overlap)
            yCor = self._center[1] + direction * (self._vdiameter - overlap)
            new_ctr = np.array((xCor, yCor))
            yield LineKernel(self._hradius, self._vradius, new_ctr,self._hgroup, self._vgroup, self._flag_ceiling, self._flag_floor)

    @property
    def fundamental_domains(self):
        for pos, translate, _ in self.translational_fundamental_domains:

            if not self._flag_ceiling and not self._flag_floor:
                for k_h in range(len(self._hgroup.dihedral_subgroup)):
                    transform = self._hgroup.dihedral_subgroup[k_h]
                    transform = np.dot(translate, transform)

                    prism = shapes.Prism(self._fd_height, *self._hgroup.fd_vertices)
                    prism.transform(transform)
                    yield (pos, k_h, 0), transform, prism
            else:
                for k_h in range(len(self._hgroup.dihedral_subgroup)):
                    for k_v in range(len(self._vgroup.dihedral_subgroup)):
                        transform = np.dot(self._hgroup.dihedral_subgroup[k_h], self._vgroup.dihedral_subgroup[k_v])
                        transform = np.dot(translate, transform)

                        prism = shapes.Prism(self._fd_height, *self._hgroup.fd_vertices)
                        prism.transform(transform)
                        yield (pos, k_h, k_v), transform, prism

    @property
    def translational_fundamental_domains(self):

        for j in range(self._hdiameter):

            for i in range(1) if not self._flag_ceiling and not self._flag_floor else range(self._vdiameter):

                h_pos = np.array((j - self._hradius)) + np.array(self._center[0])
                v_pos = np.array((i - self._vradius)) + np.array(self._center[1]) if self._vdiameter != 0 else 0

                if self._flag_ceiling and not self._flag_floor and v_pos < 0:
                    continue;

                if self._flag_floor and not self._flag_ceiling and v_pos > 0:
                    continue;

                v_ctr = abs(v_pos) * self._vgroup.translational_subgroup_basis[1 if v_pos > 0 else 0]
                if self._flag_floor and not self._flag_ceiling:
                    v_ctr = abs(v_pos) * self._vgroup.translational_subgroup_basis[0 if v_pos > 0 else 1]

                h_ctr = abs(h_pos) * self._hgroup.translational_subgroup_basis[1 if h_pos > 0 else 0]

                translate = utils.translation_matrix(v_ctr + h_ctr)

                prism = shapes.Prism(self._tfd_height, *self._hgroup.translational_fd_vertices)
                prism.transform(translate)
                yield np.array([h_pos, v_pos]), translate, prism

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


# class DihedralKernel(object):
#     def __init__(self, group):
#         self._group = group
#
#     def __str__(self):
#         return "Dihedral Kernel of order %d" % self._group.n
#
#     def adjacent_kernels(self, overlap):
#         return []
#
#     @property
#     def fundamental_domains(self):
#         for k in range(len(self._group.dihedral_subgroup)):
#             transform = self._group.dihedral_subgroup[k]
#             prism = shapes.Prism(self._group.height, *self._group.fd_vertices)
#             prism.transform(transform)
#             yield (0, k), transform, prism
#
#     @property
#     def translational_fundamental_domains(self):
#         prism = shapes.Prism(self._group.height, *self._group.translational_fd_vertices)
#         yield 0, np.eye(4), prism
#
#     @property
#     def translational_fundamental_domain_transforms(self):
#         for _, tx, _ in self.translational_fundamental_domains:
#             yield tx
#
#     @property
#     def fundamental_domain_transforms(self):
#         for _, tx, _ in self.fundamental_domains:
#             yield tx
#
#     @property
#     def center(self):
#         return 0
#

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
