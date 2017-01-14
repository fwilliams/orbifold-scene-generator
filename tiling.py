import copy

import numpy as np

from geometry import shapes, utils

EPSILON = 1e-12


class ReflectionOrbifold(object):
    def __init__(self, height, vertices, mirror_edges):
        """
        Create a generic reflection group with a fundamental domain whose corners are specified in order by vertices.
        The mirrors will consist of the edges in mirror_edges which are specified as integer indexes in the range
        [0, len(vertices)] and edge i has end vertices i and i+1 mod len(vertices).

        :param vertices: The vertices of the fundamental domain
        :param mirror_edges: The edges of the fundamental domain

        """

        vs = [np.array((v[0], 0.0, v[1])) if len(v) == 2 else v for v in vertices]
        vs = [utils.make_projective_point(v) for v in vs]

        if not utils.coplanar(*vs):
            raise ValueError("Cannot construct a kaleidoscope from non-coplanar vertices")

        self._height = height
        self._ground_plane = shapes.Plane(vs[0], vs[1], vs[2])
        self._mirror_edges = []
        self._mirror_planes = []
        self._mirror_transforms = []

        for e in mirror_edges:
            v1, v2 = vs[e], vs[(e + 1) % len(vs)]
            vup = vs[e] + self._ground_plane.normal
            self._mirror_edges.append((e, e + 1 % len(vs)))
            self._mirror_planes.append(shapes.Plane(v1, v2, vup))
            self._mirror_transforms.append(utils.reflection_matrix(self._mirror_planes[-1]))
        self._vertices = vs

    @property
    def vertices(self):
        return self._vertices

    @property
    def mirror_edges(self):
        return self._mirror_edges

    @property
    def mirror_planes(self):
        return self._mirror_planes

    @property
    def mirror_transforms(self):
        return self._mirror_transforms

    @property
    def ground_plane(self):
        return self._ground_plane

    @property
    def height(self):
        return self._height


class X2222(ReflectionOrbifold):
    def __init__(self, h, v1, v2, v3, v4):
        super(X2222, self).__init__(h, (v1, v2, v3, v4), (0, 1, 2, 3))

        for i in range(len(self._vertices)):
            prv = self._vertices[(i - 1) % len(self._vertices)]
            cur = self._vertices[i]
            nxt = self._vertices[(i + 1) % len(self._vertices)]

            cos_angle = np.dot(prv - cur, nxt - cur)
            if cos_angle != 0.0:
                raise ValueError("*2222 Orbifold vertices must form a rectangle. Got inner angle between vertices "
                                 "%s, %s, %s = %f degrees"
                                 % (str(prv), str(nxt), str(cur), np.degrees(np.arccos(cos_angle))))

        w = np.linalg.norm(self.vertices[1] - self.vertices[0])
        d = np.linalg.norm(self.vertices[2] - self.vertices[1])

        self.scale = np.array((w, d))
        self._real_basis = np.array((self.mirror_planes[0].normal * w * 2, self.mirror_planes[1].normal * d * 2))

    @property
    def translational_basis(self):
        return self._real_basis

    def translational_fds(self, translational_lattice_coord):
        reflect1 = utils.reflection_matrix(self.mirror_planes[0])
        reflect2 = utils.reflection_matrix(self.mirror_planes[1])
        base = X2222.translational_lattice_to_lattice(translational_lattice_coord)
        return [(np.array((0, 0)) + base, np.identity(4)),
                (np.array((1, 0)) + base, reflect1),
                (np.array((1, 1)) + base, np.dot(reflect2, reflect1)),
                (np.array((0, 1)) + base, reflect2)]

    @staticmethod
    def translational_lattice_to_lattice(tx_lattice):
        utils.verify_matrix_shape(tx_lattice, 2)
        return np.array(tx_lattice) * 2.0


class X442(ReflectionOrbifold):
    def __init__(self, h, v1, v2, v3):
        super(X442, self).__init__(h, (v1, v2, v3), (0, 1, 2))

        self._real_basis = None

        for i in range(len(self._vertices)):
            prv_i = (i - 1) % len(self._vertices)
            nxt_i = (i + 1) % len(self._vertices)
            prv = self._vertices[prv_i]
            cur = self._vertices[i]
            nxt = self._vertices[nxt_i]

            cos_angle = np.dot(prv - cur, nxt - cur)

            if abs(cos_angle) < EPSILON:
                if self._real_basis:
                    raise ValueError("*442 Orbifold vertices must form a right angle isoceles triangle. "
                                     "Got inner angle between vertices " "%s, %s, %s = %f degrees" %
                                     (str(prv), str(nxt), str(cur), np.degrees(np.arccos(cos_angle))))
                w = np.linalg.norm(prv - cur)
                d = np.linalg.norm(nxt - cur)
                self._real_basis = np.array((self.mirror_planes[prv_i].normal * w * 2,
                                             self.mirror_planes[nxt_i].normal * d * 2))
                self.scale = np.array((w, d))
                self._hypoteneuse = nxt_i
            elif abs(np.arccos(cos_angle) - np.pi/4.0) > EPSILON:
                raise ValueError("*442 Orbifold vertices must form a right angle isoceles triangle. "
                                 "Got inner angle between vertices " "%s, %s, %s = %f degrees" %
                                 (str(prv), str(nxt), str(cur), np.degrees(np.arccos(cos_angle))))

    @property
    def translational_basis(self):
        return self._real_basis

    def translational_fds(self, translational_lattice_coord):
        reflect1 = utils.reflection_matrix(self.mirror_planes[self._hypoteneuse])
        p = copy.deepcopy(self.mirror_planes[(self._hypoteneuse + 1) % len(self.vertices)])
        p.transform(reflect1)
        reflect2 = utils.reflection_matrix(p)
        p = copy.deepcopy(self.mirror_planes[self._hypoteneuse])
        p.transform(reflect2)
        reflect3 = utils.reflection_matrix(p)

        p = copy.deepcopy(self.mirror_planes[(self._hypoteneuse + 2) % len(self.vertices)])
        reflect4 = np.dot(utils.reflection_matrix(p), reflect3)
        reflect5 = np.dot(utils.reflection_matrix(p), reflect2)
        reflect6 = np.dot(utils.reflection_matrix(p), reflect1)
        reflect7 = utils.reflection_matrix(p)

        base = X2222.translational_lattice_to_lattice(translational_lattice_coord)
        return [(np.array((0, 0, 0, 0)) + base, np.identity(4)),
                (np.array((1, 0, 0, 0)) + base, reflect1),
                (np.array((1, 1, 0, 0)) + base, reflect2),
                (np.array((1, 1, 1, 0)) + base, reflect3),
                (np.array((1, 1, 1, 1)) + base, reflect4),
                (np.array((0, 1, 1, 1)) + base, reflect5),
                (np.array((0, 0, 1, 1)) + base, reflect6),
                (np.array((0, 0, 0, 1)) + base, reflect7)]

    @staticmethod
    def translational_lattice_to_lattice(tx_lattice):
        utils.verify_matrix_shape(tx_lattice, 2)
        return tx_lattice[0] * np.array((1, 2, 1, 0)) + tx_lattice[1] * np.array((1, 0, 1, 2))
        return np.array(tx_lattice) * 2.0


class SquareKernel:
    def __init__(self, radius, center, fundamental_domain):
        self._fd = fundamental_domain
        self._radius = radius
        self._diameter = 2 * radius + 1
        self._center = np.array(center)

    def __str__(self):
        return "Square Kernel: %d by %d centered at %s" % (self._diameter, self._diameter, str(self._center))

    def adjacent_kernels(self, overlap):
        for direction in (1, 0), (1, 0), (-1, 0), (0, -1):
            new_ctr = self._center + np.array(direction) * (self._diameter - overlap)
            yield SquareKernel(self._radius, new_ctr, self._fd)

    @property
    def fundamental_domains(self):
        for i in range(self._diameter):
            for j in range(self._diameter):
                pos = np.array((i - self._radius, j - self._radius)) + np.array(self._center)
                translate = utils.translation_matrix(
                    pos[0] * self._fd.translational_basis[0] + pos[1] * self._fd.translational_basis[1])

                for coord, fd_tx in self._fd.translational_fds(pos):
                    tx = np.dot(translate, fd_tx)
                    prism = shapes.Prism(self._fd.height, *self._fd.vertices)
                    prism.transform(tx)
                    yield coord, tx, prism

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

            for i, tx, fd in next_kernel.fundamental_domains:
                if shapes.intersects(self.frustum, fd):
                    self._add_kernel_rec(next_kernel, visited)
                    break
