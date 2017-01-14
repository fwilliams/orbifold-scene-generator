import numpy as np
from multimethods import multimethod

import utils

# TODO: Make this configurable
EPSILON = 1e-12


class Plane(object):
    """
    A plane defined by 3 points in R^3, specified counter-clockwise with respect to the normal.
    The position of the base of the normal is p1
    """

    def __init__(self, p0, p1, p2):
        p0 = utils.make_projective_point(p0)
        p1 = utils.make_projective_point(p1)
        p2 = utils.make_projective_point(p2)

        v1 = (p1 - p0)[:3]
        v2 = (p2 - p1)[:3]
        n = np.cross(v1, v2)
        n /= -np.linalg.norm(n)
        self._normal = utils.make_projective_vector(n)
        self._position = utils.make_projective_point(p1)

    @property
    def normal(self):
        return self._normal

    @property
    def position(self):
        return self._position

    @property
    def normal3(self):
        return self._normal[:3]

    @property
    def position3(self):
        return self._position[:3]

    def transform(self, tx):
        tx = utils.verify_matrix_shape(tx, 4, 4)

        normal_tx = np.transpose(np.linalg.inv(tx[:3, :3]))
        n = normal_tx.dot(self.normal3)
        self._normal = utils.make_projective_vector(n)
        self._position = utils.make_projective_point(tx.dot(self.position))

    def signed_distance(self, point):
        point = utils.make_projective_point(point)
        c_to_p = point - self.position
        return np.dot(c_to_p, self.normal)

    def contains(self, point, epsilon=EPSILON):
        return np.abs(self.signed_distance(point)) < epsilon


class Polyhedron(object):
    def __init__(self):
        self._vertices = []
        self._edges = []
        self._faces = []
        self._planes = []
        self._num_vertices = len(self._vertices)
        self._num_edges = len(self._edges)
        self._num_faces = len(self._faces)

    @property
    def vertices(self):
        return self._vertices

    @property
    def vertices(self):
        return self._vertices

    @property
    def edges(self):
        return self._edges

    @property
    def faces(self):
        return self._faces

    @property
    def planes(self):
        return self._planes

    @property
    def num_vertices(self):
        return self._num_vertices

    @property
    def num_edges(self):
        return self._num_edges

    @property
    def num_faces(self):
        return self._num_faces

    def transform(self, tx):
        tx = utils.verify_matrix_shape(tx, 4, 4)
        for i in range(len(self._vertices)):
            self._vertices[i] = utils.make_projective_point(tx.dot(self._vertices[i]))
        for p in self.planes:
            p.transform(tx)


class Prism(Polyhedron):
    def __init__(self, h, *args):
        super(Prism, self).__init__()

        self._vertices = []
        input_shape = np.array(args[0]).shape

        # Verify that all the input vertices are of dimension 2 or 3. If they are of dimension 2 store them in 3D.
        for a in args:
            pt = np.array(a)

            if pt.shape != input_shape:
                raise ValueError("Prism base vertices do not all have the same dimension")

            if pt.shape == (2,):
                pt = np.array((pt[0], pt[1], 0.0, 1.0))
            elif pt.shape == (3,):
                pt = np.array((pt[0], pt[1], pt[2], 1.0))
            elif pt.shape == (4,):
                pt = utils.make_projective_point(pt)
            else:
                raise ValueError("Prism cannot be constructed from non 2d, 3d or projective 3d points. Got %s" % pt)

            self._vertices.append(pt)

        self._faces = [tuple([i for i in range(len(self._vertices))])]
        self._planes = [Plane(self._vertices[0], self._vertices[1], self._vertices[2])]  # The base plane
        self._edges = []

        base_degree = len(self._vertices)
        for i in range(base_degree):
            prv_i = (i-1) % base_degree
            nxt_i = (i+1) % base_degree

            cur = self._vertices[i]
            prv = self._vertices[prv_i]
            nxt = self._vertices[nxt_i]

            if not self._planes[0].contains(cur):
                # if np.dot(cur - self._planes[0].position, self._planes[0].normal) > EPSILON:
                raise ValueError("Prism base vertices are not co-planar")

            v1 = prv - cur
            v1 /= np.linalg.norm(v1)
            v2 = nxt - cur
            v2 /= np.linalg.norm(v2)

            if np.dot(np.cross(v1[:3], v2[:3]), self._planes[0].normal3) < 0:
                raise ValueError("Prism base polygon is not convex")

            self._vertices.append(cur + h * -self._planes[0].normal)
            self._faces.append((i, prv_i, prv_i+base_degree, i+base_degree))
            self._planes.append(Plane(self._vertices[-1], cur, prv))
            self._edges.append((i, nxt_i))
            self._edges.append((i, i + base_degree))
            self._edges.append((i + base_degree, nxt_i + base_degree))

        self._num_vertices = len(self._vertices)
        self._num_edges = len(self._edges)

        self._planes.append(Plane(self._vertices[-1], self._vertices[-2], self._vertices[-3]))
        self._faces.append(tuple([self._num_vertices - 1 - i for i in range(self._num_vertices - base_degree)]))
        self._num_faces = len(self._faces)


class Frustum(Polyhedron):
    def __init__(self, ctr, far_dist, *args):
        super(Frustum, self).__init__()

        if far_dist <= 0.0:
            raise ValueError("Frustum far distance must be positive and non-zero")

        self._origin = utils.make_projective_point(ctr)
        input_shape = np.array(args[0]).shape

        # Verify that all the input vertices are of dimension 2 or 3. If they are of dimension 2 store them in 3D.
        for a in args:
            pt = np.array(a)

            if pt.shape != input_shape:
                raise ValueError("Frustum base vertices do not all have the same dimension")

            if pt.shape == (2,):
                pt = np.array((pt[0], pt[1], 0.0, 1.0))
            elif pt.shape == (3,):
                pt = np.array((pt[0], pt[1], pt[2], 1.0))
            elif pt.shape == (4,):
                pt = utils.make_projective_point(pt)
            else:
                raise ValueError("Prism cannot be constructed from non 2d, 3d or projective 3d points. Got %s" % pt)

            self._vertices.append(pt)

        self._faces = [tuple([i for i in range(len(self._vertices))])]
        self._planes = [Plane(self._vertices[0], self._vertices[1], self._vertices[2])]  # The near plane

        near_dist = self._planes[0].signed_distance(self._origin)
        if near_dist == 0:
            raise ValueError("Center of frusum cannot be on the near plane")
        elif near_dist < 0.0:
            self._faces[0] = self.faces[0][::-1]
            self._vertices = self._vertices[::-1]
            self._planes = [Plane(self._vertices[0], self._vertices[1], self._vertices[2])]
            near_dist = self._planes[0].signed_distance(self._origin)

        ctr_on_near_plane = self._origin - self._planes[0].normal * near_dist
        base_degree = len(self._vertices)
        for i in range(base_degree):
            prv_i = (i-1) % base_degree
            nxt_i = (i+1) % base_degree

            cur = self._vertices[i]
            prv = self._vertices[prv_i]
            nxt = self._vertices[nxt_i]

            if not self._planes[0].contains(cur):
                # if np.dot(cur - self._planes[0].position, self._planes[0].normal) > EPSILON:
                raise ValueError("Frustum base vertices are not co-planar")

            v1 = prv - cur
            v1 /= np.linalg.norm(v1)
            v2 = nxt - cur
            v2 /= np.linalg.norm(v2)

            if np.dot(np.cross(v1[:3], v2[:3]), self._planes[0].normal3) < 0:
                raise ValueError("Frustum base polygon is not convex")

            ctr_to_cur = utils.normalize((cur - self._origin))
            near_ctr_to_cur = cur - ctr_on_near_plane
            scale = np.sqrt(1.0 + (np.linalg.norm(near_ctr_to_cur)/near_dist)**2.0) * (near_dist + far_dist)
            self._vertices.append(self._origin + scale * ctr_to_cur)
            self._faces.append((i, prv_i, prv_i+base_degree, i+base_degree))
            self._planes.append(Plane(self._vertices[-1], cur, prv))
            self._edges.append((i, nxt_i))
            self._edges.append((i, i + base_degree))
            self._edges.append((i + base_degree, nxt_i + base_degree))

        self._num_vertices = len(self._vertices)
        self._num_edges = len(self._edges)

        self._planes.append(Plane(self._vertices[-1], self._vertices[-2], self._vertices[-3]))
        self._faces.append(tuple([self._num_vertices - 1 - i for i in range(self._num_vertices - base_degree)]))
        self._num_faces = len(self._faces)

    @property
    def origin(self):
        return self._origin

    def transform(self, tx):
        super(Frustum, self).transform(tx)
        self._origin = tx.dot(self._origin)


class CameraFrustum(Frustum):
    """
    A frustum whose planes are normal to the positive Z axis. It is specified by 8 values determining, the size and
    skew of the near and far planes as well as the distance along Z to the near and far planes. The parameters are the
    same as the class glFrustum (https://www.opengl.org/sdk/docs/man2/xhtml/glFrustum.xml)
    """

    def __init__(self, left, right, bottom, top, near, far):
        ntl = np.array([left, top, near, 1.0])
        ntr = np.array([right, top, near, 1.0])
        nbl = np.array([left, bottom, near, 1.0])
        nbr = np.array([right, bottom, near, 1.0])

        super(CameraFrustum, self).__init__((0, 0, 0), far - near, ntl, nbl, nbr, ntr)

        self._lookat = np.array((0, 0, 1, 0))

    @property
    def ntl(self):
        return self.vertices[0]

    @property
    def nbl(self):
        return self.vertices[1]

    @property
    def nbr(self):
        return self.vertices[2]

    @property
    def ntr(self):
        return self.vertices[3]

    @property
    def ftl(self):
        return self.vertices[4]

    @property
    def fbl(self):
        return self.vertices[5]

    @property
    def fbr(self):
        return self.vertices[6]

    @property
    def ftr(self):
        return self.vertices[7]

    @property
    def near_plane(self):
        return self.planes[0]

    @property
    def top_plane(self):
        return self.planes[1]

    @property
    def left_plane(self):
        return self.planes[2]

    @property
    def bottom_plane(self):
        return self.planes[3]

    @property
    def right_plane(self):
        return self.planes[4]

    @property
    def far_plane(self):
        return self.planes[5]

    @property
    def lookat(self):
        return self._lookat

    def transform(self, tx):
        super(CameraFrustum, self).transform(tx)
        self._lookat = tx.dot(self._lookat)


class AABB(Prism):
    """
    An axis aligned bounding box defined by 2 points in R^3
    """

    def __init__(self, p1, p2):
        p1 = utils.make_projective_point(p1)
        p2 = utils.make_projective_point(p2)

        size = p2 - p1
        size_x = np.array((size[0], 0, 0, 0))
        size_y = np.array((0, size[1], 0, 0))
        size_xy = size_x + size_y

        v1, v2, v3, v4 = p1, p1 + size_x, p1 + size_xy, p1 + size_y
        super(AABB, self).__init__(size[2], v1, v2, v3, v4)

    def __str__(self):
        return str([str(self._vertices[0]), str(self._vertices[7])])

    @property
    def width(self):
        return self.size[0]

    @property
    def height(self):
        return self.size[1]

    @property
    def depth(self):
        return self.size[2]

    @property
    def size(self):
        return np.abs(self.vertices[6] - self.vertices[0])

    @property
    def center(self):
        half_sz = (self.vertices[6] - self.vertices[0]) / 2.0
        return self.vertices[0] + half_sz

    def transform(self, tx):
        tx = utils.verify_matrix_shape(tx, 4, 4)

        for i in range(len(self._vertices)):
            self._vertices[i] = tx.dot(self._vertices[i])


@multimethod(Polyhedron, Polyhedron)
def intersects(obj1, obj2):
    """
    Test if two convex solids intersect.

    :param obj1: The first solid which is tested for intersection
    :param obj2: The second solid which is tested for intersection
    :return: Whether or not both solids intersect
    """
    # Check if all the prism's vertices lie on the outside of at least one of the frustum's planes
    # In that case, the two definitely do not intersect. Otherwise they may or may not intersect.
    for plane in obj2.planes:
        out_pts, in_pts = 0, 0
        for point in obj1.vertices:
            if out_pts != 0 and in_pts != 0:
                break
            if plane.signed_distance(point) > 0:
                out_pts += 1
            else:
                in_pts += 1
        if in_pts == 0:
            return False

    # Check if all the frustum's verices lie on the outside of at least one of the frustum's planes.
    # In that case the two definitely do not intersect. Otherwise, they intersect.
    for plane in obj1.planes:
        out_pts, in_pts = 0, 0
        for v in obj2.vertices:
            if out_pts != 0 and in_pts != 0:
                break
            if plane.signed_distance(v) > 0:
                out_pts += 1
            else:
                in_pts += 1
        if in_pts == 0:
            return False

    return True
