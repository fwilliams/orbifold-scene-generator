import numpy as np

import shapes


EPSILON = 1e-12  # The margin of error used to test if vertices are co-planar


def translation_matrix(v):
    v = make_projective_point(v[:3])
    ret = np.identity(4)
    ret[:4, 3] = v
    return ret


def reflection_matrix(plane):
    """
    Return a 4x4 homogeneous reflection matrix associated with the reflection about the input plane
    :param plane: The plane of reflection
    :return: A 4x4 homogeneous reflection matrix
    """
    nx, ny, nz, _ = plane.normal
    nc = np.dot(plane.normal, plane.position)
    return np.array(((1-2*nx**2, -2*nx*ny, -2*nx*nz, 2*nx*nc),
                     (-2*ny*nx, 1-2*ny**2, -2*ny*nz, 2*ny*nc),
                     (-2*nz*nx, -2*nz*ny, 1-2*nz**2, 2*nz*nc),
                     (0, 0, 0, 1)))


def axis_angle_rotation_matrix(axis, angle):
    """
    Return the 4x4 homogeneous rotation matrix associated with counterclockwise rotation about
    the given axis by theta radians.
    :param axis: The axis of rotation.
    :param angle: The angle of rotation about the axis.
    :return: A 4x4 homogeneous rotation matrix.
    """
    axis = np.asarray(axis)
    if np.dot(axis, axis) <= 0:
        print axis
    axis = axis / np.sqrt(np.dot(axis, axis))
    a = np.cos(angle / 2.0)
    b, c, d = -axis * np.sin(angle / 2.0)
    aa, bb, cc, dd = a * a, b * b, c * c, d * d
    bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d
    return np.array(((aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac), 0),
                     (2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab), 0),
                     (2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc, 0),
                     (0, 0, 0, 1)))


def coplanar(*args):
    """
    Returns true if all the points specified in the arguments lie in the same plane. \
    The input must contain a minimum of 4 points.
    :param args: 4 or more 3d points or projective 4d points/
    :return: True if the input points are co-planar, and False otherwise
    """

    if len(args) < 3:
        raise ValueError("Can only test co-planarity of 3 or more point arguments. Got %s as input", str(args))
    elif len(args) == 3:
        return True

    vs = [make_projective_point(v) for v in args]

    plane = shapes.Plane(vs[0], vs[1], vs[2])

    for v in vs[2:]:
        d = plane.signed_distance(v)
        if np.abs(d) > EPSILON:
            return False

    return True


def verify_matrix_shape(v, *dims):
    if isinstance(v, np.matrix):
        raise ValueError("Do not use np.matrix. Use a 2D np.array instead")

    v = np.array(v)
    if v.shape != dims:
        raise ValueError("Shape mismatch. Expected %s, got %s" % (str(dims), str(v.shape)))

    return v


def normalize(vec):
    """
    Returns a unit length vector computed by dividing the input by its norm.

    :param vec: An input vector.
    :return: A unit length vector.
    """
    norm = np.linalg.norm(vec)
    if norm == 0:
        return ValueError("Cannot normalize the zero vector!")
    return vec / norm


def make_projective(v, dim=4, last_coord=1.0):
    """
    Takes an array like object or numpy array as input and outputs a dim-dimensional projective point as a numpy array.

    The input must be subscriptable with integers 0 through dim representing up to dim co-ordinates.

    If the input is not a point at infinity, the output will be the equivalent point whose last co-ordinate is 1.

    If the input is a point at infinity, the output will be the same as the input.

    If the input has dimension dim-1, the output will be the input with the value 'last_coord' appended as
    the dim'th co-ordinate.

    :param v: The input array-like object. Must have dimension dim-1 or dim.
    :param dim: The dimension of the desired output vector.
    :param last_coord: If the input has dimension dim-1, the output shall have this value as its last co-ordinate.
    :return: A normalized dim-d projective point or direction.
    """

    if isinstance(v, np.matrix):
        raise ValueError("Do not use np.matrix. Use a 2D np.array instead")

    v = np.array(v)
    if len(v.shape) != 1:
        raise ValueError("Expected numpy array with 1 dimensions. Got %d" % len(v.shape))
    if v.shape[0] == dim-1:
        res = [v[i] for i in range(dim-1)]
        res.append(last_coord)
        return np.array(res)
    elif v.shape[0] == dim:
        if v[dim-1] != 0.0:
            res = [v[i] / v[dim-1] for i in range(dim-1)]
            res.append(1.0)
            return np.array(res)
        else:
            return v


def make_projective_point(v, dim=4):
    """
    Takes an array like object or numpy array as input and outputs a dim-dimensional projective point as a numpy array.

    The input must be subscriptable with integers 0 through dim representing up to dim co-ordinates.
    If the input is a projective point, the output will be the equivalent point whose last co-ordinate is 1.

    :param v: The input array-like object. Must have dimension dim-1 or dim.
    :param dim: The dimension of the desired output vector.
    :return: A dim-dimensional projective point whose last co-ordinate is 1.
    """

    ret = make_projective(v, dim, 1.0)
    if ret[dim-1] == 0.0:
        raise ValueError("Attempted to convert homogeneous direction, %s, to a point" % v)

    return ret


def make_projective_vector(v, dim=4):
    """
    Takes an array like object or numpy array as input and outputs a dim-dimensional projective point as a numpy array.

    The input must be subscriptable with integers 0 through dim representing up to dim co-ordinates.
    If the input is a projective point, the output will be the equivalent point whose last co-ordinate is 1.

    :param v: The input array-like object. Must have dimension dim-1 or dim.
    :param dim: The dimension of the desired output vector.
    :return: A dim-dimensional projective point whose last co-ordinate is 1.
    """

    ret = make_projective(v, dim, 0.0)
    if ret[dim-1] != 0.0:
        raise ValueError("Attempted to convert projective point, %s, to a direction" % v)

    return ret
