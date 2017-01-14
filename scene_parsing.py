from lxml import etree
from geometry import shapes

import numpy as np


def make_frustum(xml_filename):
    def find_unique(element, xpath_str, allow_zero=False):
        elems = element.findall(xpath_str)
        if not allow_zero:
            assert len(elems) == 1, "find_unique(%s) in document %s expected to find only one element. Got %d" \
                                    % (str(xpath_str), etree.tostring(element), len(elems))
        else:
            if len(elems) == 0:
                return None
            else:
                assert len(elems) == 1, "find_unique(%s) in document %s expected to find one or zero elements. " \
                                        "Got %d" % (str(xpath_str), etree.tostring(element), len(elems))
        return elems[0]

    def parse_vector3(vstr):
        v = [float(i) for i in vstr.replace(',', ' ').split()]
        assert len(v) == 3, "parse_vector3 expected length 3 vector but got length %d" % len(v)
        return np.array(v)

    def parse_xyz(element, default_value):
        _x = float(element.get("x")) if element.get("x") else default_value
        _y = float(element.get("y")) if element.get("y") else default_value
        _z = float(element.get("z")) if element.get("z") else default_value
        return np.array([_x, _y, _z, 1])

    def parse_4x4_matrix(element):
        vals = [float(v) for v in element.get("value").split()]
        assert len(vals) == 16, "Got wrong number of elements in matrix transformation. " \
                                "Expected 16 but got %d" % len(vals)

        mat = np.zeros((4, 4))
        for col in range(4):
            for row in range(4):
                mat[row, col] = vals[col * 4 + row]

        return mat

    scene_doc = etree.parse(xml_filename)
    root = scene_doc.getroot()
    camera = find_unique(root, 'sensor')
    assert camera.get("type") == "perspective", "Frustum.from_scene_xml() got camera of type %s. " \
                                                "We only support perspective cameras." % camera.get("type")

    far_clip = float(find_unique(camera, "float[@name='farClip']").get("value"))
    near_clip = float(find_unique(camera, "float[@name='nearClip']").get("value"))
    fov = float(find_unique(camera, "float[@name='fov']").get("value"))
    fov_axis = find_unique(camera, "string[@name='fovAxis']").get("value")
    assert fov_axis in ('x', 'y'), "Fov axis must be one of x or y. " \
                                   "We don't support 'diagonal', 'smaller' or 'larger' modes."

    frustum_transform = np.identity(4)

    transform = find_unique(camera, "transform", allow_zero=True)
    if transform is not None:
        for tx in transform:
            if tx.tag == "translate":
                tx_matrix = np.identity(4)
                tx_matrix[:, 3] = parse_xyz(tx, 0)
                frustum_transform = tx_matrix * frustum_transform
            elif tx.tag == "rotate":
                pass
            elif tx.tag == "scale":
                tx_matrix = np.identity(4)
                np.fill_diagonal(tx_matrix, parse_xyz(tx, 1))
                frustum_transform = tx_matrix * frustum_transform
            elif tx.tag == "matrix":
                frustum_transform = parse_4x4_matrix(tx) * frustum_transform
            elif tx.tag == "lookat":
                target = parse_vector3(tx.get("target"))
                origin = parse_vector3(tx.get("origin"))
                up = parse_vector3(tx.get("up"))

                z = target - origin
                z /= np.linalg.norm(z)
                x = np.cross(up, z)
                x /= np.linalg.norm(x)
                y = np.cross(x, z)
                o = origin

                direction_tx = np.array([[x[0], y[0], z[0], o[0]],
                                         [x[1], y[1], z[1], o[1]],
                                         [x[2], y[2], z[2], o[2]],
                                         [0, 0, 0, 1]])

                # print "lookat origin =", str(origin)
                # print "lookat target =", str(target)
                # print "lookat up =", str(up)
                # print "resulting in matrix = ", str(direction_tx)
                # print x, y, z

                frustum_transform = frustum_transform.dot(direction_tx)
            else:
                raise ValueError("Got wrong tag type %s as child of <transform>" % tx.tag)

    film = find_unique(camera, "film")
    width = find_unique(film, "integer[@name='width']").get("value")
    height = find_unique(film, "integer[@name='height']").get("value")

    # print "farClip =", far_clip
    # print "nearClip =", near_clip
    # print "fov =", fov
    # print "fovAxis =", fov_axis
    # print "image dims =", (width, height)

    tan_half_fov = np.tan(np.radians(fov / 2.0))
    if fov_axis == "x":
        left = -near_clip * tan_half_fov
        right = -left
        aspect = float(width) / float(height)
        top = left / aspect
        bottom = -top
    elif fov_axis == "y":
        top = near_clip * tan_half_fov
        bottom = -top
        aspect = float(width) / float(height)
        left = top * aspect
        right = -left

    # print "Frustum(left=%f, right=%f, bottom=%f, top=%f, nearClip=%f, farClip=%f)" \
    #       % (left, right, bottom, top, near_clip, far_clip)

    frustum = shapes.CameraFrustum(left, right, bottom, top, near_clip, far_clip)
    frustum.transform(frustum_transform)
    return frustum
