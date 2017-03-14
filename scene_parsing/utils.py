from lxml import etree
import numpy as np


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