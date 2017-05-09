from __future__ import print_function
from lxml import etree
from geometry import shapes
import numpy as np
import utils
import copy


def gen_incompleteness_xml(xml_filename, tx_list, use_bidir=False):
    scene_doc = etree.parse(xml_filename)
    root = scene_doc.getroot()

    shape_nodes = root.findall("shape")
    shape_nodes_copy = copy.deepcopy(shape_nodes)

    for s in shape_nodes:
        s.getparent().remove(s)

    for s in shape_nodes_copy:
        if s.get("type") == "shapegroup":
            root.append(s)
            continue

        # if is emitter, strip the light source
        if utils.find_unique(s, "emitter", allow_zero=True) is not None:
            s.remove(utils.find_unique(s, "emitter"))

        transform = copy.deepcopy(utils.find_unique(s, "transform", allow_zero=True))
        for tx in tx_list:
            new_shape = copy.deepcopy(s)
            if transform is not None:
                new_transform = copy.deepcopy(transform)
                new_shape.remove(utils.find_unique(new_shape, "transform", allow_zero=True))
            else:
                new_transform = etree.Element("transform", name="toWorld")

            tx_value = ""
            for row in tx:
                for val in row:
                    tx_value += str(val) + " "
            new_transform.append(etree.Element("matrix", value=tx_value))
            new_shape.append(new_transform)
            root.append(new_shape)

    # print(etree.tostring(root, pretty_print=True))
    emitter = etree.SubElement(root, "emitter", type="constant")
    etree.SubElement(emitter, "rgb", name="radiance", value="#ffffff")

    integrator = root.find("integrator")
    root.remove(integrator)
    mc_integrator = etree.Element("integrator", type="multichannel")
    etree.SubElement(mc_integrator, "integrator", type="bdpt" if use_bidir else "path")
    field_integrator = etree.SubElement(mc_integrator, "integrator", type="field")
    etree.SubElement(field_integrator, "string", name="field", value="distance")
    etree.SubElement(field_integrator, "float", name="undefined", value="3.40282e+38")
    root.append(mc_integrator)

    sensor = root.find("sensor")
    film = sensor.find("film")
    component_format = root.xpath("//string[@name='componentFormat']")
    if component_format:
        film.remove(component_format)
    etree.SubElement(film, "string", name="componentFormat", value="float32")
    etree.SubElement(film, "string", name="pixelFormat", value="rgb, luminance")
    etree.SubElement(film, "string", name="channelNames", value="color, distance")
    etree.SubElement(integrator, "boolean", name="incompleteness mode", value="true")
    return root


def gen_scene_xml(xml_filename, tx_list):
    scene_doc = etree.parse(xml_filename)
    root = scene_doc.getroot()

    shape_nodes = root.findall("shape")
    shape_nodes.extend(root.findall("emitter"))
    shape_nodes_copy = copy.deepcopy(shape_nodes)

    for s in shape_nodes:
        s.getparent().remove(s)

    for s in shape_nodes_copy:
        if s.get("type") == "shapegroup":
            root.append(s)
            continue

        transform = copy.deepcopy(utils.find_unique(s, "transform", allow_zero=True))
        for tx in tx_list:
            new_shape = copy.deepcopy(s)
            if transform is not None:
                new_transform = copy.deepcopy(transform)
                new_shape.remove(utils.find_unique(new_shape, "transform", allow_zero=True))
            else:
                new_transform = etree.Element("transform", name="toWorld")

            tx_value = ""
            for row in tx:
                for val in row:
                    tx_value += str(val) + " "
            new_transform.append(etree.Element("matrix", value=tx_value))
            new_shape.append(new_transform)
            root.append(new_shape)

    sensor = root.find("sensor")
    film = sensor.find("film")
    component_format = root.xpath("//string[@name='componentFormat']")
    if component_format:
        film.remove(component_format)
    etree.SubElement(film, "string", name="componentFormat", value="float32")

    # print(etree.tostring(root, pretty_print=True))
    return root


def make_frustum(xml_filename, __dbg=False):
    def _dbg_print(*args):
        if __dbg:
            for i in args:
                print('{}'.format(i), end=' ')
            print("")

    scene_doc = etree.parse(xml_filename)
    root = scene_doc.getroot()
    camera = utils.find_unique(root, 'sensor')
    assert camera.get("type") == "perspective", "Frustum.from_scene_xml() got camera of type %s. " \
                                                "We only support perspective cameras." % camera.get("type")

    far_clip = float(utils.find_unique(camera, "float[@name='farClip']").get("value"))
    near_clip = float(utils.find_unique(camera, "float[@name='nearClip']").get("value"))
    fov = float(utils.find_unique(camera, "float[@name='fov']").get("value"))
    fov_axis = utils.find_unique(camera, "string[@name='fovAxis']").get("value")
    assert fov_axis in ('x', 'y'), "Fov axis must be one of x or y. " \
                                   "We don't support 'diagonal', 'smaller' or 'larger' modes."

    frustum_transform = np.identity(4)

    transform = utils.find_unique(camera, "transform", allow_zero=True)
    if transform is not None:
        for tx in transform:
            if tx.tag == "translate":
                tx_matrix = np.identity(4)
                tx_matrix[:, 3] = utils.parse_xyz(tx, 0)
                frustum_transform = tx_matrix * frustum_transform
            elif tx.tag == "rotate":
                pass
            elif tx.tag == "scale":
                tx_matrix = np.identity(4)
                np.fill_diagonal(tx_matrix, utils.parse_xyz(tx, 1))
                frustum_transform = tx_matrix * frustum_transform
            elif tx.tag == "matrix":
                frustum_transform = utils.parse_4x4_matrix(tx) * frustum_transform
            elif tx.tag == "lookat":
                target = utils.parse_vector3(tx.get("target"))
                origin = utils.parse_vector3(tx.get("origin"))
                up = utils.parse_vector3(tx.get("up"))

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

                _dbg_print("lookat origin =", str(origin))
                _dbg_print("lookat target =", str(target))
                _dbg_print("lookat up =", str(up))
                _dbg_print("resulting in matrix = ", str(direction_tx))
                _dbg_print(x, y, z)

                frustum_transform = frustum_transform.dot(direction_tx)
            else:
                raise ValueError("Got wrong tag type %s as child of <transform>" % tx.tag)

    film = utils.find_unique(camera, "film")
    width = utils.find_unique(film, "integer[@name='width']").get("value")
    height = utils.find_unique(film, "integer[@name='height']").get("value")

    _dbg_print("farClip =", far_clip)
    _dbg_print("nearClip =", near_clip)
    _dbg_print("fov =", fov)
    _dbg_print("fovAxis =", fov_axis)
    _dbg_print("image dims =", (width, height))

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
    else:
        assert False, "fov_axis must be x or y"

    _dbg_print("Frustum(left=%f, right=%f, bottom=%f, top=%f, nearClip=%f, farClip=%f)"
               % (left, right, bottom, top, near_clip, far_clip))

    frustum = shapes.CameraFrustum(left, right, bottom, top, near_clip, far_clip)
    frustum.transform(frustum_transform)
    return frustum
