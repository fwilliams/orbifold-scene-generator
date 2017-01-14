from lxml import etree
from geometry.shapes import Frustum

MODES = ("mirror", "dup", "inc", "depth")


def gen_transform(x, y):
    transform = etree.Element("transform", name="toWorld")
    if abs(x) % 2 != 0:
        etree.SubElement(transform, "translate", x="0", y="0", z="%d" % (-560/2))
        etree.SubElement(transform, "scale", x="1", y="1", z="-1")
        etree.SubElement(transform, "translate", x="0", y="0", z="%d" % (560/2))

    if abs(y) % 2 != 0:
        etree.SubElement(transform, "translate", x="%d" % (-560/2), y="0", z="0")
        etree.SubElement(transform, "scale", x="-1", y="1", z="1")
        etree.SubElement(transform, "translate", x="%d" % (560/2), y="0", z="0")

    etree.SubElement(transform, "translate", x="%d" % (560*y), y="0", z="%d" % (560*x))

    return transform


def gen_scene_instance(grpname):
    instance = etree.Element("shape", type="instance")
    etree.SubElement(instance, "ref", id=grpname)
    return instance


def gen_light():
    light = etree.Element("shape", type="obj")
    etree.SubElement(light, "string", name="filename", value="meshes/cbox_luminaire.obj")
    etree.SubElement(light, "ref", id="light")

    emitter = etree.SubElement(light, "emitter", type="area")
    etree.SubElement(emitter, "spectrum", name="radiance", value="400:18.4, 500:18.4, 600:18.4, 700:18.4")

    return light


def gen_obj_shape(material, filename):
    obj = etree.Element("shape", type="obj")
    etree.SubElement(obj, "string", name="filename", value=filename)
    if etree.iselement(material):
        obj.append(material)
    else:
        assert isinstance(material, str), "material parameter to gen_obj_shape should be a string or an XML tree"
        etree.SubElement(obj, "ref", id=material)

    return obj


def append_includes(root, mode):
    if mode == "depth":
        etree.SubElement(root, "integrator", type="depth")
    else:
        etree.SubElement(root, "integrator", type="opath")
    etree.SubElement(root, "include", filename="camera.xml")
    etree.SubElement(root, "include", filename="defs.xml")


def make_img_xx(mode, num_x):
    assert mode in MODES, "mode parameter must be one of %s. Got %s" % (str(MODES), mode)

    root = etree.Element("scene", version="0.5.0")
    append_includes(root, mode)

    sceneinstance = etree.SubElement(root, "shape", type="shapegroup", id="wallsnmirrors")
    sceneinstance.append(gen_obj_shape("gray", "meshes/cbox_greenwall.obj"))
    sceneinstance.append(gen_obj_shape("gray", "meshes/cbox_redwall.obj"))
    if mode == "mirror":
        sceneinstance.append(gen_obj_shape("mirror", "meshes/cbox_front.obj"))
        sceneinstance.append(gen_obj_shape("mirror", "meshes/cbox_back.obj"))

    for i in range(-num_x, num_x+1):
        instance = gen_scene_instance("scene")
        instance_walls = gen_scene_instance("wallsnmirrors")
        light = gen_light()

        instance.append(gen_transform(0, i))
        instance_walls.append(gen_transform(0, i))
        light.append(gen_transform(0, i))

        root.append(instance)
        root.append(instance_walls)
        if mode != "inc":
            root.append(light)

    with open("/home/francis/mitsuba/scenes/cbox/x2222_gen.xml", 'w') as f:
        f.write(etree.tostring(root, pretty_print=True))

    print(etree.tostring(root, pretty_print=True))


def make_img_x2222(mode, num_x, num_y):
    assert mode in MODES, "mode parameter must be one of %s. Got %s" % (str(MODES), mode)

    root = etree.Element("scene", version="0.5.0")
    append_includes(root, mode)

    sceneinstance = etree.SubElement(root, "shape", type="shapegroup", id="wallsnmirrors")

    if mode == "mirror":
        num_x = 0
        num_y = 0
        sceneinstance.append(gen_obj_shape("mirror", "meshes/cbox_greenwall.obj"))
        sceneinstance.append(gen_obj_shape("mirror", "meshes/cbox_redwall.obj"))
        sceneinstance.append(gen_obj_shape("mirror", "meshes/cbox_front.obj"))
        sceneinstance.append(gen_obj_shape("mirror", "meshes/cbox_back.obj"))
    elif mode == "inc":
        emitter = etree.SubElement(root, "emitter", type="constant")
        etree.SubElement(emitter, "rgb", name="radiance", value="#ffffff")

    for i in range(-num_x, num_x+1):
        for j in range(-num_y, num_y+1):
            instance = gen_scene_instance("scene")
            instance_walls = gen_scene_instance("wallsnmirrors")
            light = gen_light()

            instance.append(gen_transform(i, j))
            instance_walls.append(gen_transform(i, j))
            light.append(gen_transform(i, j))

            root.append(instance)
            root.append(instance_walls)

            if mode != "inc":
                root.append(light)

    with open("/home/francis/mitsuba/scenes/cbox/x2222_gen.xml", 'w') as f:
        f.write(etree.tostring(root, pretty_print=True))

    print(etree.tostring(root, pretty_print=True))


# make_img_x2222("mirror", 3, 3)
Frustum.from_scene_xml("./camera.xml")
