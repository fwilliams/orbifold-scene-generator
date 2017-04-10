import argparse
import numpy as np
from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *
from lxml import etree
import scene_parsing
import scene_parsing as sp
import tiling
import time
from visualiser import gl_geometry
from visualiser.viewer import Viewer


def init(viewer):
    viewer.camera_controller.set_dist_from_center(5000)
    viewer.camera_controller.set_zoom_speed(750)

    glutInit(len(sys.argv), sys.argv)
    glClearColor(0, 0, 0, 1.0)
    glEnable(GL_MULTISAMPLE)
    glEnable(GL_DEPTH_TEST)

    glMatrixMode(GL_PROJECTION)
    gluPerspective(60, float(viewer.width()) / float(viewer.height()),
                   0.5, np.linalg.norm(frustum.far_plane.position)*5)

    glMatrixMode(GL_MODELVIEW)
    glLoadIdentity()

    glEnable(GL_LIGHTING)
    glEnable(GL_LIGHT0)
    glLightfv(GL_LIGHT0, GL_POSITION, (0, 1, 1, 0))
    glLightfv(GL_LIGHT0, GL_AMBIENT, (0.0, 0.0, 0.0, 1.0))
    glLightfv(GL_LIGHT0, GL_DIFFUSE, (0.3, 0.3, 0.3, 1.0))

    print("Generating geometry...")

    tilemap = dict()
    for k in kt.visible_kernels:
        for index, f, p in k.fundamental_domains:
            if str(index) in tilemap:
                tilemap[str(index)][2] += 1
            else:
                tilemap[str(index)] = [p, f, 1]

    global geometry_display_list
    geometry_display_list = glGenLists(1)
    glNewList(geometry_display_list, GL_COMPILE)
    glPushAttrib(GL_ENABLE_BIT)
    glEnable(GL_LIGHTING)
    for t in tilemap.values():
        if t[2] == 1:
            color = np.array((0.5, 0.5, 0.5))
        elif t[2] == 2:
            color = np.array((0.9, 0.9, 0.0))
        elif t[2] == 3:
            color = np.array((0.9, 0.7, 0.0))
        else:
            color = np.array((0.9, 0.1, 0.1))
        glMaterialfv(GL_FRONT_AND_BACK, GL_DIFFUSE, (0.7, 0.7, 0.7))
        glMaterialfv(GL_FRONT_AND_BACK, GL_AMBIENT, color)
        gl_geometry.draw_solid_prism(t[0])

        glDisable(GL_DEPTH_TEST)
        glDisable(GL_LIGHTING)
        glColor3f(1, 1, 1)
        gl_geometry.draw_prism_normals(t[0], 100.0)
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_LIGHTING)

    glDisable(GL_DEPTH_TEST)
    glDisable(GL_LIGHTING)
    glColor3f(1, 1, 1)
    for t in tilemap.values():
        gl_geometry.draw_wire_prism(t[0])

    glPopAttrib(GL_ENABLE_BIT)
    glEndList()

    print("Done generating geometry!")


def draw(viewer):
    glMatrixMode(GL_MODELVIEW)
    glLoadIdentity()

    rm = viewer.camera_controller.camera_matrix
    glMultMatrixf(rm.transpose())

    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

    viewer.skybox.draw(viewer.camera_controller.camera_rotation.transpose(),
                       np.linalg.norm(frustum.far_plane.position)*5)

    glCallList(geometry_display_list)

    glPushAttrib(GL_ENABLE_BIT)
    glDisable(GL_DEPTH_TEST)
    glDisable(GL_LIGHTING)
    gl_geometry.draw_axes((10000, 10000, 10000))
    glColor3f(1, 1, 1)
    gl_geometry.draw_wire_prism(frustum)
    glPopAttrib(GL_ENABLE_BIT)

    glFinish()


def resize(viewer):
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    gluPerspective(60, float(viewer.width()) / float(viewer.height()),
                   0.5, np.linalg.norm(frustum.far_plane.position)*5)
    glMatrixMode(GL_MODELVIEW)

argparser = argparse.ArgumentParser()
argparser.add_argument("file", "The name of the scene file to render.", type=str)
argparser.add_argument("-t", "--type", help="The type of the scene. Must be one of xx x2222 x442 x642 x333 or xN, "
                                            "where N is a positive integer.")
argparser.add_argument("-r", "--radius", help="The kernel radius", type=int)
argparser.add_argument("-s", "--args.scale", help="args.scale factor for the scene.", type=float)
argparser.add_argument("-o", "--overlap", help="The amout that adjacent kernels overlap")
argparser.add_argument("-v", "--visualize", help="Visualize the kernels we are going to draw", action="store_true")
args = argparser.parse_args()


if args.type == "xx":
    group = tiling.FriezeReflectionGroup(args.scale, (0, 1, 0),
                                        (0, 0.5*args.scale, 0), (0, 0.5*args.scale, args.scale))
    base_kernel = tiling.LineKernel(args.radius, 0, group)
elif args.type == "2222":
    # *2222
    group = tiling.PlanarReflectionGroup(args.scale, (0, 0, 0),
                                         (args.scale, 0, 0), (args.scale, 0, args.scale), (0, 0, args.scale))
    base_kernel = tiling.SquareKernel(args.radius, (0, 0), group)
elif args.type == "442":
    # *2222
    group = tiling.PlanarReflectionGroup(args.scale, (0, 0, 0), (args.scale, 0, 0), (args.scale, 0, args.scale))
    base_kernel = tiling.SquareKernel(args.radius, (0, 0), group)
elif args.type == "632":
    # *632
    group = tiling.PlanarReflectionGroup(args.scale, (0, 0, 0),
                                         (0.5*args.scale, 0, 0), (0, 0, args.scale * np.sqrt(3.0) / 2.0))
    base_kernel = tiling.HexKernel(args.radius, (0, 0), group)
elif args.type == "333":
    # *333
    group = tiling.PlanarReflectionGroup(args.scale, (0, 0, 0),
                                         (args.scale, 0, 0), (0.5*args.scale, 0, args.scale * np.sqrt(3.0) / 2.0))
    base_kernel = tiling.HexKernel(args.radius, (0, 0), group)
else:
    if args.type.startswith("x"):
        try:
            order = int(args.type[1:])
        except ValueError:
            assert False, "Invalid scene type, %s. Must be one of xx x2222 x442 x642 x333 or xN, " \
                          "where N is a positive integer." % args.type
        # TODO: Dihedral case
        group = None
        base_kernel = None
    else:
        assert False, "Invalid scene type, %s. Must be one of xx x2222 x442 x642 x333 or xN, " \
                      "where N is a positive integer." % args.type

output_dir = "./output_%s_%s" % (args.file, str(int(time.time())))
output_dir = os.path.realpath(output_dir)
os.mkdir(output_dir)

frustum = scene_parsing.make_frustum(args.file)
kt = tiling.KernelTiling(base_kernel, frustum, args.overlap)
geometry_display_list = None

i = 0
for kernel in kt.visible_kernels:
    scene_doc = sp.gen_scene_xml(args.file, list(base_kernel.fundamental_domain_transforms))
    inc_doc = sp.gen_incompleteness_xml(args.file, list(base_kernel.fundamental_domain_transforms))
    depth_doc = sp.gen_depth_xml_from_scene(scene_doc)

    with open(os.path.join(output_dir, "_%d.clr" % i), "w+") as f:
        f.write(etree.tostring(scene_doc, pretty_print=True))
    with open(os.path.join(output_dir, "_%d" % i), "w+") as f:
        f.write(etree.tostring(inc_doc, pretty_print=True))
    with open(os.path.join(output_dir, "_%d" % i), "w+") as f:
        f.write(etree.tostring(depth_doc, pretty_print=True))
    i += 1

if args.visualize:
    gl_viewer = Viewer()
    gl_viewer.set_init_function(init)
    gl_viewer.set_draw_function(draw)
    gl_viewer.set_resize_function(resize)
    gl_viewer.run()