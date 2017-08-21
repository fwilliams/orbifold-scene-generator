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

    glutInit(1, [sys.argv[0]])
    glClearColor(0, 0, 0, 1.0)
    glEnable(GL_MULTISAMPLE)
    glEnable(GL_DEPTH_TEST)

    glMatrixMode(GL_PROJECTION)

    # the viewing frustum will be changed in the resize function which will also be called at start
    # gluPerspective(60, float(viewer.width()) / float(viewer.height()),0.5, np.linalg.norm(frustum.far_plane.position)*5)

    glMatrixMode(GL_MODELVIEW)
    glLoadIdentity()

    glEnable(GL_LIGHTING)
    glEnable(GL_LIGHT0)
    glLightfv(GL_LIGHT0, GL_POSITION, (0, 1, 1, 0))
    glLightfv(GL_LIGHT0, GL_AMBIENT, (0.0, 0.0, 0.0, 1.0))
    glLightfv(GL_LIGHT0, GL_DIFFUSE, (0.3, 0.3, 0.3, 1.0))

    print("Loading geometry into viewer...")

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
            color = np.array((0.0, 0.7, 0.0))
        else:
            color = np.array((0.9, 0.1, 0.1))
        glMaterialfv(GL_FRONT_AND_BACK, GL_DIFFUSE, (0.7, 0.7, 0.7))
        glMaterialfv(GL_FRONT_AND_BACK, GL_AMBIENT, color)
        gl_geometry.draw_solid_prism(t[0])


    glPopAttrib(GL_ENABLE_BIT)
    glEndList()

    print("Done loading geometry!")


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

    for t in tilemap.values():
        if gl_viewer.flag_normals:
            glColor3f(1, 1, 1)
            gl_geometry.draw_prism_normals(t[0], 100.0)

        if gl_viewer.flag_wires:
            glColor3f(1, 1, 1)
            gl_geometry.draw_wire_prism(t[0])

    if gl_viewer.flag_axes:
        gl_geometry.draw_axes((10000, 10000, 10000))

    glColor3f(1, 1, 1)
    gl_geometry.draw_wire_prism(frustum)
    glPopAttrib(GL_ENABLE_BIT)

    glFinish()


def resize(viewer):
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()

    gluPerspective(60, float(viewer.width()) / float(viewer.height()), 0.5, np.linalg.norm(frustum.far_plane.position)*50)

    glMatrixMode(GL_MODELVIEW)

argparser = argparse.ArgumentParser()
# argparser.add_argument("filename", help="The name of the scene file to render.", type=str)
# argparser.add_argument("type", help="The type of the scene. Must be one of xx x2222 x442 x642 x333 or xN, "
#                                     "where N is a positive integer.")
# argparser.add_argument("radius", help="The kernel radius", type=int)
# argparser.add_argument("overlap", help="The amout that adjacent kernels overlap", type=int)
# argparser.add_argument("scale", help="args.scale factor for the scene.", type=float, default=560.0)
# argparser.add_argument("-v", "--visualize", help="Visualize the kernels we are going to draw", action="store_true")
# argparser.add_argument("-b", "--bidir", help="Use bidirectional path tracing instead of path tracing for "
#                                                  "incompleteness images", action="store_true")
# argparser.add_argument("ceiling", help="The flag used to generate ceiling reflections", type = bool, default = False)
# argparser.add_argument("floor", help="The flag used to generate floor reflections", type = bool, default = False)
args = argparser.parse_args()

args.type = "xx"
args.filename = "./example_xml/xxx.xml"
args.radius = 2
args.overlap = 0
args.scale = 560
args.bidir = False;
args.visualize = True;


if args.type == "xx":
    group = tiling.FriezeReflectionGroup(args.scale, (0, 1, 0),
                                         (0, 0.5*args.scale, 0), (0, 0.5*args.scale, args.scale))
    base_kernel = tiling.LineKernel(args.radius, 0, group)
elif args.type == "x2222":
    # *2222
    group = tiling.PlanarReflectionGroup(args.scale, (0, 0, 0),
                                         (args.scale, 0, 0), (args.scale, 0, args.scale), (0, 0, args.scale))
    base_kernel = tiling.SquareKernel(args.radius, (0, 0), group)
elif args.type == "x442":
    # *2222
    group = tiling.PlanarReflectionGroup(args.scale, (0, 0, 0), (args.scale, 0, 0), (args.scale, 0, args.scale))
    base_kernel = tiling.SquareKernel(args.radius, (0, 0), group)
elif args.type == "x632":
    # *632
    group = tiling.PlanarReflectionGroup(args.scale, (0, 0, 0),
                                         (0.5*args.scale, 0, 0), (0, 0, args.scale * np.sqrt(3.0) / 2.0))
    base_kernel = tiling.HexKernel(args.radius, (0, 0), group)
elif args.type == "x333":
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
        raise NotImplementedError("Botong, if you see this, message me. I still need to write a "
                                  "one-liner to make this work")
    else:
        assert False, "Invalid scene type, %s. Must be one of xx x2222 x442 x642 x333 or xN, " \
                      "where N is a positive integer." % args.type


frustum = scene_parsing.make_frustum(args.filename)
kt = tiling.KernelTiling(base_kernel, frustum, args.overlap)
geometry_display_list = None

# output_dir = "./output_%s_%s" % (os.path.basename(args.filename), str(int(time.time())))
# output_dir = os.path.realpath(output_dir)
# os.mkdir(output_dir)

# print("Generating scene data...")
# i = 0
# for kernel in kt.visible_kernels:
#     scene_doc = sp.gen_scene_xml(args.filename, list(kernel.fundamental_domain_transforms))
#     inc_doc = sp.gen_incompleteness_xml(args.filename, list(kernel.fundamental_domain_transforms), use_bidir=args.bidir)
#
#     with open(os.path.join(output_dir, "img_%d_clr.xml" % i), "w+") as f:
#         f.write(etree.tostring(scene_doc, pretty_print=True))
#     with open(os.path.join(output_dir, "inc_img_%d_clr.xml" % i), "w+") as f:
#         f.write(etree.tostring(inc_doc, pretty_print=True))
#     i += 1
#
# print("Saved scene data to %s" % output_dir)

if args.visualize:
    gl_viewer = Viewer()

    tilemap = dict()
    gl_viewer.set_init_function(init)
    gl_viewer.set_draw_function(draw)
    gl_viewer.set_resize_function(resize)
    gl_viewer.run()