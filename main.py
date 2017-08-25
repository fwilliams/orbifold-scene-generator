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
from geometry import shapes, utils

def init(viewer):
    viewer.camera_controller.set_dist_from_center(5000)
    viewer.camera_controller.set_zoom_speed(750)

    glutInit(1, [sys.argv[0]])
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

    print("Loading geometry into viewer...")

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
    glPopAttrib(GL_ENABLE_BIT)
    glEndList()

    global normal_display_list
    normal_display_list = glGenLists(1)
    glNewList(normal_display_list, GL_COMPILE)
    glPushAttrib(GL_ENABLE_BIT)

    for t in tilemap.values():
        glColor3f(1, 1, 1)
        gl_geometry.draw_prism_normals(t[0], 100.0)

    glPopAttrib(GL_ENABLE_BIT)
    glEndList()

    global wire_display_list
    wire_display_list = glGenLists(1)
    glNewList(wire_display_list, GL_COMPILE)
    glPushAttrib(GL_ENABLE_BIT)

    glColor3f(1, 1, 1)
    for t in tilemap.values():
        gl_geometry.draw_wire_prism(t[0])


    glPopAttrib(GL_ENABLE_BIT)
    glEndList()

    global sample_display_list
    sample_display_list = glGenLists(1)
    glNewList(sample_display_list, GL_COMPILE)
    glPushAttrib(GL_ENABLE_BIT)

    for t in tilemap.values():
        # tri = shapes.Triangle([230, 100, 50], [330, 100, 50], [280, 300, 150])
        tri = shapes.Triangle([10, 100, 50], [110, 100, 50], [55, 300, 150])
        tri.transform(t[1])
        gl_geometry.draw_triangle(tri)

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

    if gl_viewer.flag_normals:
        glCallList(normal_display_list)

    if gl_viewer.flag_wires:
        glCallList(wire_display_list)

    if gl_viewer.flag_Samples:
        glCallList(sample_display_list)

    if gl_viewer.flag_axes:
        gl_geometry.draw_axes((10000, 10000, 10000))

    glPopAttrib(GL_ENABLE_BIT)

    glFinish()


def resize(viewer):
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    gluPerspective(60, float(viewer.width()) / float(viewer.height()),
                   0.5, np.linalg.norm(frustum.far_plane.position)*5)
    glMatrixMode(GL_MODELVIEW)

argparser = argparse.ArgumentParser()
argparser.add_argument("filename", help="The name of the scene file to render.", type=str)
argparser.add_argument("type", help="The type of the scene. Must be one of xx x2222 x442 x642 x333 or xN, "
                                    "where N is a positive integer.")
argparser.add_argument("radius", help="The kernel radius", type=int)
argparser.add_argument("overlap", help="The amout that adjacent kernels overlap", type=int)
argparser.add_argument("scale", help="args.scale factor for the scene.", type=float, default=560.0)
argparser.add_argument("-v", "--visualize", help="Visualize the kernels we are going to draw", action="store_true")
argparser.add_argument("-b", "--bidir", help="Use bidirectional path tracing instead of path tracing for "
                                                 "incompleteness images", action="store_true")
argparser.add_argument("-i", "--inc", help="output incompleteness scenes", action="store_true")
args = argparser.parse_args()

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

output_dir = "./output_%s_%s" % (os.path.basename(args.filename), str(int(time.time())))
output_dir = os.path.realpath(output_dir)
os.mkdir(output_dir)

split_paths = os.path.split(args.filename)
print("Generating fds for scenes %s..." %split_paths[1])
filename = os.path.splitext(split_paths[1])

frustum = scene_parsing.make_frustum(args.filename)
kt = tiling.KernelTiling(base_kernel, frustum, args.overlap)
geometry_display_list = None

print("Generating scene data...")
i = 0
for kernel in kt.visible_kernels:

    print("Generating the scene data for kernel %d ..." % i)
    with etree.xmlfile(os.path.join(output_dir, "%s_img_%d_clr.xml" % (filename[0], i)), encoding='utf-8') as xf:
        sp.gen_scene_xml_incremental(args.filename, list(kernel.fundamental_domain_transforms), xf)

    if args.inc:
        print("Generating the inc scene data for kernel %d ..." % i)
        with etree.xmlfile(os.path.join(output_dir, "%s_inc_img_%d_clr.xml" % (filename[0], i)), encoding='utf-8') as xf:
            sp.gen_incompleteness_xml_incremental(args.filename, list(kernel.fundamental_domain_transforms), xf,use_bidir=args.bidir, )

    i += 1

print("Saved scene data to %s" % output_dir)

if args.visualize:
    gl_viewer = Viewer()
    gl_viewer.set_init_function(init)
    gl_viewer.set_draw_function(draw)
    gl_viewer.set_resize_function(resize)
    gl_viewer.run()