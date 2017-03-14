import numpy as np
import tiling
import scene_parsing
from visualiser import gl_geometry
from visualiser.viewer import Viewer
from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *


# group = tiling.PlanarReflectionGroup(560, (0, 0, 0), (560, 0, 0), (560, 0, 560), (0, 0, 560))
# base_kernel = tiling.SquareKernel(2, (0, 0), group)
group = tiling.PlanarReflectionGroup(560, (280, 0, 0), (560, 0, 0), (280, 0, 560 * np.sqrt(3.0) / 2.0))
base_kernel = tiling.HexKernel(1, (0, 0), group)
frustum = scene_parsing.make_frustum("test_xml/camera.xml")
kt = tiling.KernelTiling(base_kernel, frustum, 1)
geometry_display_list = None


def init(viewer):
    viewer.camera_controller.set_dist_from_center(5000)
    viewer.camera_controller.set_zoom_speed(75)

    glutInit(len(sys.argv), sys.argv)
    glClearColor(0, 0, 0, 1.0)
    glEnable(GL_MULTISAMPLE)
    glEnable(GL_DEPTH_TEST)

    glMatrixMode(GL_PROJECTION)
    gluPerspective(60, float(viewer.width()) / float(viewer.height()), 0.5, np.linalg.norm(frustum.far_plane.position)*5)

    glMatrixMode(GL_MODELVIEW)
    glLoadIdentity()

    glEnable(GL_LIGHTING)
    glEnable(GL_LIGHT0)
    glLightfv(GL_LIGHT0, GL_POSITION, (0, 1, 1, 0))
    glLightfv(GL_LIGHT0, GL_AMBIENT, (0.0, 0.0, 0.0, 1.0))
    glLightfv(GL_LIGHT0, GL_DIFFUSE, (0.3, 0.3, 0.3, 1.0))

    print "Generating geometry..."

    tilemap = dict()
    for k in kt.visible_kernels:
        for index, f, p in k.translational_fundamental_domains:
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

    print "Done generating geometry!"


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

gl_viewer = Viewer()
gl_viewer.set_init_function(init)
gl_viewer.set_draw_function(draw)
gl_viewer.run()