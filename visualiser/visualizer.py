import sys
from PyQt5 import QtCore, QtOpenGL, QtWidgets

import numpy as np

import camera_control
import scene_parsing
import tiling
from skybox import Skybox
from visualiser import gl_geometry

try:
    from OpenGL.GL import *
    from OpenGL.GLU import *
    from OpenGL.GLUT import *

except ImportError:
    app = QtWidgets.QApplication(sys.argv)
    QtWidgets.QMessageBox.critical(None, "OpenGL Import Error", "PyOpenGL must be installed to run this example.")
    sys.exit(1)


class Window(QtWidgets.QWidget):
    def __init__(self):
        super(Window, self).__init__()

        self.gl_widget = GLWidget()

        main_layout = QtWidgets.QHBoxLayout()
        main_layout.addWidget(self.gl_widget)

        self.setLayout(main_layout)
        self.setWindowTitle("Hello GL")


class GLWidget(QtOpenGL.QGLWidget):
    def __init__(self, parent=None):
        fmt = QtOpenGL.QGLFormat()
        fmt.setSampleBuffers(True)
        fmt.setSamples(8)
        fmt.setDoubleBuffer(True)
        super(GLWidget, self).__init__(fmt, parent)

        self.setMouseTracking(True)
        self.elapsed = 0.0

        prg = tiling.PlanarReflectionGroup(560, (0, 0, 0), (0.5, 0, 0), (0.5, 0, 0.5*np.sqrt(3.0)))
        print prg.n
        print prg.translational_subgroup_basis
        fd = tiling.X442(560, (0, 0, 0), (0, 0, 560), (560, 0, 0))
        self.base_kernel = tiling.SquareKernel(1, (0, 0), fd)
        self.frustum = scene_parsing.make_frustum("test_xml/camera.xml")
        self.kt = tiling.KernelTiling(self.base_kernel, self.frustum, 1)

        self.camera_controller = camera_control.ArcballCameraController((0, 0, 0), 7000)
        self.skybox = Skybox(bottom_color=(0.1, 0.1, 0.5), top_color=(0.5, 0.5, 0.9))
        self.installEventFilter(self.camera_controller)

        self.geometry_display_list = None

        timer = QtCore.QTimer(self)
        timer.timeout.connect(self.animate)
        timer.start(15)

    def minimumSizeHint(self):
        return QtCore.QSize(512, 512)

    def sizeHint(self):
        return QtCore.QSize(800, 600)

    def initializeGL(self):
        glutInit(len(sys.argv), sys.argv)
        glClearColor(0, 0, 0, 1.0)
        glEnable(GL_MULTISAMPLE)
        glEnable(GL_DEPTH_TEST)

        glMatrixMode(GL_PROJECTION)
        gluPerspective(60, float(self.width())/float(self.height()), 0.5, 15000.0)

        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()

        glEnable(GL_LIGHTING)
        glEnable(GL_LIGHT0)
        glLightfv(GL_LIGHT0, GL_POSITION, (0, 1, 1, 0))
        glLightfv(GL_LIGHT0, GL_AMBIENT, (0.0, 0.0, 0.0, 1.0))
        glLightfv(GL_LIGHT0, GL_DIFFUSE, (0.3, 0.3, 0.3, 1.0))

        print "Generating geometry..."

        tilemap = dict()
        for k in self.kt.visible_kernels:
            color = np.random.rand(4)
            color[3] = 1.0

            for index, f, p in k.fundamental_domains:
                if str(index) in tilemap:
                    tilemap[str(index)][2] += 1
                else:
                    tilemap[str(index)] = [p, f, 1]

        self.geometry_display_list = glGenLists(1)
        glNewList(self.geometry_display_list, GL_COMPILE)
        glPushAttrib(GL_ENABLE_BIT)
        glEnable(GL_LIGHTING)
        for t in tilemap.values():
            if t[2] > 1:
                color = np.array((0.9, 0.2, 0.2))
            else:
                color = np.array((0.5, 0.5, 0.5))
            glMaterialfv(GL_FRONT_AND_BACK, GL_DIFFUSE, color)
            gl_geometry.draw_solid_prism(t[0])

        glDisable(GL_DEPTH_TEST)
        glDisable(GL_LIGHTING)
        glColor3f(1, 1, 1)
        for t in tilemap.values():
            gl_geometry.draw_wire_prism(t[0])

        glPopAttrib(GL_ENABLE_BIT)
        glEndList()

        print "Done generating geometry!"

    def paintGL(self):
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()

        rm = self.camera_controller.camera_matrix
        glMultMatrixf(rm.transpose())

        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        self.skybox.draw(self.camera_controller.camera_rotation.transpose(), 8000)

        glCallList(self.geometry_display_list)

        glPushAttrib(GL_ENABLE_BIT)
        glDisable(GL_DEPTH_TEST)
        glDisable(GL_LIGHTING)
        gl_geometry.draw_axes((10000, 10000, 10000))
        glColor3f(1, 1, 1)
        gl_geometry.draw_wire_prism(self.frustum)
        glPopAttrib(GL_ENABLE_BIT)

        glFinish()

    def resizeGL(self, w, h):
        glViewport(0, 0, w, h)
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluPerspective(60, float(self.width())/float(self.height()), 0.5, 15000.0)
        glMatrixMode(GL_MODELVIEW)

    def animate(self):
        self.elapsed = (self.elapsed + self.sender().interval())
        self.repaint()
        pass


if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    window = Window()
    window.show()
    sys.exit(app.exec_())

