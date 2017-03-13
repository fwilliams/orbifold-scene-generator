import sys
from PyQt5 import QtCore, QtOpenGL, QtWidgets

import camera_control
import gl_geometry
from skybox import Skybox

try:
    from OpenGL.GL import *
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


class UserPluggableEventFilter(QtCore.QObject):
    def __init__(self):
        super(UserPluggableEventFilter, self).__init__()
        self.event_callback = None

    def eventFilter(self, source, event):
        if self.event_callback:
            self.event_callback(source, event)
        else:
            return super(UserPluggableEventFilter, self).eventFilter(source, event)


def null_user_callback(viewer):
    pass


class GLWidget(QtOpenGL.QGLWidget):
    def __init__(self, parent=None, size=(800, 600)):
        fmt = QtOpenGL.QGLFormat()
        fmt.setSampleBuffers(True)
        fmt.setSamples(8)
        fmt.setDoubleBuffer(True)
        super(GLWidget, self).__init__(fmt, parent)

        self.size = size
        self.setMouseTracking(True)
        self.elapsed = 0.0

        self.camera_controller = camera_control.ArcballCameraController((0, 0, 0), 3)
        self.skybox = Skybox(bottom_color=(0.1, 0.1, 0.5), top_color=(0.5, 0.5, 0.9))
        self.installEventFilter(self.camera_controller)

        self.draw_callback = null_user_callback
        self.init_callback = null_user_callback
        self.resize_callback = null_user_callback
        self.user_event_callback = UserPluggableEventFilter()
        self.installEventFilter(self.user_event_callback)

        timer = QtCore.QTimer(self)
        timer.timeout.connect(self.animate)
        timer.start(15)

    def minimumSizeHint(self):
        return QtCore.QSize(512, 512)

    def sizeHint(self):
        return QtCore.QSize(*self.size)

    def initializeGL(self):
        self.init_callback(self)

    def paintGL(self):
        self.draw_callback(self)
        glFinish()

    def resizeGL(self, w, h):
        glViewport(0, 0, w, h)
        self.resize_callback(self)

    def animate(self):
        self.elapsed = (self.elapsed + self.sender().interval())
        self.repaint()
        pass


class Viewer(QtWidgets.QWidget):
    def __init__(self, title="GL Viewer", size=(800, 600)):
        super(Viewer, self).__init__()

        self.gl_widget = GLWidget(size=size)

        main_layout = QtWidgets.QHBoxLayout()
        main_layout.addWidget(self.gl_widget)

        self.setLayout(main_layout)
        self.setWindowTitle(title)

    def set_draw_function(self, draw_func):
        self.gl_widget.draw_callback = draw_func

    def set_init_function(self, init_func):
        self.gl_widget.init_callback = init_func

    def set_resize_function(self, resize_func):
        self.gl_widget.resize_callback = resize_func

    def set_event_function(self, event_func):
        self.gl_widget.user_event_callback.event_callback = event_func

    def run(self):
        self.show()
        sys.exit(app.exec_())

app = QtWidgets.QApplication(sys.argv)


