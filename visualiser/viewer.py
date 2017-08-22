import sys
from PyQt5 import QtCore, QtOpenGL, QtWidgets
from PyQt5.QtWidgets import QWidget, QCheckBox, QApplication, QHBoxLayout,QMainWindow,QToolTip,QPushButton, QMessageBox,QAction,QTextEdit,QLabel, qApp
from PyQt5.QtGui import QIcon, QFont
from PyQt5.QtCore import Qt, QCoreApplication

import camera_control
import gl_geometry
from skybox import Skybox

try:
    from OpenGL.GL import *
except ImportError:
    app = QtWidgets.QApplication(sys.argv)
    QtWidgets.QMessageBox.critical(None, "OpenGL Import Error", "PyOpenGL must be installed to run this example.")
    sys.exit(1)

# not been used
# class Window(QtWidgets.QWidget):
#     def __init__(self):
#         super(Window, self).__init__()
#
#         self.gl_widget = GLWidget()
#
#         main_layout = QtWidgets.QHBoxLayout()
#         main_layout.addWidget(self.gl_widget)
#
#         self.setLayout(main_layout)
#         self.setWindowTitle("Hello GL")

class Window(QMainWindow):
    def __init__(self):
        super(Window, self).__init__()
        self.initUI()
        self._projectionType = 0
        self._showAxes = True;
        self._showNormals = False;
        self._showWires = True
        self._showSamples = False;

    def initUI(self):
        self.exitAction = QAction('&Exit', self)
        self.exitAction.setToolTip('Exit Application')
        self.exitAction.triggered.connect(qApp.quit)

        self.aboutAction = QAction('&Help', self)
        self.aboutAction.setToolTip('Show About Dlg')
        self.aboutAction.triggered.connect(self.showAboutDlg)

        self.changeProjectionAction = QAction('&Perspective', self)
        self.changeProjectionAction.setToolTip('Perspective / Orthogonal')
        self.changeProjectionAction.triggered.connect(self.changeProjection)

        self.toggleShowAxesAction = QAction('&Show Axes', self, checkable = True)
        self.toggleShowAxesAction.setToolTip('Toggle the axes')
        self.toggleShowAxesAction.triggered.connect(self.toggleShowAxes)
        self.toggleShowAxesAction.setChecked(True)

        self.toggleShowNormalsAction = QAction('&Show Normals', self, checkable = True)
        self.toggleShowNormalsAction.setToolTip('Toggle the normals')
        self.toggleShowNormalsAction.triggered.connect(self.toggleShowNormals)
        self.toggleShowNormalsAction.setChecked(False)

        self.toggleShowWireAction = QAction('&Show Wires', self, checkable = True)
        self.toggleShowWireAction.setToolTip('Toggle the wires')
        self.toggleShowWireAction.triggered.connect(self.toggleShowWires)
        self.toggleShowWireAction.setChecked(True)

        self.toggleShowSampleAction = QAction('&Show Sample', self, checkable = True)
        self.toggleShowSampleAction.setToolTip('Toggle the objects in prisms')
        self.toggleShowSampleAction.triggered.connect(self.toggleShowSamples)
        self.toggleShowSampleAction.setChecked(False)

        self.toolbar = self.addToolBar('ToolBar')
        self.toolbar.addAction(self.aboutAction)
        self.toolbar.addAction(self.exitAction)
        self.toolbar.addAction(self.toggleShowWireAction)
        self.toolbar.addAction(self.toggleShowSampleAction)
        self.toolbar.addAction(self.toggleShowNormalsAction)
        self.toolbar.addAction(self.toggleShowAxesAction)

        self.statusBar().showMessage('Ready') #only works for QMainWindow

        # menubar = self.menuBar()
        # fileMenu = menubar.addMenu('&Menu')
        # fileMenu.addAction(exitAction)
        #
        # aboutMenu = menubar.addMenu('&About')
        # aboutMenu.addAction(aboutAction)

        #self.gl_widget = GLWidget()
        #self.setCentralWidget(self.gl_widget)
        #self.setCentralWidget(gl_widget)

        self.setWindowTitle("GL Viewer")

    def setGLWidget(self, gl_widget):
        self.setCentralWidget(gl_widget)

    def showAboutDlg(self):
        about.show()

    def changeProjection(self):
        self._projectionType = (self._projectionType+1)%2
        if self._projectionType:
            self.changeProjectionAction.setText("Orthogonal")
        else:
            self.changeProjectionAction.setText("Perspective")

        self.update()

    def toggleShowAxes(self, checked = False):
        self._showAxes = checked

    def toggleShowNormals(self, checked = False):
        self._showNormals = checked

    def toggleShowWires(self, checked = False):
        self._showWires = checked

    def toggleShowSamples(self, checked = False):
        self._showSamples = checked

    @property
    def projectionType(self):
        return self._projectionType

    @property
    def showAxes(self):
        return self._showAxes

    @property
    def showNormals(self):
        return self._showNormals

    @property
    def showWires(self):
        return self._showWires

    @property
    def showSamples(self):
        return self._showSamples

class AboutDlg(QWidget):
    def __init__(self):
        super(AboutDlg, self).__init__()

        self.initUI()

    def initUI(self):
        text = QLabel("This is a tool designed to generate kaleidoscopic scene \n\n\
        Copyright@2017\n\n\
        Using right mouse to move the scene\n\
        Using left mouse to rotate the scene\n\
        Using middle wheel to zoom the scene\n")

        hbox = QHBoxLayout()
        hbox.addStretch(1)
        hbox.addWidget(text)

        self.setLayout(hbox)
        self.setGeometry(300, 300, 300, 200)
        self.setWindowTitle('About')


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

        self._window = Window()
        self._window.setGLWidget(self.gl_widget)

        # main_layout = QtWidgets.QHBoxLayout()
        # main_layout.addWidget(self.gl_widget)
        #
        # self.setLayout(main_layout)
        # self.setWindowTitle(title)

    def set_draw_function(self, draw_func):
        self.gl_widget.draw_callback = draw_func

    def set_init_function(self, init_func):
        self.gl_widget.init_callback = init_func

    def set_resize_function(self, resize_func):
        self.gl_widget.resize_callback = resize_func

    def set_event_function(self, event_func):
        self.gl_widget.user_event_callback.event_callback = event_func

    def run(self):
        # self.show()
        self._window.show()
        sys.exit(app.exec_())

    @property
    def mywindow(self):
        return self._window

    @property
    def flag_projection(self):
        return self._window.projectionType

    @property
    def flag_axes(self):
        return self._window.showAxes

    @property
    def flag_normals(self):
        return self._window.showNormals

    @property
    def flag_wires(self):
        return self._window.showWires

    @property
    def flag_Samples(self):
        return self._window.showSamples

app = QtWidgets.QApplication(sys.argv)
# window = Window()
about = AboutDlg()
