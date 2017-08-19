import numpy as np
from PyQt5 import QtCore

from geometry import utils


class ArcballCameraController(QtCore.QObject):
    def __init__(self, center, radius):
        super(ArcballCameraController, self).__init__()
        self.center = utils.make_projective_point(center)
        self.last_mouse_point = None
        self.current_mouse_point = None
        self._camera_rotation = np.identity(4)
        self._camera_translation = np.identity(4)
        self._center_translation = np.identity(4)

        self._camera_center = utils.make_projective_point(center)
        self._center_translation[:3, 3] = center
        self._camera_translation[2][3] -= radius

        self.size = np.array((0, 0))
        self._zoom_speed = 0.1

        self.right_last_mouse_point = None
        self.right_current_mouse_point = None

    @staticmethod
    def _screen_pos_to_spherical(*args):
        if len(args) == 2:
            x = args[0]
            y = args[1]
        elif len(args) == 1:
            v = args[0]
            x = v[0]
            y = v[1]
        else:
            raise ValueError("Input to screen_pos_to_spherical must be a two co-ordinate vector or two floats")

        h_sq = x**2.0 + y**2.0

        if h_sq <= 1.0:
            return np.array((x, y, np.sqrt(1.0 - h_sq)))
        else:
            v = np.array((x, y, 0.0))
            return v / np.linalg.norm(v)

    @property
    def camera_matrix(self):
        mat = np.dot(self._camera_translation, self._camera_rotation)
        mat = np.dot(mat, self._center_translation)
        return mat

    @property
    def camera_rotation(self):
        return self._camera_rotation

    @property
    def camera_translation(self):
        return self._camera_translation

    @property
    def camera_center(self):
        return self._camera_center

    @property
    def camera_position(self):
        mat = np.dot(self._camera_rotation, self.camera_translation)
        mat = np.dot(self._center_translation, mat)
        return mat[:, 3]

    @property
    def camera_position3(self):
        mat = np.dot(self._camera_rotation, self.camera_translation)
        mat = np.dot(self._center_translation, mat)
        return mat[:3, 3]

    @property
    def dist_from_center(self):
        return np.linalg.norm(self.camera_position - self._camera_center)

    @property
    def zoom_speed(self):
        return self._zoom_speed

    def set_dist_from_center(self, dist):
        v = (self.camera_position - self._camera_center) / np.linalg.norm(self.camera_position - self._camera_center)
        self._camera_translation[2][3] = -dist

    def set_zoom_speed(self, zs):
        self._zoom_speed = zs

    def eventFilter(self, source, event):
        if event.type() == QtCore.QEvent.Resize:
            sz = event.size()
            self.size = np.array((sz.width(), sz.height()))
        if event.type() == QtCore.QEvent.MouseButtonPress:
            if event.buttons() == QtCore.Qt.LeftButton:
                pos = event.pos()
                self.last_mouse_point = np.array((pos.x(), pos.y()))
                self.current_mouse_point = self.last_mouse_point
                return True
            if event.buttons() == QtCore.Qt.RightButton:
                pos = event.pos()
                self.right_last_mouse_point = np.array((pos.x(), pos.y()))
                self.right_current_mouse_point = self.right_last_mouse_point
                return True
        elif event.type() == QtCore.QEvent.MouseMove:
            if event.buttons() == QtCore.Qt.LeftButton:
                pos = event.pos()

                self.last_mouse_point = self.current_mouse_point
                self.current_mouse_point = np.array((pos.x(), pos.y()))

                flip_y = np.array((1, -1))

                nrm_last_pos = (2.0 * self.last_mouse_point / self.size - 1.0) * flip_y
                nrm_cur_pos = (2.0 * self.current_mouse_point / self.size - 1.0) * flip_y

                p1 = ArcballCameraController._screen_pos_to_spherical(nrm_last_pos)
                p2 = ArcballCameraController._screen_pos_to_spherical(nrm_cur_pos)

                if (p1 == p2).all():
                    return super(ArcballCameraController, self).eventFilter(source, event)

                angle = np.arccos(np.dot(p1, p2))
                axis = np.cross(p1, p2)

                r_mat = utils.axis_angle_rotation_matrix(axis, angle)
                self._camera_rotation = np.dot(r_mat, self._camera_rotation)

            #move the camera
            if event.buttons() == QtCore.Qt.RightButton:
                pos = event.pos()

                self.right_last_mouse_point = self.right_current_mouse_point
                self.right_current_mouse_point = np.array((pos.x(), pos.y()))

                flip_y = np.array((1, -1))

                nrm_last_pos = (self.right_last_mouse_point) * flip_y
                nrm_cur_pos = (self.right_current_mouse_point) * flip_y

                if (nrm_last_pos == nrm_cur_pos).all():
                    return super(ArcballCameraController, self).eventFilter(source, event)

                # print("p1 is ",nrm_last_pos," p2 is",nrm_cur_pos)
                offset = 0.01 * (self.size[0] + self.size[1])
                d = (nrm_cur_pos - nrm_last_pos) * offset

                self._camera_translation[0:2, 3] += (d[0], d[1])
        elif event.type() == QtCore.QEvent.Wheel:
            notches = event.angleDelta().y() / 120
            self._camera_translation[2][3] += notches*self._zoom_speed
        return super(ArcballCameraController, self).eventFilter(source, event)
