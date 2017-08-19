from OpenGL.GL import *


class Skybox(object):
    def __init__(self, bottom_color=(0, 0, 0), top_color=(0, 0, 0), surrounding_color=(0.5, 0.0, 0.0)):
        self._bottom_color = bottom_color
        self._top_color = top_color
        self._surrounding_color = surrounding_color

    def draw(self, rotation_mat, size):
        glPushAttrib(GL_ENABLE_BIT)
        glPushAttrib(GL_TRANSFORM_BIT)
        glPushAttrib(GL_TEXTURE_BIT)
        glPushMatrix()

        glLoadIdentity()
        glScalef(size, size, size)
        glDisable(GL_DEPTH_TEST)
        glDisable(GL_LIGHTING)
        glDisable(GL_CULL_FACE)
        glDisable(GL_BLEND)

        glLoadMatrixf(rotation_mat)
        glBegin(GL_QUADS)

        #left
        glColor3f(*self._surrounding_color)
        glVertex3f(-1, -1, -1)
        glVertex3f(-1, -1, 1)
        glColor3f(*self._top_color)
        glVertex3f(-1, 1, 1)
        glVertex3f(-1, 1, -1)

        #right
        glColor3f(*self._bottom_color)
        glVertex3f(1, -1, -1)
        glVertex3f(1, -1, 1)
        glColor3f(*self._top_color)
        glVertex3f(1, 1, 1)
        glVertex3f(1, 1, -1)

        #bottom
        glColor3f(*self._bottom_color)
        glVertex3f(-1, -1, -1)
        glVertex3f(1, -1, -1)
        glVertex3f(1, -1, 1)
        glVertex3f(-1, -1, 1)

        #top
        glColor3f(*self._top_color)
        glVertex3f(-1, 1, -1)
        glVertex3f(1, 1, -1)
        glVertex3f(1, 1, 1)
        glVertex3f(-1, 1, 1)

        #back
        glColor3f(*self._bottom_color)
        glVertex3f(-1, -1, -1)
        glVertex3f(1, -1, -1)
        glColor3f(*self._top_color)
        glVertex3f(1, 1, -1)
        glVertex3f(-1, 1, -1)

        #front
        glColor3f(*self._bottom_color)
        glVertex3f(-1, -1, 1)
        glVertex3f(1, -1, 1)
        glColor3f(*self._top_color)
        glVertex3f(1, 1, 1)
        glVertex3f(-1, 1, 1)

        glEnd()

        glPopMatrix()
        glPopAttrib(GL_ENABLE_BIT)
        glPopAttrib(GL_TRANSFORM_BIT)
        glPopAttrib(GL_TEXTURE_BIT)

