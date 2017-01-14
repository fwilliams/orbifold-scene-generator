import numpy as np
from OpenGL.GL import *


def draw_wire_grid(width, height, nx, ny):
    dx = width / nx
    dy = height / ny

    glBegin(GL_LINES)
    for i in range(-ny, ny):
        glVertex3f(-width, 0, dy * i)
        glVertex3f(width, 0, dy * i)

    for i in range(-nx, nx):
        glVertex3f(dx * i, 0, -height)
        glVertex3f(dx * i, 0, height)
    glEnd()


def draw_solid_grid(width, height, nx, ny, color1=(1, 1, 1, 1), color2=(0.5, 0.5, 0.5, 1.0)):
    glBegin(GL_QUADS)

    dx = width / nx
    dy = height / ny
    tile_type = 0
    for i in range(-ny, ny):
        tile_type = (tile_type + 1) % 2
        for j in range(-nx, nx):
            if tile_type == 1:
                glColor4f(*color1)
            else:
                glColor4f(*color2)

            glVertex3f(j * dx, 0, (i + 1) * dy)
            glVertex3f((j + 1) * dx, 0, (i + 1) * dy)
            glVertex3f((j + 1) * dx, 0, i * dy)
            glVertex3f(j*dx, 0, i*dy)

            tile_type = (tile_type + 1) % 2

    glEnd()


def draw_solid_prism(prism):
    i = 0
    for f in prism.faces:
        glBegin(GL_TRIANGLE_FAN)
        for v in f:
            glNormal3f(*prism.planes[i].normal3)
            glVertex4f(*prism.vertices[v])
        glEnd()
        i += 1


def draw_wire_prism(prism):
    glBegin(GL_LINES)
    for e in prism.edges:
        glVertex4f(*prism.vertices[e[0]])
        glVertex4f(*prism.vertices[e[1]])
    glEnd()


def draw_prism_normals(prism, scale):
    glBegin(GL_LINES)
    i = 0
    for f in prism.faces:
        bad_centroid = np.array((0.0, 0.0, 0.0))
        for v_i in f:
            bad_centroid += prism.vertices[v_i][:3]
        bad_centroid /= len(f)

        glVertex3f(*bad_centroid)
        glVertex3f(*(bad_centroid + scale*prism.planes[i].normal3))
        i += 1
    glEnd()


def draw_axes(size, color_x=(1, 0, 0), color_y=(0, 1, 0), color_z=(0, 0, 1), line_width=2.0):
    glPushAttrib(GL_ENABLE_BIT)
    glDisable(GL_LIGHTING)
    old_lw = glGetFloat(GL_LINE_WIDTH)
    glLineWidth(line_width)
    glBegin(GL_LINES)
    # X axis
    glColor3f(*color_x)
    glVertex3f(0.0, 0.0, 0.0)
    glVertex3f(size[0], 0.0, 0.0)
    # Y axis
    glColor3f(*color_y)
    glVertex3f(0.0, 0.0, 0.0)
    glVertex3f(0.0, size[1], 0.0)
    # Z axis
    glColor3f(*color_z)
    glVertex3f(0.0, 0.0, 0.0)
    glVertex3f(0.0, 0.0, size[2])
    glEnd()
    glLineWidth(old_lw)
    glPopAttrib(GL_ENABLE_BIT)

