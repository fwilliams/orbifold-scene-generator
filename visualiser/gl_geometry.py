import numpy as np
from OpenGL.GL import *


def draw_frustum_normals(frustum, scale=1.0, flip_normals=False):
    def compute_center(n1, n2, f1, f2):
        half_near = n1 + 0.5*(n2 - n1)
        half_far = f1 + 0.5*(f2 - f1)
        return half_near + 0.5*(half_far - half_near)

    scale = -scale if flip_normals else scale

    near_normal = frustum.near_plane.normal * scale
    left_normal = frustum.left_plane.normal * scale
    top_normal = frustum.top_plane.normal * scale
    far_normal = frustum.far_plane.normal * scale
    right_normal = frustum.right_plane.normal * scale
    bottom_normal = frustum.bottom_plane.normal * scale

    near_center = frustum.nbr + 0.5*(frustum.ntl - frustum.nbr)
    far_center = frustum.fbr + 0.5*(frustum.ftl - frustum.fbr)
    top_center = compute_center(frustum.ntl, frustum.ntr, frustum.ftl, frustum.ftr)
    bottom_center = compute_center(frustum.nbl, frustum.nbr, frustum.fbl, frustum.fbr)
    left_center = compute_center(frustum.nbl, frustum.ntl, frustum.fbl, frustum.ftl)
    right_center = compute_center(frustum.nbr, frustum.ntr, frustum.fbr, frustum.ftr)

    glBegin(GL_LINES)
    glVertex4f(*near_center)
    glVertex4f(*(near_center + near_normal))

    glVertex4f(*far_center)
    glVertex4f(*(far_center + far_normal))

    glVertex4f(*left_center)
    glVertex4f(*(left_center + left_normal))

    glVertex4f(*right_center)
    glVertex4f(*(right_center + right_normal))

    glVertex4f(*bottom_center)
    glVertex4f(*(bottom_center + bottom_normal))

    glVertex4f(*top_center)
    glVertex4f(*(top_center + top_normal))
    glEnd()


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


def draw_checker_plane(width, height, nx, ny, color1=(1, 1, 1, 1), color2=(0.5, 0.5, 0.5, 1.0)):
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

