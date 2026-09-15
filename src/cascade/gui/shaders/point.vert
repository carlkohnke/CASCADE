#version 330 core
// Project tissue sample positions and pass their per-point colors to point.frag.
// The retained OpenGL preview loads this packaged shader; it is not standalone.
layout(location = 0) in vec3 a_position;
layout(location = 1) in vec4 a_color;
uniform float u_point_size;
out vec4 v_color;

uniform vec3 u_center;
uniform vec3 u_rotation_0;
uniform vec3 u_rotation_1;
uniform vec3 u_rotation_2;
uniform vec2 u_xy_scale;
uniform float u_span;

vec4 cascade_project(vec3 world) {
    vec3 p = (world - u_center) / max(u_span, 1.0e-12);
    vec3 rotated = vec3(
        dot(u_rotation_0, p),
        dot(u_rotation_1, p),
        dot(u_rotation_2, p)
    );
    float perspective = 1.0 / clamp(1.55 - 0.42 * rotated.z, 0.75, 2.2);
    return vec4(rotated.xy * u_xy_scale * perspective, -0.45 * rotated.z, 1.0);
}

void main() {
    gl_Position = cascade_project(a_position);
    gl_PointSize = u_point_size;
    v_color = a_color;
}
