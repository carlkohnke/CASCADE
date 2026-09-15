#version 330 core
// Project domain-boundary line vertices for the retained Studio preview.
layout(location = 0) in vec3 a_position;

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

void main() { gl_Position = cascade_project(a_position); }
