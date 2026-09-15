#version 330 core
// Render each tissue sample as a circular point sprite with its supplied color.
in vec4 v_color;
out vec4 frag_color;
void main() {
    vec2 p = 2.0 * gl_PointCoord - 1.0;
    if (dot(p, p) > 1.0) discard;
    frag_color = v_color;
}
