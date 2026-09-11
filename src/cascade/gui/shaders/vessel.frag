#version 330 core
in vec4 v_color;
in vec2 v_capsule;
in float v_length;
in float v_half_width;
out vec4 frag_color;

void main() {
    float beyond = max(max(-v_capsule.x, v_capsule.x - v_length), 0.0);
    if (length(vec2(beyond, v_capsule.y)) > v_half_width) discard;
    frag_color = v_color;
}
