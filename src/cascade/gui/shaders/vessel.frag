#version 330 core
// Clip the expanded vessel quad to a rounded capsule and apply its field color.
in vec4 v_start_color;
in vec4 v_end_color;
in vec2 v_capsule;
in float v_length;
in float v_half_width;
out vec4 frag_color;

void main() {
    float beyond = max(max(-v_capsule.x, v_capsule.x - v_length), 0.0);
    if (length(vec2(beyond, v_capsule.y)) > v_half_width) discard;
    float axial_position = clamp(v_capsule.x / max(v_length, 1.0e-5), 0.0, 1.0);
    frag_color = mix(v_start_color, v_end_color, axial_position);
}
