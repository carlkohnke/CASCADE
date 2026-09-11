#version 330 core
layout(location = 0) in vec3 a_start;
layout(location = 1) in vec3 a_end;
layout(location = 2) in vec4 a_color;
layout(location = 3) in float a_radius;

uniform vec2 u_pixel_scale;
uniform float u_min_viewport;
uniform float u_zoom;
uniform float u_max_radius;
uniform float u_width_boost;

out vec4 v_color;
out vec2 v_capsule;
out float v_length;
out float v_half_width;

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
    const vec2 corner[6] = vec2[6](
        vec2(0.0, -1.0), vec2(1.0, -1.0), vec2(1.0, 1.0),
        vec2(0.0, -1.0), vec2(1.0, 1.0), vec2(0.0, 1.0)
    );
    vec4 p0 = cascade_project(a_start);
    vec4 p1 = cascade_project(a_end);
    vec2 delta_pixels = (p1.xy - p0.xy) / u_pixel_scale;
    float line_length = max(length(delta_pixels), 1.0e-5);
    vec2 direction = delta_pixels / line_length;
    vec2 normal = vec2(-direction.y, direction.x);
    float width = 1.7;
    if (a_radius >= 0.0 && u_max_radius > 0.0) {
        float ratio = clamp(a_radius / u_max_radius, 0.0, 1.0);
        float physical = 2.0 * max(a_radius, 0.0) * u_min_viewport
                         * u_zoom / max(u_span, 1.0e-12);
        width = clamp(0.75 + physical + 0.55 * sqrt(ratio), 0.8, 12.0);
    }
    width += u_width_boost;
    float half_width = 0.5 * width;
    vec2 c = corner[gl_VertexID];
    vec4 position = mix(p0, p1, c.x);
    float end_sign = c.x * 2.0 - 1.0;
    position.xy += (normal * c.y * half_width + direction * end_sign * half_width)
                   * u_pixel_scale;
    gl_Position = position;
    v_color = a_color;
    v_capsule = vec2(mix(-half_width, line_length + half_width, c.x),
                     c.y * half_width);
    v_length = line_length;
    v_half_width = half_width;
}
