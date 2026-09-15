#version 330 core
// Apply one uniform color to projected domain-boundary geometry.
uniform vec4 u_color;
out vec4 frag_color;
void main() { frag_color = u_color; }
