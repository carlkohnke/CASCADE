"""Color-map names and stops shared by the Qt and OpenGL preview renderers."""

COLORMAPS = ("viridis", "magma", "plasma", "inferno", "coolwarm", "jet", "gray")

MAP_STOPS = {
    "viridis": ((68, 1, 84), (49, 104, 142), (53, 183, 121), (253, 231, 37)),
    "magma": ((0, 0, 4), (91, 22, 126), (221, 73, 104), (252, 253, 191)),
    "plasma": ((13, 8, 135), (156, 23, 158), (237, 121, 83), (240, 249, 33)),
    "inferno": ((0, 0, 4), (87, 16, 110), (220, 81, 57), (252, 255, 164)),
    "coolwarm": ((59, 76, 192), (141, 176, 254), (244, 152, 122), (180, 4, 38)),
    "jet": ((0, 0, 128), (0, 220, 255), (255, 235, 0), (128, 0, 0)),
    "gray": ((30, 34, 42), (105, 112, 125), (185, 190, 199), (250, 250, 250)),
}

# Private alias retained while the preview implementation migrates to the
# public constant name.
_MAP_STOPS = MAP_STOPS

__all__ = ["COLORMAPS", "MAP_STOPS"]
