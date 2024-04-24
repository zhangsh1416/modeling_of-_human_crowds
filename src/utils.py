import json
from dataclasses import dataclass
import matplotlib.cm as cm
import numpy as np
import numpy.typing as npt


@dataclass
class Size:
    """A class to represent a size tuple of (width, height)."""

    width: int
    height: int

    def __repr__(self):
        return f"{self.width}x{self.height}"


@dataclass(order=True)
class Position:
    """A class to represent a position tuple of (x, y)."""

    x: int
    y: int

    def __repr__(self):
        return f"({self.x}, {self.y})"

    def __iter__(self):
        return iter((self.x, self.y))


def parse_json(filename: str) -> dict:
    """Parses data from 'filename' into a dictionary.

    Parameters:
    -----------
    filename : str
        The path to the of a .json file.

    Returns:
    --------
    dict
        The dictionary with parsed key-value data.
    """

    with open(filename) as fin:
        content = json.load(fin)
    return content


def get_distance_colors(
    distances: npt.NDArray[np.float64], colormap: str = "viridis"
) -> npt.NDArray[np.int64]:
    """Assigns colors to distance values based on a uniform scale.

    Parameters:
    -----------
    distances : npt.NDArray[np.float64],
        An array containing distance values for each cell. The infinite
        distance value signifies that the target is unreachable from the
        given cell.
    colormap : str = 'viridis'
        A matplotlib compatible name of a colormap to get colors from.

    Returns:
    --------
    npt.NDArray[np.int64]
        An array of shape (*distance.shape, 3) holding RGB colors for
        each cell. The infinite distance values are transformed to the
        lowest color of the colormap.
    """
    mask = distances == float("inf")
    finite_distances = np.copy(distances)
    finite_distances[mask] = 0
    cmap = cm.get_cmap(colormap)
    rgba = cmap(1 - finite_distances / finite_distances.max())
    int_rgb = np.round(rgba[..., :3] * 255).astype(np.int64)
    return int_rgb


def get_canvas_size(
    window_size: Size, grid_size: Size, height_buffer: int, margin: int = 20
) -> Size:
    """Computes the canvas size that keeps the ratio of the grid_size.

    Parameters:
    -----------
    window_size : Size
        The size of the window holding the canvas in pixels.
    grid_size : Size
        The size of the grid in cells.
    height_buffer : int
        The height in pixels that is reserved for other elements of the
        window, e.g. buttons.
    margin : int
        The length in pixels for the margin between edges of the window
        and the canvas.

    Returns:
    --------
    Size
        The size of the canvas.
    """
    width_ratio = (window_size.width - margin) / grid_size.width
    height_ratio = (
        window_size.height - height_buffer - margin
    ) / grid_size.height
    ratio = int(min(width_ratio, height_ratio))
    return Size(grid_size.width * ratio, grid_size.height * ratio)


def get_distance(pos_a: Position, pos_b: Position) -> np.float64:
    """Computes the Euclidean distance between two positions."""
    return np.sqrt((pos_a.x - pos_b.x) ** 2 + (pos_a.y - pos_b.y) ** 2)
