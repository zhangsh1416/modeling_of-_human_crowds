import json
import numpy as np
import csv
import os


CONFIG_FOLDER = "configs"
OUTPUT_FOLDER = "outputs"

CELL_SIZE = 0.4


def get_vertical_object(x: int, y_start: int, y_end: int) -> tuple[dict]:
    """Generates a list of positions in the provided vertical range.

    Parameters:
    -----------
    x : int
        The x coordinate of the vertical object.
    y_start : int
        The y coordinate of the lower end of the object.
    y_end : int
        The y coordinate of the upped end of the object.

    Returns:
    --------
    tuple[dict]
        A list of dictionaries with keys ('x', 'y') specifying positions
        of the object entries. Both ends at y_start and y_end are
        included to the object.
    """
    return tuple({"x": x, "y": y} for y in range(y_start, y_end + 1))


def get_horizontal_object(x_start: int, x_end: int, y: int) -> tuple[dict]:
    """Generates a list of positions in the provided horizontal range.

    Parameters:
    -----------
    x_start : int
        The x coordinate of the left end of the object.
    x_end : int
        The x coordinate of the right end of the object.
    y : int
        The y coordinate of the horizontal object.

    Returns:
    --------
    tuple[dict]
        A list of dictionaries with keys ('x', 'y') specifying positions
        of the object entries. Both ends at x_start and x_end are
        included to the object.
    """
    return tuple({"x": x, "y": y} for x in range(x_start, x_end + 1))


def generate_pedestrians(
    hor_span: tuple[int, int],
    vert_span: tuple[int, int],
    n_pedestrians: int,
    speed_bounds: tuple[float, float] = (1, 1),
    random_seed: int = 42,
) -> list[dict]:
    """Generates pedestrians in the specified rectangular area.

    Parameters:
    -----------
    hor_span : Tuple[int, int]
        A tuple (x_min, x_max) specifying horizontal borders of the
        spawn area.
    vert_span : Tuple[int, int]
        A tuple (y_min, y_max) specifying vertical borders of the
        spawn area.
    n_pedestrians : int
        The number of pedestrians to generate. The positions of the
        pedestrians are sampled from the uniform distribution.
    speed_bounds : Tuple[float, float] = (1, 1)
        A tuple (speed_min, speed_max) specifying speed bounds of
        pedestrians. The unit of speed is cells/step. A speed value for
        each pedestrian is sampled from the uniform distribution.
    random_seed : int = 42
        The random seed used to define a random generator.

    Returns:
    --------
    list[dict]
        A list of dictionaries with keys ('ID', 'x', 'y', 'speed')
        specifying the initial configuration of pedestrians in the
        simulation.
    """
    rng = np.random.default_rng(random_seed)
    spawn_width = hor_span[1] - hor_span[0] + 1
    spawn_height = vert_span[1] - vert_span[0] + 1

    positions = rng.choice(
        spawn_width * spawn_height, size=n_pedestrians, replace=False
    )
    xs = positions % spawn_width + hor_span[0]
    ys = positions // spawn_width + vert_span[0]
    speeds = rng.uniform(*speed_bounds, size=n_pedestrians)
    pedestrians = []
    for i, (x, y, speed) in enumerate(zip(xs, ys, speeds)):
        pedestrians.append({"ID": i, "x": int(x), "y": int(y), "speed": speed})
    return pedestrians


def save_json(
    filename: str,
    grid_size: dict,
    targets: tuple[dict],
    measuring_points: tuple[dict],
    obstacles: tuple[dict],
    pedestrians: tuple[dict],
    is_absorbing: bool,
    distance_computation: str,
    output_filename: str,
):
    """Saves the simulation configuration to a .json file."""

    config = {
        "grid_size": grid_size,
        "targets": targets,
        "obstacles": obstacles,
        "pedestrians": pedestrians,
        "measuring_points": measuring_points,
        "is_absorbing": is_absorbing,
        "distance_computation": distance_computation,
        "output_filename": output_filename,
    }
    with open(filename, "w") as fout:
        json.dump(config, fout, sort_keys=True, indent=4)


def task_1(filename: str):
    """Saves a configuration file for the Task 1."""

    config_filename = os.path.join(CONFIG_FOLDER, f"{filename}.json")
    grid_size = {"width": 5, "height": 5}
    targets = [{"x": 3, "y": 2}]
    measuring_points = []
    obstacles = []
    pedestrians = [{"ID": 1, "x": 1, "y": 1, "speed": 1}]
    is_absorbing = False
    distance_computation = "naive"
    output_filename = os.path.join(OUTPUT_FOLDER, f"{filename}.csv")

    save_json(
        config_filename,
        grid_size,
        targets,
        measuring_points,
        obstacles,
        pedestrians,
        is_absorbing,
        distance_computation,
        output_filename,
    )

# TODO: create configs for other tasks.

if __name__ == "__main__":
    task_1("toy_example")
