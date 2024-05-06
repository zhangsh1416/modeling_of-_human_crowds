import json
import numpy as np
import os


CONFIG_FOLDER = "configs"
OUTPUT_FOLDER = "outputs"

CELL_SIZE = 0.4

new_root_directory = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 改变当前工作目录
os.chdir(new_root_directory)

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
    os.makedirs(os.path.dirname(filename), exist_ok=True)

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
def task_4(filename: str):

    # Constants
    CELL_SIZE = 0.4  # each cell is 0.4 meters
    WIDTH_ROOM1 = int(10 / CELL_SIZE)
    HEIGHT_ROOM1 = int(10 / CELL_SIZE)
    WIDTH_ROOM2 = int(10 / CELL_SIZE)
    HEIGHT_ROOM2 = int(10 / CELL_SIZE)
    WIDTH_CORRIDOR = int(1 / CELL_SIZE)
    HEIGHT_CORRIDOR = int(5 / CELL_SIZE)

    # Obstacle setup: walls around the rooms and the corridor
    obstacles = (
        get_horizontal_object(5, 29, 5) +  # Top wall of Room 1
        get_horizontal_object(5, 29, 30) +  # Bottom wall of Room 1
        get_vertical_object(5, 6, 29) +  # Left wall of Room 1
        get_vertical_object(30, 5, 16) +  # Right wall of Room 1
        get_vertical_object(30, 19, 30) +
        get_horizontal_object(31,43, 16) +
        get_horizontal_object(31,43,19) +
        get_horizontal_object(43,68,5) +  # Top wall of Room 2
        get_horizontal_object(43,68,30) +  # Bottom wall of Room 2
        get_vertical_object(43,6,15) +  # Left wall of Room 2
        get_vertical_object(43,20,29) +
        get_vertical_object(68,6,16) + # Right wall of Room 2
        get_vertical_object(68,19,29)
    )

    # Generate pedestrians in the left quarter of Room 1
    n_pedestrians = 50
    hor_span = (6, 13)
    vert_span = (6, 24)
    pedestrians = generate_pedestrians(hor_span, vert_span, n_pedestrians, speed_bounds=(0.5,5))

    # Set the target at the exit in Room 2
    targets = [{"x": 68, "y": y} for y in range(17, 19)]

    # Save configuration to JSON
    config_filename = os.path.join(CONFIG_FOLDER, f"{filename}.json")
    print(config_filename)
    save_json(
        config_filename,
        {"width": 75, "height": 35 },
        tuple(targets),
        tuple([]),
        tuple(obstacles),
        tuple(pedestrians),
        is_absorbing=True,
        distance_computation="naive",
        output_filename=os.path.join(OUTPUT_FOLDER, f"{filename}.csv"),
    )

def task_4_chicken_test(filename: str):
    # Constants
    CELL_SIZE = 0.4  # each cell is 0.4 meters
    WIDTH_ROOM1 = int(10 / CELL_SIZE)
    HEIGHT_ROOM1 = int(10 / CELL_SIZE)
    WIDTH_ROOM2 = int(10 / CELL_SIZE)
    HEIGHT_ROOM2 = int(10 / CELL_SIZE)
    WIDTH_CORRIDOR = int(1 / CELL_SIZE)
    HEIGHT_CORRIDOR = int(5 / CELL_SIZE)

    # Obstacle setup: walls around the rooms and the corridor
    obstacles = (
            get_horizontal_object(5, 29, 5) +
            get_horizontal_object(5, 29, 30) +
            get_vertical_object(30, 5, 30)
    )

    # Generate pedestrians in the left quarter of Room 1
    n_pedestrians = 50
    hor_span = (6, 13)
    vert_span = (6, 24)
    pedestrians = generate_pedestrians(hor_span, vert_span, n_pedestrians, speed_bounds=(0.5,0.5))

    # Set the target at the exit in Room 2
    targets = [{"x": 35, "y": 18}]

    # Save configuration to JSON
    config_filename = os.path.join(CONFIG_FOLDER, f"{filename}.json")
    print(config_filename)
    save_json(
        config_filename,
        {"width": 40, "height": 35},
        tuple(targets),
        tuple([]),
        tuple(obstacles),
        tuple(pedestrians),
        is_absorbing=True,
        distance_computation="naive",
        output_filename=os.path.join(OUTPUT_FOLDER, f"{filename}.csv"),
    )

def rimea_1():

    # Obstacle setup: walls around the rooms and the corridor
    obstacles = (
            get_horizontal_object(5, 105, 4) +  # Top wall of Room 1
            get_horizontal_object(5, 105, 10)  # Bottom wall of Room 1
    )

    n_pedestrians = 1
    hor_span = (5, 5)
    vert_span = (8, 8)
    pedestrians = generate_pedestrians(hor_span, vert_span, n_pedestrians, speed_bounds=(3.125, 3.542))
    # Set the target at the exit in Room 2
    targets = [{"x": 105, "y": y} for y in range(5, 10)]

    # Save configuration to JSON
    config_filename = os.path.join(CONFIG_FOLDER, "rimea_1.json")
    print(config_filename)
    save_json(
        config_filename,
        {"width": 110, "height": 20},
        tuple(targets),
        tuple([]),
        tuple(obstacles),
        tuple(pedestrians),
        is_absorbing=True,
        distance_computation="naive",
        output_filename=os.path.join(OUTPUT_FOLDER, "rimea_1.csv"),
    )
def rimea_2():
    # Obstacle setup: walls around the rooms and the corridor
    obstacles = (
            get_horizontal_object(5, 255, 4) +  # Top wall of Room 1
            get_horizontal_object(5, 255, 25)  # Bottom wall of Room 1
    )

    n_pedestrians = 80
    hor_span = (6, 30)
    vert_span = (5, 24)
    pedestrians = generate_pedestrians(hor_span, vert_span, n_pedestrians, speed_bounds=(1,1.5))
    # Set the target at the exit in Room 2
    targets = [{"x": 255, "y": y} for y in range(5, 25)]
    mp = [
        {
            "ID": 0,
           "upper_left": {"x": 125, "y": 5},
            "size": {"width": 4, "height": 2},
            "delay": 10,
            "measuring_time": 200
        },
        {
            "ID": 1,
            "upper_left": {"x": 105, "y": 15},
            "size": {"width": 5, "height": 5},
            "delay": 10,
            "measuring_time": 200
        }
    ]

    # Save configuration to JSON
    config_filename = os.path.join(CONFIG_FOLDER, "rimea_2.json")
    print(config_filename)
    save_json(
        config_filename,
        {"width": 260, "height": 35},
        tuple(targets),
        tuple(mp),
        tuple(obstacles),
        tuple(pedestrians),
        is_absorbing=True,
        distance_computation="naive",
        output_filename=os.path.join(OUTPUT_FOLDER, "rimea_2.csv"),
    )
def rimea_3():
    # Obstacle setup: walls around the rooms and the corridor
    obstacles = (
            get_horizontal_object(0, 30, 30) +  # Top wall of Room 1
            get_horizontal_object(0, 36, 36) + # Bottom wall of Room 1
            get_vertical_object(30,5,29) +
            get_vertical_object(36,5,35)
    )

    n_pedestrians = 20
    hor_span = (5, 14)
    vert_span = (31, 35)
    pedestrians = generate_pedestrians(hor_span, vert_span, n_pedestrians, speed_bounds=(3.125, 3.542))
    # Set the target at the exit in Room 2
    targets = [{"x": x, "y": 5} for x in range(31, 36)]

    # Save configuration to JSON
    config_filename = os.path.join(CONFIG_FOLDER, "rimea_3.json")
    print(config_filename)
    save_json(
        config_filename,
        {"width": 40, "height": 40},
        tuple(targets),
        tuple([]),
        tuple(obstacles),
        tuple(pedestrians),
        is_absorbing=True,
        distance_computation="dijkstra",
        output_filename=os.path.join(OUTPUT_FOLDER, "rimea_3.csv"),
    )
def rimea_4():

    # Obstacle setup: walls around the rooms and the corridor
    obstacles = (
            get_horizontal_object(5, 105, 4) +  # Top wall of Room 1
            get_horizontal_object(5, 105, 20)  # Bottom wall of Room 1
    )

    n_pedestrians = 50
    hor_span = (5, 25)
    vert_span = (5, 19)
    pedestrians = generate_pedestrians(hor_span, vert_span, n_pedestrians, speed_bounds=(3.125, 3.542))
    # Set the target at the exit in Room 2
    targets = [{"x": 105, "y": y} for y in range(5, 20)]

    # Save configuration to JSON
    config_filename = os.path.join(CONFIG_FOLDER, "rimea_4.json")
    print(config_filename)
    save_json(
        config_filename,
        {"width": 110, "height": 30},
        tuple(targets),
        tuple([]),
        tuple(obstacles),
        tuple(pedestrians),
        is_absorbing=True,
        distance_computation="naive",
        output_filename=os.path.join(OUTPUT_FOLDER, "rimea_4.csv"),
    )
# TODO: create configs for other tasks.

if __name__ == "__main__":
       #task_1("toy_example")
  #  task_4("task_4")
    #task_4_chicken_test("task_4_chicken_test")
    #rimea_1()
   rimea_2()
   #rimea_3()
   #rimea_4()