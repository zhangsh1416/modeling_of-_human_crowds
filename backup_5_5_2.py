import scipy.spatial.distance
import numpy as np
import numpy.typing as npt
from src import elements as el, utils
import queue
import math


class Simulation:
    """A class to simulate pedestrian movement in a grid environment."""

    def __init__(self, config: el.SimulationConfig, random_seed: int = 42):
        """
        Initializes a new instance of the Simulation.

        Parameters:
        config : el.SimulationConfig
            Configuration settings for the simulation including grid size, obstacles,
            targets, and pedestrian settings.
        random_seed : int
            Seed for the random number generator to ensure reproducibility (default 42).
        """
        self.grid_size = config.grid_size
        self.width, self.height = (
            config.grid_size.width,
            config.grid_size.height,
        )
        self.output_filename = config.output_filename
        self.grid = np.full(
            (self.width, self.height), el.ScenarioElement.empty
        )
        self.pedestrians = config.pedestrians
        self.targets = config.targets
        self.obstacles = config.obstacles
        self.is_absorbing = config.is_absorbing
        self.distance_computation = config.distance_computation
        self.measuring_points = (
            config.measuring_points
        )  # Assuming this is provided by the configuration
        self.current_step = 0  # Initialize current_step attribute
        self.measuring_point_data = {
            mp.ID: [] for mp in self.measuring_points
        }  # Data collection
        self.occupied_positions = set()
        for target in self.targets:
            self.grid[target.x, target.y] = el.ScenarioElement.target
        for obstacle in self.obstacles:
            self.grid[obstacle.x, obstacle.y] = el.ScenarioElement.obstacle
            self.occupied_positions.add((obstacle.x, obstacle.y))
        for pedestrian in self.pedestrians:
            print('adding pedestrian')
            self.grid[pedestrian.get_position().x, pedestrian.get_position().y] = el.ScenarioElement.pedestrian
        np.random.seed(random_seed)
        # obstacles在一次模拟中是固定的，所以计算distance to closest target网格只需要进行一次
        self.distance_to_targets = self._compute_distance_grid(self.targets)

    # Continue with your calculation

    def _cost_function(self, r, r_max) -> float:
        """
        Compute a cost function over a grid based on the distance to the nearest pedestrian,
        applying an exponential decay effect that diminishes beyond r_max.

        Args:
        r (np.ndarray): A 2D array of distances to the nearest pedestrian.
        r_max (float): The maximum effective distance for the cost calculation.

        Returns:
        np.ndarray: A 2D array of costs, where each element is calculated based on the proximity to the nearest pedestrian.
        """
        # Apply cost function only where the distance is less than r_max
        # and avoid division by zero or negative square root by ensuring r^2 - r_max^2 is positive
        mask = r < r_max
        safe_r = np.where(
            mask, r, r_max - 1e-10
        )  # Use r_max where r >= r_max to avoid negative under the square root
        cost = np.where(mask, np.exp(1 / (safe_r**2 - r_max**2)), 0)

        return cost

    def _compute_pedestrian_grid(
        self, current_pedestrian
    ) -> npt.NDArray[np.float64]:
        """Computes a grid with distances to the closest pedestrian."""
        pedestrian_positions = [
            (pedestrian.x, pedestrian.y)
            for pedestrian in self.pedestrians
            if pedestrian != current_pedestrian
        ]
        pedestrian_positions = np.array(pedestrian_positions)
        pedestrian_positions = tuple(pedestrian_positions)
        pedestrian_distances = self._compute_naive_distance_grid(
            pedestrian_positions
        )
        return pedestrian_distances

    def _compute_utility(self, pedestrian_grid, r_max) -> float:
        """
        Calculate the utility for a given position based on distance to target and interaction with other pedestrians.

        Parameters:
        pedestrian_grid - A grid indicating the presence of other pedestrians, used for calculating interaction costs.
        r_max - Hyperparamter of the cost function representing the avoidance radius of a pedestrian

        Returns:
        Utility value for the new position.
        """
        cost = self._cost_function(pedestrian_grid, r_max)
        utility = -self.distance_to_targets - cost
        return utility

    def update(self, perturb: bool = True) -> bool:
        """ Performs one step of the simulation.

        Parameters:

            perturb - bool. If true, shuffle the pedestrians in order to have a random movement order, and avoid having
            the same pedestrian moving first all the time.

            """
        if perturb:
            np.random.shuffle(self.pedestrians)

        finished = True
        # pedestrian_position = () # initialize a tuple to store position of current pedestrian
        targets = [(pos.x, pos.y) for pos in self.targets]


        for mp in self.measuring_points:
            if self.current_step >= mp.delay:
                for ped in self.pedestrians:
                    mp.record_entry(ped, self.current_step)
        for pedestrian in self.pedestrians:
            # Increase pedestrian move credit
            pedestrian.move_credit += pedestrian.speed
            """
            pedestrian_position = (pedestrian.x, pedestrian.y)
                       if pedestrian_position in mp1: # mp1 is a list consists of positions[()]
                m1 == True
            elif pedestrian_position in mp2:
                m2 == True
            elif pedestrian_position in mp3:
                m3 == True
            """
            while pedestrian.move_credit >= math.sqrt(2):
                pedestrian_distance = self._compute_pedestrian_grid(pedestrian)
                neighbours = self._get_neighbors((pedestrian.x, pedestrian.y))
                reachable_positions = self.get_reachable_positions(
                    pedestrian, neighbours
                )
                highest_utility = -float("inf")
                best_position = (0, 0)
                utility_values = self._compute_utility(
                    pedestrian_distance, r_max=2
                )
                if not reachable_positions:
                    break
                if reachable_positions:
                    for pos in reachable_positions:
                        x, y = pos
                        utility_value = utility_values[x][y]
                        # print(utility_value)
                        if utility_value > highest_utility:
                            highest_utility = utility_value
                            best_position = pos
                        if utility_value == highest_utility and (
                            abs(pedestrian.x - x) + abs(pedestrian.y - y) == 1
                        ):
                            highest_utility = utility_value
                            best_position = pos
                moving_distance = math.sqrt(
                    (pedestrian.x - best_position[0]) ** 2
                    + (pedestrian.y - best_position[1]) ** 2
                )
                # print(moving_distance)
                if best_position in targets:
                    if self.is_absorbing:
                        # Absorbent targets, removed when pedestrians arrive
                        self.pedestrians.remove(pedestrian)
                        pedestrian.move_credit = 0
                        break
                        finished = True
                    else:
                        # non absorbing target where pedestrians arrive but are not removed
                        break
                        pedestrian.move_credit -= 1
                        finished = True
                else:
                    # Move to a non-target position
                    pedestrian.x, pedestrian.y = best_position
                    self.grid[pedestrian.x, pedestrian.y] = (
                        el.ScenarioElement.pedestrian
                    )
                    pedestrian.move_credit -= moving_distance
                    if moving_distance == 0:
                        pedestrian.move_credit -= pedestrian.speed
                    finished = False
        for mp in self.measuring_points:
            if self.current_step >= mp.delay:
                for ped in self.pedestrians:
                    mp.record_exit_and_calculate_speed(ped, self.current_step)
                mp.calculate_instantaneous_flow(self.current_step)

        self.current_step += 1
        return finished



    def get_reachable_positions(self, pedestrian, neighbours):
        """calculate all reachable neighbours for a single pedestrian"""
        reachable_positions = []
        for neighbour in neighbours:
            pos_tuple = (neighbour.x, neighbour.y)
            self_position = (pedestrian.x, pedestrian.y)
            if pos_tuple not in self.occupied_positions:
                if not self.is_position_occupied_by_other_pedestrian(
                    neighbour.x, neighbour.y, pedestrian
                ):
                    reachable_positions.append(pos_tuple)
                    reachable_positions.append(self_position)
        return reachable_positions

    def is_position_occupied_by_other_pedestrian(self, x, y, pedestrian):
        # This function checks if any pedestrian other than the one passed as argument occupies the position (x, y)
        for ped in self.pedestrians:
            if ped != pedestrian and ped.x == x and ped.y == y:
                return True
        return False

    def get_grid(self) -> npt.NDArray[el.ScenarioElement]:
        """Returns a full state grid of the shape (width, height)."""
        grid = np.full(
            (self.width, self.height),
            el.ScenarioElement.empty,
            dtype=el.ScenarioElement,
        )
        # Place static elements: targets and obstacles
        for target in self.targets:
            grid[target.x, target.y] = el.ScenarioElement.target
        for obstacle in self.obstacles:
            grid[obstacle.x, obstacle.y] = el.ScenarioElement.obstacle

        # Place dynamic elements: pedestrians
        for pedestrian in self.pedestrians:
            # Check if the pedestrian is within grid bounds to avoid out-of-index errors
            if (
                0 <= pedestrian.x < self.width
                and 0 <= pedestrian.y < self.height
            ):
                grid[pedestrian.x, pedestrian.y] = (
                    el.ScenarioElement.pedestrian
                )
        return grid

    def get_distance_grid(self) -> npt.NDArray[np.float64]:
        """Returns a grid with distances to a closest target."""

        # TODO: return a distance grid.
        distance_grid = self._compute_distance_grid(self.targets)
        return distance_grid

    def get_measured_flows(self):
        mp = {mp.ID: mp.get_average_flow() for mp in self.measuring_points}
        return mp

    def _compute_distance_grid(
        self, targets: tuple[utils.Position]
    ) -> npt.NDArray[np.float64]:
        """TODO: write a docstring."""

        if len(targets) == 0:
            distances = np.zeros((self.width, self.height))
            return distances

        match self.distance_computation:
            case "naive":
                distances = self._compute_naive_distance_grid(targets)
            case "dijkstra":
                distances = self._compute_dijkstra_distance_grid(targets)
            case _:
                print(
                    "Unknown algorithm for computing the distance grid: "
                    f"{self.distance_computation}. Defaulting to the "
                    "'naive' option."
                )
                distances = self._compute_naive_distance_grid(targets)
        return distances

    def _compute_naive_distance_grid(
        self, targets: tuple[utils.Position]
    ) -> npt.NDArray[np.float64]:
        """Computes a distance grid without considering obstacles.

        Arguments:
        ----------
        targets : Tuple[utils.Position]
            A tuple of targets on the grid. For each cell, the algorithm
            computes the distance to the closes target.

        Returns:
        --------
        npt.NDArray[np.float64]
            An array of distances of the same shape as the main grid.
        """
        if not targets:  # 检查目标列表是否为空
            return np.full(
                (self.width, self.height), np.inf
            )  # 返回充满无穷大的数组，或其他合适的默认值
        targets = [[*target] for target in targets]
        targets = np.row_stack(targets)
        x_space = np.arange(0, self.width)
        y_space = np.arange(0, self.height)
        xx, yy = np.meshgrid(x_space, y_space)
        positions = np.column_stack([xx.ravel(), yy.ravel()])

        # after the target positions and all grid cell positions are stored,
        # compute the pair-wise distances in one step with scipy.
        distances = scipy.spatial.distance.cdist(targets, positions)

        # now, compute the minimum over all distances to all targets.
        distances = np.min(distances, axis=0)
        distances = distances.reshape((self.height, self.width)).T
        # print("naive")

        return distances

    def _compute_dijkstra_distance_grid(
        self, targets: tuple[utils.Position]
    ) -> npt.NDArray[np.float64]:
        """Computes the distance grid using Dijkstra's algorithm with Euclidean distance, considering obstacles as impassable.
        Each cell's distance is initialized to infinity unless it is a target. If a cell is an obstacle,
        or it is unreachable from any target, its distance remains infinity."""
        # Initialize the distance grid with infinity values
        distances = np.full((self.width, self.height), np.inf)
        # Determine obstacle locations in the grid
        obstacles = self.grid == el.ScenarioElement.obstacle

        # Priority queue to manage cells by distance
        pq = queue.PriorityQueue()

        # Initialize the queue with target positions at distance 0
        for target in targets:
            x, y = target.x, target.y
            if not obstacles[
                x, y
            ]:  # Only proceed if the target is not an obstacle
                distances[x, y] = 0
                pq.put((0, (x, y)))

        # Define relative positions for neighbor cells (N, E, S, W and the diagonals)
        directions = [
            (-1, -1),
            (-1, 0),
            (-1, 1),
            (0, -1),
            (0, 1),
            (1, -1),
            (1, 0),
            (1, 1),
        ]

        # Process the queue
        while not pq.empty():
            current_distance, (x, y) = pq.get()

            # Skip processing if we find a shorter path already processed
            if current_distance > distances[x, y]:
                continue

            # Check each neighbor of the current cell
            for dx, dy in directions:
                nx, ny = x + dx, y + dy
                # Ensure the neighbor is within bounds and is not an obstacle
                if (
                    0 <= nx < self.width
                    and 0 <= ny < self.height
                    and not obstacles[nx, ny]
                ):
                    # Calculate Euclidean distance for the step
                    euclidean_distance = math.sqrt(dx**2 + dy**2)
                    new_distance = (
                        current_distance + euclidean_distance
                    )  # Update distance using Euclidean formula
                    # Update the neighbor's distance if a shorter path is found
                    if new_distance < distances[nx, ny]:
                        distances[nx, ny] = new_distance
                        pq.put((new_distance, (nx, ny)))

        # The distances grid is returned, where unreachable and obstacle cells remain infinity
        return distances

    def _get_neighbors(
        self, position: utils.Position, shuffle: bool = True
    ) -> list[utils.Position]:
        """Returns a list of neighboring cells for the position.

        Arguments:
        ----------
        positions : utils.Position
            A position on the grid.
        shuffle : bool
            An indicator if neighbors should be shuffled or returned
            in the fixed order.

        Returns:
        --------
        list[utils.Position]
            An array of neighboring cells. Two cells are neighbors
            if they share a common vertex.
        """

        x, y = position
        neighbors = [
            utils.Position(x + dx, y + dy)
            for dx in range(-1, 2)
            for dy in range(-1, 2)
            if (dx != 0 or dy != 0)
            and (0 <= x + dx < self.width)
            and (0 <= y + dy < self.height)
        ]
        if shuffle:
            np.random.shuffle(neighbors)
        return neighbors

    def _post_process(self):
        """Saves the simulation results to a file."""
        if self.output_filename:
            with open(self.output_filename, "w") as f:
                f.write("ID, X, Y\n")
                for p in self.pedestrians:
                    f.write(f"{p.ID}, {p.x}, {p.y}\n")
