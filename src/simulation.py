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
        self.width, self.height = config.grid_size.width, config.grid_size.height
        self.output_filename = config.output_filename
        self.grid = np.full((self.width, self.height), el.ScenarioElement.empty)
        self.pedestrians = config.pedestrians
        self.targets = config.targets
        self.obstacles = config.obstacles
        self.is_absorbing = config.is_absorbing
        self.distance_computation = config.distance_computation
        self.measuring_points = config.measuring_points  # Assuming this is provided by the configuration
        self.current_step = 0  # Initialize current_step attribute
        self.measuring_point_data = {mp.ID: [] for mp in self.measuring_points}  # Data collection
        self.occupied_positions = set()

        for target in self.targets:
            self.grid[target.x, target.y] = el.ScenarioElement.target
        for obstacle in self.obstacles:
            self.grid[obstacle.x, obstacle.y] = el.ScenarioElement.obstacle
            self.occupied_positions.add((obstacle.x, obstacle.y))
        np.random.seed(random_seed)
        # obstacles在一次模拟中是固定的，所以计算distance to closest target网格只需要进行一次
        self.target_positions = tuple(self.targets)
        self.distance_to_targets = self._compute_distance_grid(self.target_positions)

    def _cost_function(self,r, r_max):
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
        safe_r = np.where(mask, r, r_max - 1e-10)  # Use r_max where r >= r_max to avoid negative under the square root
        cost = np.where(mask, np.exp(1 / (safe_r ** 2 - r_max ** 2)), 0)

        return cost

    def _compute_pedestrian_grid(self) -> npt.NDArray[np.float64]:
        """Computes a grid with distances to the closest pedestrian."""
        pedestrian_positions = [(pedestrian.x, pedestrian.y) for pedestrian in self.pedestrians]
        pedestrian_positions = np.array(pedestrian_positions)
        pedestrian_positions = tuple(pedestrian_positions)
        pedestrian_distances = self._compute_naive_distance_grid(pedestrian_positions)
        return pedestrian_distances

    def _compute_utility(self, pedestrian_grid, r_max):
        """
        Calculate the utility for a given position based on distance to target and interaction with other pedestrians.

        Parameters:
        pedestrian - The pedestrian for whom to calculate the utility.
        new_position - The tuple (x, y) representing the new position to evaluate.
        distance_to_target_grid - A grid containing distance to the nearest target, adjusted for obstacles via Dijkstra's algorithm.
        pedestrian_grid - A grid indicating the presence of other pedestrians, used for calculating interaction costs.

        Returns:
        Utility value for the new position.
        """
        utility = -self.distance_to_targets - self._cost_function(pedestrian_grid,r_max)
        return utility
    def update(self, perturb: bool = True) -> bool:

        """Performs one step of the simulation.

        Arguments:
        ----------
        perturb : bool
            If perturb=False, pedestrians' positions are updates in the
            fixed order. Otherwise, the pedestrians are shuffle before
            performing an update.

        Returns:
        --------
        bool
            True if all pedestrians reached a target and the simulation
            is over, False otherwise.
        """

        if perturb:
            np.random.shuffle(self.pedestrians)

        finished = True
        active_pedestrians = []
        for pedestrian in self.pedestrians:
            reachable_positions = self.get_reachable_positions(pedestrian, pedestrian.speed)
            highest_utility = -float('inf')
            best_position = None
            target_positions = [(target.x, target.y) for target in self.targets]
            pedestrian_distance = self._compute_pedestrian_grid()
            utility_grid = self._compute_utility(pedestrian_distance,r_max=5)
            for pos in reachable_positions:
                x, y = pos
                utility_value = utility_grid[x][y]  # Assuming utility grid is precomputed correctly
                if utility_value > highest_utility:
                    highest_utility = utility_value
                    best_position = pos
            if best_position not in target_positions or not self.is_absorbing:
                self.grid[pedestrian.x, pedestrian.y] = el.ScenarioElement.empty
                pedestrian.x, pedestrian.y = best_position
                active_pedestrians.append(pedestrian)
                self.grid[pedestrian.x, pedestrian.y] = el.ScenarioElement.pedestrian
            elif best_position in target_positions and self.is_absorbing:
                pedestrian.x, pedestrian.y = best_position
              # If position is a target, do not add to active_pedestrians, simulating absorption
                finished = False
            else:
                active_pedestrians.append(pedestrian)  # No movement but still active

        self.pedestrians = active_pedestrians
        self.current_step += 1

        return finished

    def get_reachable_positions(self, pedestrian, speed):
        """Calculate reachable positions for a pedestrian considering their speed, other pedestrians, and obstacles."""
        x_start, y_start = pedestrian.x, pedestrian.y
        reachable_positions = []
        speed = math.ceil(speed)

        # Update occupied positions to consider current pedestrians and obstacles.
        occupied_positions = set((p.x, p.y) for p in self.pedestrians if p != pedestrian)
        occupied_positions.update(self.occupied_positions)

        # Check all cells within the movement radius defined by speed
        for dx in range(-speed, speed + 1):
            for dy in range(-speed, speed + 1):
                if dx ** 2 + dy ** 2 <= speed ** 2:  # Check within circular movement range
                    new_x, new_y = x_start + dx, y_start + dy
                    if 0 <= new_x < self.width and 0 <= new_y < self.height:
                        if (new_x, new_y) not in occupied_positions:
                            reachable_positions.append((new_x, new_y))
        return reachable_positions

    def get_grid(self) -> npt.NDArray[el.ScenarioElement]:
        """Returns a full state grid of the shape (width, height)."""
        grid = np.full((self.width, self.height), el.ScenarioElement.empty, dtype=el.ScenarioElement)
        # Place static elements: targets and obstacles
        for target in self.targets:
            grid[target.x, target.y] = el.ScenarioElement.target
        for obstacle in self.obstacles:
            grid[obstacle.x, obstacle.y] = el.ScenarioElement.obstacle

        # Place dynamic elements: pedestrians
        for pedestrian in self.pedestrians:
            # Check if the pedestrian is within grid bounds to avoid out-of-index errors
            if 0 <= pedestrian.x < self.width and 0 <= pedestrian.y < self.height:
                grid[pedestrian.x, pedestrian.y] = el.ScenarioElement.pedestrian

        return grid

    def get_distance_grid(self) -> npt.NDArray[np.float64]:
        """Returns a grid with distances to a closest target."""

        # TODO: return a distance grid.
        distance_grid = self._compute_distance_grid(self.targets)
        return distance_grid

    def is_within_bounds(self, mp, position):
        """Check if a position is within the bounds defined by a measuring point."""
        upper_left = mp.upper_left
        lower_right_x = upper_left.x + mp.size.width
        lower_right_y = upper_left.y + mp.size.height
        return (upper_left.x <= position.x < lower_right_x and
                upper_left.y <= position.y < lower_right_y)

    def get_measured_flows(self):
        mean_flows = {}
        for mp_id, speeds in self.measuring_point_data.items():
            if speeds:
                mean_flows[mp_id] = sum(speeds) / len(speeds)
            else:
                mean_flows[mp_id] = 0.0
        return mean_flows

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

        return distances

    def _compute_dijkstra_distance_grid(self, targets: tuple[utils.Position]) -> npt.NDArray[np.float64]:
        """Computes the distance grid using Dijkstra's algorithm, considering obstacles as impassable."""
        distances = np.full((self.width, self.height), np.inf)  # Initialize distances to infinity
        obstacles = self.grid == el.ScenarioElement.obstacle  # Identify obstacle locations

        pq = queue.PriorityQueue()
        for target in targets:
            x, y = target.x, target.y
            if not obstacles[x, y]:  # Ensure target is not an obstacle
                distances[x, y] = 0
                pq.put((0, (x, y)))

        directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
        while not pq.empty():
            current_distance, (x, y) = pq.get()
            if current_distance > distances[x, y]:  # Check if a shorter path to (x, y) has been found
                continue

            for dx, dy in directions:
                nx, ny = x + dx, y + dy
                if 0 <= nx < self.width and 0 <= ny < self.height and not obstacles[nx, ny]:
                    new_distance = current_distance + 1  # Step cost is 1
                    if new_distance < distances[nx, ny]:
                        distances[nx, ny] = new_distance
                        pq.put((new_distance, (nx, ny)))

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
        neighbors = [utils.Position(x + dx, y + dy) for dx in range(-1, 2) for dy in range(-1, 2)
                     if (dx != 0 or dy != 0) and (0 <= x + dx < self.width) and (0 <= y + dy < self.height)]
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