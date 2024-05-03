import scipy.spatial.distance
import numpy as np
import numpy.typing as npt

from src import elements as el, utils
import queue


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

        for target in self.targets:
            self.grid[target.x, target.y] = el.ScenarioElement.target
        for obstacle in self.obstacles:
            self.grid[obstacle.x, obstacle.y] = el.ScenarioElement.obstacle
        np.random.seed(random_seed)

    def cost_function(r, r_max):
        if r < r_max:
            return np.exp(1 / (r ** 2 - r_max ** 2))
        else:
            return 0

    def _compute_pedestrian_grid(self) -> npt.NDArray[np.float64]:
        """Computes a grid with distances to the closest pedestrian."""
        pedestrian_positions = [(pedestrian.x, pedestrian.y) for pedestrian in self.pedestrians]
        pedestrian_positions = np.array(pedestrian_positions)
        pedestrian_distances = self._compute_naive_distance_grid(pedestrian_positions)
        return pedestrian_distances

    def compute_utility(self, pedestrian, new_position, distance_to_target_grid, pedestrian_grid):
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
        x, y = new_position

        # If the position is outside the grid bounds, return a very low utility
        if not (0 <= x < distance_to_target_grid.shape[0] and 0 <= y < distance_to_target_grid.shape[1]):
            return -float('inf')

        # Distance to target cost
        distance_to_target = distance_to_target_grid[x, y]

        # Interaction cost with other pedestrians (can be refined based on specific requirements)
        interaction_cost = 0
        r_max = 3  # hypothetical max interaction radius

        for other in self.pedestrians:
            if other != pedestrian:
                other_x, other_y = other.x, other.y
                # Check if other pedestrian is close enough to consider for interaction cost
                if abs(other_x - x) <= r_max and abs(other_y - y) <= r_max:
                    dist_to_other = np.hypot(x - other_x, y - other_y)
                    if dist_to_other < r_max:
                        interaction_cost += np.exp(1 / (dist_to_other ** 2 - r_max ** 2))

        # Calculate the total utility
        utility = -distance_to_target - interaction_cost
        return utility

    def is_within_bounds(self, measuring_point, position):
        """Check if a position is within the bounds defined by a measuring point."""
        upper_left = measuring_point.upper_left
        lower_right_x = upper_left.x + measuring_point.size.width
        lower_right_y = upper_left.y + measuring_point.size.height

        return (upper_left.x <= position.x < lower_right_x and
                upper_left.y <= position.y < lower_right_y)

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

        distance_grid = self.get_distance_grid()
        pedestrian_grid = self._compute_pedestrian_grid()
        finished = True
        active_pedestrians = []

        for pedestrian in self.pedestrians:
            current_position = utils.Position(pedestrian.x, pedestrian.y)

            for mp in self.measuring_points:
                if self.current_step >= mp.delay and self.current_step < mp.delay + mp.measuring_time:
                    if self.is_within_bounds(mp, current_position):
                        mp.record_pedestrian(pedestrian.speed)

            # Move pedestrians if they have not reached their target
            if distance_grid[current_position.x, current_position.y] == 0:
                if self.is_absorbing:
                    continue  # If the target is absorbing, do not add this pedestrian to the active list
                else:
                    active_pedestrians.append(pedestrian)  # Pedestrian remains active but doesn't move
                    continue

            possible_positions = []
            highest_utility = -float('inf')

            # Consider movement within the speed limit of the pedestrian
            for dx in range(-pedestrian.speed, pedestrian.speed + 1):
                for dy in range(-pedestrian.speed, pedestrian.speed + 1):
                    new_x, new_y = pedestrian.x + dx, pedestrian.y + dy
                    if 0 <= new_x < self.width and 0 <= new_y < self.height:
                        if self.grid[
                            new_x, new_y] != el.ScenarioElement.obstacle:  # Ensure the new position is not an obstacle
                            new_position = (new_x, new_y)
                            utility = self.compute_utility(pedestrian, new_position, distance_grid, pedestrian_grid)
                            if utility > highest_utility:
                                highest_utility = utility
                                possible_positions = [new_position]
                            elif utility == highest_utility:
                                possible_positions.append(new_position)

            if possible_positions:
                best_position = possible_positions[np.random.randint(len(possible_positions))]
                pedestrian.x, pedestrian.y = best_position  # Update pedestrian's position
                finished = False

            # Check if pedestrian has reached a target
            if not (distance_grid[(pedestrian.x, pedestrian.y)] == 0 and self.is_absorbing):
                active_pedestrians.append(pedestrian)

        self.pedestrians = active_pedestrians  # Update the list of active pedestrians
        self.current_step += 1  # Increment the simulation step

        return finished  # Return True if simulation should terminate

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

    def get_measured_flows(self) -> dict[int, float]:
        """Returns a map of measuring points' ids to their flows.

        Returns:
        --------
        dict[int, float]
            A dict in the form {measuring_point_id: flow}.
        """
        return {mp.id: mp.get_flow() for mp in self.measuring_points}

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

    def _compute_dijkstra_distance_grid(
            self, targets: tuple[utils.Position]
    ) -> npt.NDArray[np.float64]:
        """Computes the distance grid using Dijkstra's algorithm, considering obstacles as impassable.
        Targets start with a distance of zero. Cells that are obstacles or unreachable remain at infinity."""

        # Initialize the distance array with infinity
        distances = np.full((self.width, self.height), np.inf)
        # Mark obstacles in a separate array to exclude them from processing
        obstacles = self.grid == el.ScenarioElement.obstacle

        # Priority queue for Dijkstra's algorithm
        pq = queue.PriorityQueue()
        # Set the distance for target cells and add them to the queue
        for target in targets:
            x, y = target.x, target.y
            if not obstacles[x, y]:  # Only proceed if the target is not an obstacle
                distances[x, y] = 0
                pq.put((0, (x, y)))

        # Define movement directions (right, down, left, up)
        directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]

        while not pq.empty():
            current_distance, (x, y) = pq.get()

            # Process each neighbor
            for dx, dy in directions:
                nx, ny = x + dx, y + dy
                # Ensure the neighbor is within grid bounds and is not an obstacle
                if 0 <= nx < self.width and 0 <= ny < self.height and not obstacles[nx, ny]:
                    new_distance = current_distance + 1  # Increment the distance
                    # Only update if the new distance is smaller
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