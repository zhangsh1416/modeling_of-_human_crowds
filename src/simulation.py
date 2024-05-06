import scipy.spatial.distance
import numpy as np
import numpy.typing as npt
from src import elements as el, utils
import queue
import math
import pandas as pd

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
        self.rimea4 = {p.ID:[p.x,p.y,0,p.age]for p in self.pedestrians}
        for target in self.targets:
            self.grid[target.x, target.y] = el.ScenarioElement.target
        for obstacle in self.obstacles:
            self.grid[obstacle.x, obstacle.y] = el.ScenarioElement.obstacle
        np.random.seed(random_seed)
        self.distance_to_targets = self._compute_distance_grid(self.targets)
        self.init_pos = {p.ID:[p.x,p.y,0,p.age]for p in self.pedestrians}
        # Read the CSV file
       # data = pd.read_csv('C:\\Users\\shihong\\Desktop\MLCMS\\mlicms24newex1-groupf\\configs\\rimea_7_speeds.csv')
        #sample = data.sample(n=len(self.pedestrians), random_state=1, replace=False)
        self.age = []
        self.flows = []

        # Iterate over the sampled data and assign to pedestrians
        """        
        it was used for 4th test in task5
                for pedestrian, (index, row) in zip(self.pedestrians, sample.iterrows()):
            pedestrian.age = row['age']
            pedestrian.speed = row['speed']
            self.age.append(pedestrian.age)
            print(pedestrian.age,pedestrian.ID, pedestrian.speed)
            
            """

        #print(self.targets)
        #print(self.is_absorbing)



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
        Calculate the utility for a position in the pedestrian grid by considering the distance to targets and the
    interaction costs with other pedestrians. The utility is computed as the negative sum of the distance to the
    nearest target and the interaction cost, which increases with pedestrian density.

    Parameters:
    - pedestrian_grid (array-like): A grid representing the current positions of pedestrians, where each element's
      value indicates the presence of a pedestrian.
    - r_max (int): The maximum radius to consider for pedestrian interactions, used in the cost function to
      determine the impact of nearby pedestrians on the utility.
        """
        utility = -self.distance_to_targets - self._cost_function(pedestrian_grid,r_max)
        return utility


    def update(self, perturb: bool = True) -> bool:
        """Performs a single step of the simulation, updating the positions of pedestrians based on their speeds, utility
    of potential positions, and interaction with measuring points. Optionally shuffles the order of pedestrians to
    simulate randomness in their movement orders.

    Parameters:
    - perturb (bool): If True, shuffles the list of pedestrians to randomize the update order, simulating more
      realistic variations in movement. Default is True.

    Returns:
    - bool: Returns True if all pedestrians have reached their targets or no pedestrians are left to move;
      returns False otherwise.
      """
        if perturb:
            np.random.shuffle(self.pedestrians)

        finished = True
        for mp in self.measuring_points:
            if (mp.delay+mp.measuring_time)>= self.current_step >= mp.delay:
                for pedestrian in self.pedestrians:
                    if (mp.upper_left.x ) <= pedestrian.x <= (mp.upper_left.x + mp.size.width) and (mp.upper_left.y ) <= pedestrian.y <= (mp.upper_left.y + mp.size.height):
                        mp.pedestrians_in[pedestrian.ID] = [pedestrian.x,pedestrian.y]
        for pedestrian in self.pedestrians:
            # 增加行人的移动信用
            pedestrian.move_credit += pedestrian.speed

            # 如果移动信用大于或等于1，则尝试移动行人
            while pedestrian.move_credit >= 1:
                reachable_positions = self.get_reachable_positions(pedestrian)
                highest_utility = -float('inf')
                best_position = None
                utility_values = self._compute_utility(self.distance_to_targets,r_max=3)

                for pos in reachable_positions:
                    x, y = pos
                    utility_value = utility_values[x][y]
                    if utility_value > highest_utility:
                        highest_utility = utility_value
                        best_position = pos

                moving_distance = math.sqrt(
                    (pedestrian.x - best_position.x) ** 2 + (pedestrian.y - best_position.y) ** 2)

                if best_position in self.targets:
                    if self.is_absorbing:
                        # 吸收型目标，行人到达后被移除
                        self.rimea4[pedestrian.ID] = [best_position.x,best_position.y,pedestrian.took_steps]
                        self.pedestrians.remove(pedestrian)
                        break
                    else:
                        # 非吸收型目标，行人到达但不被移除
                        self.rimea4[pedestrian.ID] = [best_position.x,best_position.y,pedestrian.took_steps]
                        pedestrian.move_credit = 0
                        break
                else:
                    # 移动到非目标位置
                    pedestrian.x, pedestrian.y = best_position
                    self.grid[pedestrian.x, pedestrian.y] = el.ScenarioElement.pedestrian
                    pedestrian.move_credit -= moving_distance
            pedestrian.took_steps += 1

        for mp in self.measuring_points:
            if (mp.delay+mp.measuring_time)>= self.current_step >= mp.delay:
                for ID in mp.pedestrians_in:
                    for pedestrian in self.pedestrians:
                        if ID == pedestrian.ID:
                            speed = math.sqrt((mp.pedestrians_in[ID][0]-pedestrian.x)**2 + (mp.pedestrians_in[ID][1]-pedestrian.y)**2)
                            speeds = []
                            speeds.append(speed)
                            mp.flows.append(sum(speeds))
                            speeds.clear()
            mp.pedestrians_in.clear()
        if self.pedestrians:
            finished = False
        self.current_step += 1
        return finished
    def get_reachable_positions(self, pedestrian):
        reachable_positions = []
        current_position = (pedestrian.x,pedestrian.y)
        neighbours = self._get_neighbors(current_position)
        pedestrians_pos = [(p.x, p.y) for p in self.pedestrians]
        targets_po = [(t.x, t.y) for t in self.targets]
        for neighbour in neighbours:
            if neighbour not in pedestrians_pos:
                if neighbour not in targets_po:
                    reachable_positions.append(neighbour)
        if current_position not in targets_po:
            reachable_positions.append(current_position)
        return reachable_positions

    def is_position_occupied_by_other_pedestrian(self, x, y, pedestrian):
        # This function checks if any pedestrian other than the one passed as argument occupies the position (x, y)
        for ped in self.pedestrians:
            if ped != pedestrian and ped.x == x and ped.y == y:
                return True
        return False

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
        # print(self.current_step)
        return grid

    def get_distance_grid(self) -> npt.NDArray[np.float64]:
        """Returns a grid with distances to a closest target."""

        # TODO: return a distance grid.
        distance_grid = self._compute_distance_grid(self.targets)
        return distance_grid

    def get_measured_flows(self):
        """not been used. mp function has been achieved by other methods """
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
        """
    Compute and return a grid of distances from each grid point to the nearest target in the given tuple of targets.

    The method calculates the distance based on the specified distance computation algorithm. It supports multiple
    algorithms including a 'naive' method and a 'dijkstra' method for more complex scenarios involving weighted paths.

    Parameters:
    - targets (tuple[utils.Position]): A tuple of Position objects representing the target locations for which distances are to be calculated.

    Returns:
    - npt.NDArray[np.float64]: A 2D NumPy array with distances, where each element at position (i, j) in the array represents the distance from the grid point at (i, j) to the nearest target.

    Raises:
    - ValueError: If an unknown distance computation algorithm is specified.
         """

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
        """
    Computes a distance grid for a grid-based layout using Dijkstra's algorithm, where each cell's distance is calculated
    considering obstacles as impassable and using Euclidean distance between cells. The distance from each cell in the
    grid to the nearest target is calculated, with distances initialized to infinity for all non-target cells.

    Parameters:
    - targets (tuple[utils.Position]): A tuple of Position objects representing the target positions within the grid
      for which the distances are to be calculated.

    Returns:
    - npt.NDArray[np.float64]: A 2D NumPy array of distances. Each element in the array represents the minimum
      distance from that cell to the nearest target. Cells that are either obstacles or unreachable from any target
      retain a value of infinity.

    """
        # Initialize the distance grid with infinity values
        distances = np.full((self.width, self.height), np.inf)
        # Determine obstacle locations in the grid
        obstacles = self.grid == el.ScenarioElement.obstacle

        # Priority queue to manage cells by distance
        pq = queue.PriorityQueue()

        # Initialize the queue with target positions at distance 0
        for target in targets:
            x, y = target.x, target.y
            if not obstacles[x, y]:  # Only proceed if the target is not an obstacle
                distances[x, y] = 0
                pq.put((0, (x, y)))

        # Define relative positions for neighbor cells (N, E, S, W and the diagonals)
        directions = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]

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
                if 0 <= nx < self.width and 0 <= ny < self.height and not obstacles[nx, ny]:
                    # Calculate Euclidean distance for the step
                    euclidean_distance = math.sqrt(dx ** 2 + dy ** 2)
                    new_distance = current_distance + euclidean_distance  # Update distance using Euclidean formula
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

        x, y = position[0],position[1]
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
