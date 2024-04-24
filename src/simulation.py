import scipy.spatial.distance
import numpy as np
import numpy.typing as npt

from src import elements as el, utils
import copy
import queue


class Simulation:
    """TODO: write a docstring."""

    def __init__(self, config: el.SimulationConfig, random_seed: int = 42):
        """TODO: write a docstring."""

        self.width, self.height = config.grid_size.width, config.grid_size.height
        self.output_filename = config.output_filename

        # TODO: initialize other fields.

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

        # TODO: implement the update.

        finished = True
        if finished:
            self._post_process()
        return finished

    def get_grid(self) -> npt.NDArray[el.ScenarioElement]:
        """Returns a full state grid of the shape (width, height)."""

        # TODO: return a grid for visualization.
        
        grid = np.empty(shape=(0, 0))
        return grid

    def get_distance_grid(self) -> npt.NDArray[np.float64]:
        """Returns a grid with distances to a closest target."""

        # TODO: return a distance grid.
        distance_grid = np.full((self.width, self.height), np.inf)
        return distance_grid

    def get_measured_flows(self) -> dict[int, float]:
        """Returns a map of measuring points' ids to their flows.

        Returns:
        --------
        dict[int, float]
            A dict in the form {measuring_point_id: flow}.
        """
        return {}

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
        """TODO: write a docstring."""

        # TODO: implement the Dijkstra algorithm.

        distances = np.full((self.width, self.height), np.inf)
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

        # TODO: return all neighbors.

        neighbors = []
        return neighbors

    def _post_process(self):
        """TODO: write a docstring."""

        if self.output_filename is None:
            return
        
        # TODO: store output for analysis.