import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import math
from src import simulation as sim, elements as el, utils
import matplotlib.pyplot as plt

class CellularAutomatonGUI:
    """This class defines a GUI for a cellular automaton simulation.

    Methods:
    -------
    start_gui(title: str = "Cellular Automaton GUI"):
        Starts the GUI with the provided window title.

    """

    def __init__(
        self, gui_config: el.GUIConfig, simulation_config: el.SimulationConfig
    ):
        """Parameters
        ----------
        gui_config : el.GUIConfig
            The configuration for the GUI.
        simulation_config : el.SimulationConfig
            The configuration for the simulation.
        """
        self._colors = gui_config.colors
        self._window_size = gui_config.window_size
        self._step_ms = gui_config.step_ms
        self._buttons_height = gui_config.buttons_height

        self._simulation_config = simulation_config
        self._simulation = sim.Simulation(simulation_config)

        self._win = None
        self._canvas = None
        self._canvas_size = None
        self._canvas_image = None
        self._grid_image = None
        self._show_distance = False
        self._is_running = False


    def start_gui(self, title: str = "Cellular Automaton GUI"):
        """Starts the GUI with the provided window title.

        The GUI contains a canvas showing a simulation state and buttons
        for manipulating the executing of a simulation. The GUI allows
        the user to run/stop the current simulation, perform one step of
        the simulation, restart the simulation, load a simulation from a
        configuration file, and show/hide distance values for cells of
        the cellular grid.

        Parameters
        ----------
        title : str
            The title of the GUI window.

        Returns
        -------
        None
        """

        # Create the main window.
        self._win = tk.Tk()
        self._win.geometry(f"{self._window_size}")
        self._win.title(title)

        # Add elements to the canvas.
        self._create_buttons()
        self._create_canvas()

        # Show the result.
        self._show_simulation_state()
        self._win.mainloop()

    def _load_simulation(self):
        """Opens a file dialog to load a simulation from a file."""

        simulation_json = filedialog.askopenfilename()

        if not simulation_json:
            return

        simulation_dict = utils.parse_json(simulation_json)
        self._simulation_config = el.SimulationConfig.from_dict(
            simulation_dict
        )
        self._simulation = sim.Simulation(self._simulation_config)

        self._canvas.pack_forget()
        self._create_canvas()
        self._show_simulation_state()

    def _restart_simulation(self):
        """Restarts the current simulation."""

        self._simulation = sim.Simulation(self._simulation_config)
        self._is_running = False
        self._show_simulation_state()

    def _switch_is_running(self):
        """Starts or stop the animation of the simulation."""

        if self._is_running:
            self._is_running = False
        else:
            self._is_running = True
            self._step_simulation(anim=True)

    def _switch_show_distance(self):
        """Switches between showing/hiding the distance values."""

        self._show_distance = not self._show_distance
        self._show_simulation_state()

    def calculate_speeds(self,rimea4, init_pos):
        # Create a dictionary to store the speed of each point
        speeds = []
        # Iterate over all items in one of the dictionaries
        for id, end_pos in rimea4.items():
            if id in init_pos:
                # Retrieve the initial position
                start_pos = init_pos[id]
                # Calculate the Euclidean distance
                distance = math.sqrt((end_pos[0] - start_pos[0]) ** 2 + (end_pos[1] - start_pos[1]) ** 2)
                # Get the time difference
                time = end_pos[2]
                # Calculate the speed, avoiding division by zero
                if time != 0:
                    speed = distance / time
                else:
                    speed = 0  # Define speed as 0 if time difference is 0 to avoid division by zero
                # Store the speed in the dictionary
                speeds.append(speed)
                # Print the ID and speed of each point
                #print(f"ID: {id}, Speed: {speed:.2f} units/time")
        return speeds


    def _step_simulation(self, anim=False):
        """Performs one step of a simulation.

        Parameters:
        -----------
        anim : bool = False
            If anim==True, the step is performed as a part of animation.

        Returns:
        --------
        None
        """

        finished = self._simulation.update()
        self._show_simulation_state()

        if finished:
            self._is_running = False
            """speeds = self.calculate_speeds(self._simulation.rimea4,self._simulation.init_pos)
            print(speeds)
            
            ages = self._simulation.age
            plt.scatter(ages, speeds)

            # 标题和标签
            plt.title("Age vs Observed Horizontal Walking Speed")
            plt.xlabel("Age (in years)")
            plt.ylabel("Walking Speed (m/s)")

            # 显示网格（可选）
            plt.grid(True)

            # 显示图表
            plt.show()"""
            for mp in self._simulation.measuring_points:
                if len(mp.flows) != 0:
                     print(mp.ID,sum(mp.flows)/len(mp.flows))
            print("finished")
        if anim and self._is_running:
            self._canvas.after(self._step_ms, self._step_simulation, anim)

    def _simulation_to_image(self):
        """Transforms the state of the simulation to a PIL image."""

        im = Image.new(
            mode="RGB", size=(self._simulation.width, self._simulation.height)
        )
        pix = im.load()
        distance_colors = utils.get_distance_colors(
            self._simulation.get_distance_grid()
        )
        grid = self._simulation.get_grid()

        for x in range(self._simulation.width):
            for y in range(self._simulation.height):
                if (
                    self._show_distance
                    and grid[x, y] == el.ScenarioElement.empty
                ):
                    pix[x, y] = tuple(distance_colors[x, y])
                else:
                    pix[x, y] = tuple(self._colors[grid[x, y]])
        return im

    def _show_simulation_state(self):
        """Shows the current state of the simulation on the canvas."""

        im = self._simulation_to_image()
        im = im.resize(
            (self._canvas_size.width, self._canvas_size.height),
            resample=Image.NONE,
        )
        self._grid_image = ImageTk.PhotoImage(im)
        self._canvas.itemconfigure(self._canvas_image, image=self._grid_image)

    def _create_buttons(self):
        """Creates buttons to control execution of the simulation."""

        # Create frame to hold two rows of buttons.
        master_frame = tk.Frame(self._win, height=self._buttons_height)
        top_frame = tk.Frame(master_frame, height=self._buttons_height // 2)
        bottom_frame = tk.Frame(master_frame, height=self._buttons_height // 2)

        # Create the buttons with the corresponding handlers.
        buttons = [
            (top_frame, "Run/stop the simulation", self._switch_is_running),
            (top_frame, "Step the simulation", self._step_simulation),
            (top_frame, "Restart the simulation", self._restart_simulation),
            (bottom_frame, "Load a simulation", self._load_simulation),
            (bottom_frame, "Show/hide distances", self._switch_show_distance),
        ]

        # Place buttons to the main frame.
        for frame, title, handler in buttons:
            btn = tk.Button(frame, text=title, command=handler)
            btn.pack(side=tk.LEFT, fill=None, expand=True)
        top_frame.pack(side=tk.TOP, fill=None, expand=True)
        bottom_frame.pack(side=tk.TOP, fill=None, expand=True)
        master_frame.pack()

    def _create_canvas(self):
        """Creates the main canvas to draw the simulation state on."""

        grid_size = self._simulation.grid_size
        self._canvas_size = utils.get_canvas_size(
            self._window_size, grid_size, self._buttons_height
        )
        self._canvas = tk.Canvas(
            self._win,
            width=self._canvas_size.width,
            height=self._canvas_size.height,
        )
        self._canvas_image = self._canvas.create_image(
            0, 0, image=None, anchor=tk.NW
        )
        self._canvas.pack(side=tk.TOP, fill=None, expand=True)
