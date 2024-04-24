import argparse
from src import elements as el, gui, utils


if __name__ == "__main__":
    # Define and parse command line arguments.
    parser = argparse.ArgumentParser(
        prog="mlcms_ex_1",
        description="Runs a vizualization of a cellular automaton.",
    )
    parser.add_argument(
        "--gui",
        type=str,
        help="Path to the .json file specifying the gui configuration.",
    )
    parser.add_argument(
        "--simulation",
        type=str,
        help="Path to the .json file specifying the simulation configuration.",
    )
    args = parser.parse_args()

    # Parse scenario configuration
    gui_dict = utils.parse_json(args.gui)
    gui_config = el.GUIConfig.from_dict(gui_dict)
    simulation_dict = utils.parse_json(args.simulation)
    simulation_config = el.SimulationConfig.from_dict(simulation_dict)

    # Start the simulation
    main_gui = gui.CellularAutomatonGUI(gui_config, simulation_config)
    main_gui.start_gui()
