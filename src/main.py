"""Main launcher script for this RL-based MRI sequence design project"""

# Path setup
import os
import sys

src = os.path.dirname(os.path.abspath(__file__))
root = os.path.dirname(src)
if root not in sys.path: sys.path.append(root)
if src not in sys.path: sys.path.append(src)

# File-specific imports
import argparse                 # noqa: E402
import subprocess               # noqa: E402
import json                     # noqa: E402
import atexit                   # noqa: E402
from datetime import datetime   # noqa: E402


def parse_args():
    """Parse arguments for this script"""

    # Setup arg parser
    parser = argparse.ArgumentParser(
        description="RL-based MRI sequence design : Main launcher"
    )
    parser._action_groups.pop()
    required = parser.add_argument_group('Required arguments')
    optional = parser.add_argument_group('Optional arguments')

    # Add arguments
    required.add_argument(
        "--metric", "-m", metavar="metric", type=str, required=True,
        help="Metric to optimize. Available: snr/cnr"
    )
    required.add_argument(
        "--platform", "-p", metavar="platform", type=str, required=True,
        help="Type of platform/environment. Available: scan/sim/epg"
    )
    optional.add_argument(
        "--mode", "-mo", metavar="mode", type=str,
        help="Mode to operate in. Available: train/test/both*/validation",
        default="both"
    )
    optional.add_argument(
        "--agent", "-a", metavar="agent", type=str,
        help=(
            "Type of optimizer agent. "
            "Available: dqn/ddpg/rdpg*/validation"
        ),
        default="rdpg"
    )
    optional.add_argument(
        "--single_fa", "-sf", action="store_true",
        help=(
            "If passed, we train the model to optimize a single flip angle. "
            "If not, we optimize the entirety of the echo train."
        )
    )
    optional.add_argument(
        "--pretrained_path", metavar="pretrained_path", type=str,
        help="Optional: Path to pretrained model"
    )
    optional.add_argument(
        "--episodes", metavar="episodes", type=str,
        help="Optional: Override the number of episodes to train with"
    )
    optional.add_argument(
        '--keep_files', '-kf', action="store_true",
        help=(
            "Whether to keep or clear the communication files already there. "
            "If argument is passed, keep them."
        )
    )
    optional.add_argument(
        '--keep_queue', '-kq', action="store_true",
        help=(
            "Whether to keep or clear the RabbitMQ queue. "
            "If argument is passed, keep it."
        )
    )
    optional.add_argument(
        "--auto_done", "-ad", action="store_true",
        help="Optional: Override the 'done' signal given by the model"
    )

    # Parse arguments
    args = parser.parse_args()

    # Return arguments
    return args


def read_config():
    """Read info from config file for scanner interaction"""

    # Declare global variables
    global root

    # Read config file
    with open(os.path.join(root, "config.json"), 'r') as f:
        config = json.load(f)

    return config


def check_args(args, config):
    """Check arguments and config file validity"""

    # Argument check
    if args.metric.lower() not in ["snr", "cnr"]:
        raise ValueError(
            "The 'metric' argument should be in ['snr', 'cnr']"
        )
    if args.platform.lower() not in ["scan", "sim", "epg"]:
        raise ValueError(
            "The 'platform' argument should be in ['scan', 'sim', 'epg']"
        )
    if args.mode.lower() not in ["train", "test", "both", "validation"]:
        raise ValueError(
            "The 'mode' argument should be in "
            "['train', 'test', 'both', 'validation']"
        )
    if args.agent.lower() not in ['dqn', 'ddpg', 'rdpg', 'validation']:
        raise ValueError(
            "The 'agent' argument should be in "
            "['dqn', 'ddpg', 'rdpg', 'validation']"
        )

    # Config check (only for 'sim' and 'scan' modes)
    if args.platform.lower() in ["sim", "scan"]:
        # Check if all necessary fields are there
        if not all(
            elem in config.keys()
            for elem in ["param_loc", "data_loc", "data_store_loc"]
        ):
            raise UserWarning(
                "Config file incomplete. It should contain the following "
                "fields:\n['param_loc', 'data_loc', 'data_store_loc']"
            )

    # Argument interaction check
    if not args.single_fa and args.platform.lower() != "epg":
        raise NotImplementedError(
            "The full echo train optimization is only implemented for "
            "the 'epg' mode for now."
        )
    if not args.single_fa and args.agent.lower() not in ["rdpg", "validation"]:
        raise NotImplementedError(
            "The full echo train optimization is only implemented for "
            "the 'rdpg' optimizer (because this works best)."
        )


def clear_files():
    """Clear communication files (read from config file)"""

    # Data files
    if os.path.exists(config["data_loc"]):
        os.remove(config["data_loc"])
    if os.path.exists(config["data_loc"] + ".lck"):
        os.remove(config["data_loc"] + ".lck")

    # Parameter files
    if os.path.exists(config["param_loc"]):
        os.remove(config["param_loc"])
    if os.path.exists(config["param_loc"] + ".lck"):
        os.remove(config["param_loc"] + ".lck")


def launch_processes(args):
    """Launch the appropriate processes for current session"""

    # Determine global vars
    global root, src

    # If on scan (scanner interface) or sim (scanner simulator) platform,
    # start the interface or simulator here.
    if args.platform in ["scan", "sim"]:
        # Scanner platform
        if args.platform == "scan":
            # Determine the process call
            platform = "interface"
            arguments = ["-k"] if args.keep_queue else []
            process_call = [
                "python",
                os.path.join(src, f"scanner_{platform}", "main.py"),
                *arguments
            ]
            # Generate logs path
            log_path = os.path.join(root, "logs", "interface_logs.txt")
        # Simulator platform
        elif args.platform == "sim":
            # Determine the process call
            platform = "simulator"
            arguments = ["-n" "1"] if args.metric == "snr" else ["-n", "2"]
            process_call = [
                "python",
                os.path.join(src, f"scanner_{platform}", "main.py"),
                *arguments
            ]
            # Generate logs path
            log_path = os.path.join(root, "logs", "simulator_logs.txt")
        else:
            raise RuntimeError("Shouldn't happen")

        # Run process and append logfile
        with open(log_path, mode="a") as log_file:
            # Print new line in logs
            now = datetime.now()
            log_file.write(
                now.strftime("\n========== %H:%M:%S %d-%m-%Y ==========\n")
            )
            # Start process
            scanner_process = subprocess.Popen(
                process_call,
                stdout=log_file,
                stderr=log_file
            )

    elif args.platform == "epg":
        scanner_process = None
    else:
        raise RuntimeWarning("This shouldn't happen")

    # Start the optimizer process
    process_call = [
        "python", os.path.join(src, "optimizer", "main.py"),
        "--metric", args.metric,
        "--platform", "scan" if args.platform in ["scan", "sim"] else "epg",
        "--agent", args.agent,
        "--mode", args.mode
    ]
    if args.pretrained_path:
        process_call.extend(["--pretrained_path", args.pretrained_path])
    if args.episodes:
        process_call.extend(["--episodes", args.episodes])
    if args.single_fa:
        process_call.extend(["--single_fa"])
    if args.auto_done:
        process_call.extend(["--auto_done"])
    if args.keep_queue:
        process_call.extend(["-kq"])

    optimizer_process = subprocess.Popen(
        process_call
    )

    return scanner_process, optimizer_process


def kill_processes():
    """Kill all subprocesses (used at exit)"""

    global scanner_process, optimizer_process

    if scanner_process: scanner_process.kill()
    if optimizer_process: optimizer_process.kill()


if __name__ == "__main__":

    # Parse arguments
    args = parse_args()

    # Read config file
    config = read_config()

    # Check argument validity
    check_args(args, config)

    # If applicable, clear files
    if not args.keep_files:
        clear_files()

    # Launch appropriate subprocesses
    scanner_process, optimizer_process = launch_processes(args)
    # Connect at-exit function
    atexit.register(kill_processes)

    # Wait till the optimizer process is done, then exit
    optimizer_process.wait()
