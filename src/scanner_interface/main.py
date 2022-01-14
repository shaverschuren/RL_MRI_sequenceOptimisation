"""Main launcher for the scanner interface module"""

# Path setup
import os
import sys

dir = os.path.dirname(os.path.realpath(__file__))
src = os.path.dirname(dir)
root = os.path.dirname(src)
if dir not in sys.path: sys.path.append(dir)
if src not in sys.path: sys.path.append(src)
if root not in sys.path: sys.path.append(root)

# File-specific imports
import argparse                     # noqa: E402
import json                         # noqa: E402
import rmqReceiveImage as rmq       # noqa: E402


def get_args():
    """Create argument parser and extract arguments"""

    # Create parser
    parser = argparse.ArgumentParser(
        description='RMQ receive image inline python interface.'
    )
    # Add arguments
    parser.add_argument(
        '-m', metavar='machine_id', nargs='?',
        default="rtrabbit", help='rtrabbit or trumer'
    )
    parser.add_argument(
        '-ut', metavar='unit_test', type=bool, nargs='?',
        default=False, const=True,
        help=(
            'check whether server connection can be established, '
            'test is succesful if "Image received" is printed.'
        )
    )
    # Parse and return arguments
    args = parser.parse_args()
    return args


def read_config():
    """Read info from config file for scanner interaction"""

    # Define config path
    global root
    config_path = os.path.join(root, "config.json")

    # Read config file
    with open(config_path, 'r') as f:
        config = json.load(f)

    return config


if __name__ == "__main__":

    # Define starting image number
    img_number = 0

    # Read config data
    config = read_config()
    # Get arguments
    args = get_args()

    # Define global variables for rmq
    rmq.set_global_vars(img_number, args, config)

    # Setup channel to remote computer
    channel = rmq.rmq_setup_channel(args.m)

    # If -ut is passed, run a test
    if args.ut:
        rmq.rmq_unit_test(args.m)

    # Start listening
    print('[*] Waiting for images on', args.m, '. Type CTRL+C to exit')
    channel.start_consuming()
