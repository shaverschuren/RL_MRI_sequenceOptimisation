#!/usr/bin/python3
''' Some remarks, protobuf (protoc) needs to be installed on your system '''

# Path setup
import os
import sys

dir = os.path.dirname(os.path.realpath(__file__))
src = os.path.dirname(os.path.dirname(dir))
root = os.path.dirname(src)
if dir not in sys.path: sys.path.append(dir)
if src not in sys.path: sys.path.append(src)
if root not in sys.path: sys.path.append(root)

# File-specific imports
import argparse                 # noqa: E402
import numpy as np              # noqa: E402
import json                     # noqa: E402
import h5py                     # noqa: E402
import pika                     # noqa: E402
import dataobject_pb2 as pb     # noqa: E402 (pb = protobuf)


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


def callback_image(ch, method, properties, body):
    """Callback function for receiving an image"""

    # Increment image number
    global img_number
    img_number += 1
    # Print some info
    print("Image number", img_number, "received.")

    # Store received data in proper format (np array)
    dataobject = pb.DataObject()
    dataobject.ParseFromString(body)
    image_data = dataobject.image
    image_data = np.reshape(image_data, dataobject.size_dim)

    # Store image in proper files (might do this in memory later)
    # We write it in the img_data file (used as most recent image),
    # as well as in the image store folder (mostly for debugging)
    global config
    data_paths = [
        # Most recent image
        config['data_loc'],
        # Stored images
        os.path.join(config['data_store_loc'], f"img_{img_number}.h5")
    ]
    for data_path in data_paths:
        # Write data to locked h5 file
        hf = h5py.File(data_path + ".lck", 'w')
        hf.create_dataset('/img', data=np.array(image_data, dtype=np.float16))
        hf.close()
        # Move data file to actual reading location
        os.system(f"mv {data_path + '.lck'} {data_path}")


def rmq_setup_channel(machine_id):
    """Setup connection to remote device (rtrabbit)"""

    # Fill in credentials
    global config
    cred = pika.PlainCredentials(*config['remote_credentials'])
    # Establish connection
    connection = pika.BlockingConnection(pika.ConnectionParameters(
        host=machine_id, credentials=cred
    ))

    # Open channel and bind queues + callback
    channel = connection.channel()
    channel.queue_declare(queue='reconsocket_image_dev')
    channel.queue_bind(
        exchange='reconsocket_image_dev',
        queue='reconsocket_image_dev',
        routing_key='reconsocket.image'
    )
    channel.basic_consume(
        queue='reconsocket_image_dev',
        on_message_callback=callback_image,
        auto_ack=True
    )

    # Return the channel
    return channel


def rmq_unit_test(machine_id):
    """Test function for passing empty image"""

    # Establish path to script (in this directory)
    script_path = os.path.dirname(os.path.realpath(__file__))

    # Create and run command
    test_str = (
        sys.executable, ' ',
        str(script_path), '/rmqSendEmptyImage.py',
        ' -m ', machine_id, ' &'
    )
    os.system(''.join(test_str))


if __name__ == "__main__":

    # Define starting image number
    img_number = 0

    # Read config data
    config = read_config()
    # Get arguments
    args = get_args()

    # Setup channel to remote computer
    channel = rmq_setup_channel(args.m)

    # If -ut is passed, run a test
    if args.ut:
        rmq_unit_test(args.m)

    # Start listening
    print('[*] Waiting for images on', args.m, '. Type CTRL+C to exit')
    channel.start_consuming()
