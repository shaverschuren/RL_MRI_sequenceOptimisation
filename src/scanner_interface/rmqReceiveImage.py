"""Module implementing an image receiver for scanner-interface

This is a necessary step in the communication between the
MRI scanner and our PyTorch model. We use an RT-rabbit based
framework.
"""

# File-specific imports
import os
import sys
import numpy as np
import h5py
import threading
import time
import pika
import dataobject_pb2 as pb  # (pb = protobuf)


def set_global_vars(img_number_, args_, config_, channel_):
    """Set global variables"""

    global img_number, args, config, channel

    img_number = img_number_
    args = args_
    config = config_
    channel = channel_


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
        str(script_path), '/tests/rmqSendEmptyImage.py',
        ' -m ', machine_id, ' &'
    )
    os.system(''.join(test_str))


def caught_channel_consumption():
    """Start channel consumption but catch errors"""

    # Define global vars
    global channel

    try:
        channel.start_consuming()
    except Exception:
        pass


def clear_channel_queue():
    """Clear the image queue for the connected channel"""

    # Declare global vars
    global channel, img_number, config

    # Print some info
    print("Clearing scanner interface queue...")

    # Call on queue, wait for a moment, then close it again
    # We use multithreading for this (can't close if running on same thread)
    channel_thread = threading.Thread(
        name="clear_channel_queue",
        target=caught_channel_consumption,
        daemon=True
    )

    channel_thread.start()

    # Wait for some time, then close the channel and kill the thread
    time.sleep(2)
    try:
        channel.close()
    except Exception:
        pass
    channel_thread.join()

    # Remove the created data file
    if os.path.exists(config["data_loc"]): os.remove(config["data_loc"])

    # Delete its redundant output
    for _ in range(img_number):
        print("\033[A                             \033[A")
    # Print some info
    print(f"Found and removed {img_number} image(s).")

    # Reset img_number
    img_number = 0


def close_channel():
    """Close channel"""

    global channel

    try:
        channel.close()
    except Exception:
        print("Channel closed")
