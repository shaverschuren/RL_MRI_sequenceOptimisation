"""Module used to store some data loggers

They are used to e.g. store some results or to save
model weights.
"""

import os
from typing import Union
import csv
import numpy as np
import tensorflow as tf


class GeneralLogger():
    """Class used to represent a logger for general processes

    We use this logger to e.g. log the results during training.
    This data may be used to make some graphs at a later point.
    """

    def __init__(
            self,
            log_path: Union[str, bytes, os.PathLike],
            columns: list[str] = ["episode", "step", "snr", "fa"],
            mode: str = "w",
            seperator: str = ","):
        """Constructs attributes for this logger"""

        # Append attributes to self
        self.log_path = log_path
        self.columns = columns
        self.mode = mode
        self.seperator = seperator

        # Check validity of attributes
        if not os.path.isdir(os.path.dirname(log_path)):
            raise ValueError(
                "\nPassed logger path is invalid."
                f"\nReceived {log_path:s}"
            )
        if mode not in ["w", "a", "r"]:
            raise ValueError(
                "\nInvalid mode passed."
                f"\nGot '{mode:s}' but expected either "
                "'a' (append), 'w' (overwrite) or 'r' (read)."
            )

        # Initialize logs file
        self.init_file()

    def init_file(self):
        """Initialize logs file"""

        # Create file (if not already there)
        if not os.path.isfile(self.log_path):
            # Check whether we're not in read mode
            if self.mode == "r":
                raise ValueError(
                    "File doesn't exist. Nothing to read!"
                    f"\nTried: {self.log_path}"
                )
            else:
                # Generate first line
                first_line = ""
                for column in self.columns:
                    first_line += f"{column:s}{self.seperator:s}"
                first_line = first_line[:-1] + "\n"
                # Create file
                file = open(self.log_path, "w")
                file.write(first_line)
                file.close()
        else:
            # If overwrite mode, overwrite.
            # Otherwise, just append or read later.
            if self.mode == "w":
                # Remove file
                os.remove(self.log_path)
                # Generate first line
                first_line = ""
                for column in self.columns:
                    first_line += f"{column:s}{self.seperator:s}"
                first_line = first_line[:-1] + "\n"
                # Create file
                file = open(self.log_path, "w")
                file.write(first_line)
                file.close()

    def push(self, values: list[Union[int, float]]):
        """Push a line of values to the log file"""

        # Check whether the list of values is the same
        # size as the number of columns available
        if not len(values) == len(self.columns):
            raise ValueError(
                "\nWrong number of values passed!"
                f"\nExpected {len(self.columns)} but got {len(values)}."
            )

        # Generate line to add
        line = ""
        for value in values:
            line += f"{value}{self.seperator:s}"
        line = line[:-1] + "\n"

        # Add line to file
        file = open(self.log_path, "a")
        file.write(line)
        file.close()

    def pull(self):
        """Pull logs from file and return as list and `np.ndarray`"""

        with open(self.log_path, "r") as file:
            # Read file
            csv_reader = csv.reader(file, delimiter=self.seperator)

            # Loop over lines and fill columns and data
            line = 0
            columns = []
            data = np.array([[]])
            for row in csv_reader:
                if line == 0:
                    columns = list(row)
                else:
                    data = np.append(data, np.array((list(row))))

        return list(columns), np.array(data)


class TensorBoardLogger(object):
    """Class of logging object to log scalars and images to Tensorboard

    This logger may e.g. be used to log the trainings process.
    For more info on TensorBoard, we refer to
    https://www.tensorflow.org/tensorboard.
    """

    def __init__(
            self,
            log_dir,
            data_fields: Union[None, list[str]] = None):
        """Constructs attributes for this logger"""

        # Define list of datafields and writers. We may append this later
        self.writers = {}
        if data_fields:
            for field in data_fields:
                self.writers[field] = tf.summary.create_file_writer(
                    os.path.join(log_dir, field)
                )
        else:
            data_fields = []

        # Define attributes
        self.log_dir = log_dir
        self.data_fields = data_fields

        # Check log directory. If not there, mkdir
        if not os.path.isdir(log_dir):
            os.mkdir(log_dir)

    def add_datafields(self, fields):
        """Function to add datafields"""

        if type(fields) == list:
            for field in fields:
                # Append field to fields
                self.data_fields.append(field)
                # Append writer to writers
                self.writers[field] = tf.summary.create_file_writer(
                    os.path.join(self.log_dir, field)
                )
        if type(fields) == str:
            # Append field to fields
            self.data_fields.append(fields)
            # Append writer to writers
            self.writers[fields] = tf.summary.create_file_writer(
                os.path.join(self.log_dir, fields)
            )
        else:
            raise TypeError(
                "Type of 'fields' should be either list or str"
            )

    def log_scalar(self, field, tag, value, step):
        """Function to log scalar values"""

        # Check whether field exists
        if field not in self.writers.keys():
            raise ValueError(
                f"'{field}' is not a known datafield."
                f"\nExpected:{self.writers.keys()}"
            )

        # Write scalar to field
        with self.writers[field].as_default():
            # Write scalar to logs
            tf.summary.scalar(tag, value, step=step)
            # Flush logger
            self.writers[field].flush()

    def log_image(self, field, tag, image, step):
        """Function to log [2D] images"""

        # Check whether field exists
        if field not in self.writers.keys():
            raise ValueError(
                f"'{field}' is not a known datafield."
                f"\nExpected:{self.writers.keys()}"
            )

        # Write image to field
        with self.writers[field].as_default():
            # Write image(s) to logs
            tf.summary.image(
                tag, np.expand_dims(np.expand_dims(np.asarray(image), 0), -1),
                step=step
            )
            # Flush logger
            self.writers[field].flush()
