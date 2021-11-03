"""Module used to store some data loggers

They are used to e.g. store some results or to save
model weights."""

import os
from typing import Union
import csv
import numpy as np


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
