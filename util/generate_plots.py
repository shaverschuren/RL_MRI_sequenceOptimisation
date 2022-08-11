import os
import sys
from typing import Union
import traceback
from glob import glob
import pandas as pd
import tensorflow as tf
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def tflog2pandas(path: str) -> pd.DataFrame:
    """convert single tensorflow log file to pandas DataFrame
    Parameters
    ----------
    path : str
        path to tensorflow log file
    Returns
    -------
    pd.DataFrame
        converted dataframe
    """
    DEFAULT_SIZE_GUIDANCE = {
        "compressedHistograms": 1,
        "images": 1,
        "scalars": 0,  # 0 means load all
        "histograms": 1,
    }
    runlog_data = pd.DataFrame({"tag": [], "value": [], "step": []})
    try:
        event_acc = EventAccumulator(path, DEFAULT_SIZE_GUIDANCE)
        event_acc.Reload()
        tags = event_acc.Tags()["tensors"]
        for tag in tags:
            event_list = event_acc.Tensors(tag)
            values = list(map(lambda x: float(tf.make_ndarray(x.tensor_proto)), event_list))
            step = list(map(lambda x: x.step, event_list))
            r = {"tag": [tag] * len(step), "value": values, "step": step}
            r = pd.DataFrame(r)
            runlog_data = pd.concat([runlog_data, r])
    # Dirty catch of DataLossError
    except Exception:
        print("Event file possibly corrupt: {}".format(path))
        traceback.print_exc()
    return runlog_data


def store_logs(
    from_dir: Union[str, os.PathLike],
    to_dir: Union[str, os.PathLike]
):
    """Retrieves tb logs from from_dir, extracts and stores in to_dir"""

    # Check wheter to_dir exists. if not, create it
    if not os.path.isdir(to_dir): os.mkdir(to_dir)

    # Retrieve event files from dir
    event_files = glob(os.path.join(from_dir, "*/events*"))

    # Extract event files and store in csv
    for event_file in event_files:
        field = os.path.dirname(event_file).split("/")[-1]

        print(f"Extracting data from '{field}'...", end="", flush=True)

        df = tflog2pandas(event_file)
        df.to_csv(os.path.join(to_dir, field + ".csv"))

        print(" Done")


if __name__ == '__main__':

    # Add root to path
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if root not in sys.path: sys.path.append(root)

    # Setup log directory we wish to extract
    log_dir = "logs/final_snr_rdpg_scan"
    to_dir = "tmp/tryout_logs"
    store_logs(log_dir, to_dir)
