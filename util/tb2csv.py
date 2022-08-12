"""Module for extracting TensorBoard logs to .csv files"""

import os
import sys
from typing import Union
import traceback
from glob import glob
import pandas as pd
import numpy as np
from tqdm import tqdm
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
        "images": 0,    # 0: load all
        "scalars": 0,
        "tensors": 0,
        "histograms": 1,
    }
    runlog_data = pd.DataFrame({"tag": [], "metric": [], "value": [], "step": []})
    metric = os.path.split(os.path.dirname(path))[-1]
    try:
        event_acc = EventAccumulator(path, DEFAULT_SIZE_GUIDANCE)
        event_acc.Reload()
        tags = event_acc.Tags()["tensors"]
        for tag in tags:
            event_list = event_acc.Tensors(tag)
            values = list(map(lambda x: float(tf.make_ndarray(x.tensor_proto)), event_list))
            step = list(map(lambda x: x.step, event_list))
            r = {
                "tag": [tag] * len(step),
                "metric": [metric] * len(step),
                "value": values, "step": step
            }
            r = pd.DataFrame(r)
            runlog_data = pd.concat([runlog_data, r])
    # Dirty catch of DataLossError
    except Exception:
        print("Event file possibly corrupt: {}".format(path))
        traceback.print_exc()
    return runlog_data


def sort_dataframe(df: pd.DataFrame):
    """Sort compound dataframe to create separate time-series"""

    # Retrieve unique tags and construct series names
    unique_tags = df["tag"].unique()
    series_names = [tag[20:] for tag in unique_tags]

    # Loop over tags and construct time series
    dfs_sort = []
    for tag in tqdm(unique_tags):
        # Retrieve part of dataframe from this time-series
        df_part = df.loc[df["tag"] == tag]

        # Retrieve all possible metrics and sort them in order
        metrics = df_part["metric"].unique()
        metrics.sort()
        # Retrieve all possible steps and sort them in order
        steps = df_part["step"].unique()
        steps.sort()

        # Loop over steps
        rows = []
        for step in steps:
            df_step = df_part.loc[df_part["step"] == step]
            values = [step]
            # Loop over metrics
            for metric in metrics:
                try:
                    value = df_step.loc[df_step["metric"] == metric]["value"]
                    if type(value) == float: values.append(value)
                    else: values.append(value.array[0])
                except Exception as e:
                    values.append(np.nan)

            # Append current row to rows
            rows.append(values)

        df_sort = pd.DataFrame(rows, columns=["step"] + list(metrics))

        dfs_sort.append(df_sort)

    return dfs_sort, series_names


def store_logs(
    from_dir: Union[str, os.PathLike],
    to_dir: Union[str, os.PathLike]
):
    """Retrieves tb logs from from_dir, extracts and stores in to_dir"""

    # Check wheter to_dir exists. if not, create it
    if not os.path.isdir(to_dir): os.mkdir(to_dir)

    # Retrieve event files from dir
    event_files = glob(os.path.join(from_dir, "*/events*"))

    # Initialize dataframe
    df = pd.DataFrame({"tag": [], "metric": [], "value": [], "step": []})

    # Extract event files and store in csv
    for event_file in event_files:
        # Verbose
        field = os.path.dirname(event_file).split("/")[-1]
        print(f"Extracting data from '{field}'...", end="", flush=True)

        # Extract data from event file
        if "/img/" not in event_file:
            df_file = tflog2pandas(event_file)
            df = pd.concat([df, df_file])
        else:
            pass
            # TODO: Retrieve images

        print(" " * (14 - len(field)) + "Done")

    # Store compound dataframe
    df.to_csv(os.path.join(to_dir, "logs.csv"), index=False)

    # Sort through dataframe to create separate dataframes for
    # each time series
    print("Sorting dataframes...")
    dfs_sort, series_names = sort_dataframe(df)

    # Store dataframes
    for i in range(len(series_names)):
        df_sort = dfs_sort[i]
        series_name = series_names[i]

        df_sort.to_csv(os.path.join(to_dir, series_name + ".csv"), index=False)


def main(log_dirs, to_dirs):
    """Main extractor function"""

    for i in range(len(log_dirs)):
        # Select directories
        log_dir = log_dirs[i]
        to_dir = to_dirs[i]

        # Verbose
        print("-----------------------------------")
        print(f"Extracting {log_dir}")
        print("-----------------------------------")

        # Extract logs from tb
        store_logs(log_dir, to_dir)


if __name__ == '__main__':

    # Add root to path
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if root not in sys.path: sys.path.append(root)

    # Setup log directory we wish to extract
    log_dirs = ["logs/final_snr_rdpg_scan"]
    to_dirs = ["tmp/tryout_logs"]

    main(log_dirs, to_dirs)
