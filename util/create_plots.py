"""Module used for creating plots for the paper from .csv files"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tb2csv


def plot_a_snr_train_loss(dirs):
    """Plot loss curve for experiment A (SNR)"""

    # Retrieve appropriate dataframe
    df_a_snr_train = pd.read_csv(
        os.path.join(dirs["to_dir"], "a_snr_train", "train_episodes.csv")
    )

    # Calculate rolling averages + CI bars
    df_a_snr_train['policy_loss_ave'] = df_a_snr_train.policy_loss.rolling(100).mean().shift(-99)
    df_a_snr_train['critic_loss_ave'] = df_a_snr_train.critic_loss.rolling(100).mean().shift(-99)

    df_a_snr_train["policy_loss_res"] = abs(df_a_snr_train["policy_loss"] - df_a_snr_train["policy_loss_ave"]).rolling(50).mean().shift(-49)
    policy_loss_CI_max = df_a_snr_train['policy_loss_ave'] + df_a_snr_train["policy_loss_res"]
    policy_loss_CI_min = df_a_snr_train['policy_loss_ave'] - df_a_snr_train["policy_loss_res"]

    df_a_snr_train["critic_loss_res"] = abs(df_a_snr_train["critic_loss"] - df_a_snr_train["critic_loss_ave"]).rolling(50).mean().shift(-49)
    critic_loss_CI_max = df_a_snr_train['critic_loss_ave'] + df_a_snr_train["critic_loss_res"]
    critic_loss_CI_min = df_a_snr_train['critic_loss_ave'] - df_a_snr_train["critic_loss_res"]

    # Create plots
    fig, ax = plt.subplots()
    plt.xlim(0, len(df_a_snr_train))
    plt.ylabel("Loss")
    plt.yticks([0], [0])
    plt.xlabel("Episodes")
    plt.xticks([0, len(df_a_snr_train)], [0, len(df_a_snr_train)])
    # a_snr_train = sns.lineplot("step", "value", hue="variable", data=pd.melt(df_a_snr_train, "step"))
    ax1 = sns.lineplot(x="step", y="policy_loss_ave", data=df_a_snr_train)
    ax2 = sns.lineplot(x="step", y="critic_loss_ave", data=df_a_snr_train)

    plt.fill_between(df_a_snr_train.step, policy_loss_CI_min, policy_loss_CI_max, alpha=.3)
    plt.fill_between(df_a_snr_train.step, critic_loss_CI_min, critic_loss_CI_max, alpha=.3)

    plt.axhline(y=0., color='k', linestyle='--', linewidth=.5)

    # Create and store figure
    # fig = a_snr_train.get_figure()
    fig.savefig(os.path.join(dirs["to_dir"], "a_snr_train.png"))


def plot_a_cnr_train_loss(dirs):
    """Plot loss curve for experiment A (CNR)"""

    # Retrieve appropriate dataframe
    df_a_cnr_train = pd.read_csv(
        os.path.join(dirs["to_dir"], "a_cnr_train", "train_episodes.csv")
    )

    # Calculate rolling averages + CI bars
    df_a_cnr_train['policy_loss_ave'] = df_a_cnr_train.policy_loss.rolling(100).mean().shift(-99)
    df_a_cnr_train['critic_loss_ave'] = df_a_cnr_train.critic_loss.rolling(100).mean().shift(-99)

    df_a_cnr_train["policy_loss_res"] = abs(df_a_cnr_train["policy_loss"] - df_a_cnr_train["policy_loss_ave"]).rolling(50).mean().shift(-49)
    policy_loss_CI_max = df_a_cnr_train['policy_loss_ave'] + df_a_cnr_train["policy_loss_res"]
    policy_loss_CI_min = df_a_cnr_train['policy_loss_ave'] - df_a_cnr_train["policy_loss_res"]

    df_a_cnr_train["critic_loss_res"] = abs(df_a_cnr_train["critic_loss"] - df_a_cnr_train["critic_loss_ave"]).rolling(50).mean().shift(-49)
    critic_loss_CI_max = df_a_cnr_train['critic_loss_ave'] + df_a_cnr_train["critic_loss_res"]
    critic_loss_CI_min = df_a_cnr_train['critic_loss_ave'] - df_a_cnr_train["critic_loss_res"]

    # Create plots
    fig, ax = plt.subplots()
    plt.xlim(0, len(df_a_cnr_train))
    plt.ylabel("Loss")
    plt.yticks([0], [0])
    plt.xlabel("Episodes")
    plt.xticks([0, len(df_a_cnr_train)], [0, len(df_a_cnr_train)])
    # a_snr_train = sns.lineplot("step", "value", hue="variable", data=pd.melt(df_a_snr_train, "step"))
    ax1 = sns.lineplot(x="step", y="policy_loss_ave", data=df_a_cnr_train)
    ax2 = sns.lineplot(x="step", y="critic_loss_ave", data=df_a_cnr_train)

    plt.fill_between(df_a_cnr_train.step, policy_loss_CI_min, policy_loss_CI_max, alpha=.3)
    plt.fill_between(df_a_cnr_train.step, critic_loss_CI_min, critic_loss_CI_max, alpha=.3)

    plt.axhline(y=0., color='k', linestyle='--', linewidth=.5)

    # Create and store figure
    # fig = a_snr_train.get_figure()
    fig.savefig(os.path.join(dirs["to_dir"], "a_cnr_train.png"))


def plot_a_metrics_test(dirs):
    """Plot loss curve for experiment A (CNR)"""

    # Retrieve appropriate dataframes
    df_snr = pd.concat([pd.read_csv(
        os.path.join(dirs["to_dir"], "a_snr_test", f"test_episode_{i}.csv")
    ).assign(run=[i] * 11) for i in range(1, 11)]).reset_index()
    df_cnr = pd.concat([pd.read_csv(
        os.path.join(dirs["to_dir"], "a_cnr_test", f"test_episode_{i}.csv")
    ).assign(run=[i] * 11) for i in range(1, 11)]).reset_index()

    # Create new columns for plotting
    df_snr["snr_normMax"] = df_snr["snr"] / max(df_snr["snr"])
    df_cnr["cnr_normMax"] = df_cnr["cnr"] / max(df_cnr["cnr"])

    # Create plots
    fig, ax = plt.subplots()
    plt.xlim(-1, 9)
    plt.ylim(0.5, 1)
    plt.ylabel("Percentage of maximum SNR/CNR")
    plt.yticks([0.5, 0.75, 1], ["50%", "75%", "100%"])
    plt.xlabel("Timesteps")
    plt.xticks([-1, 9], [0, "T"])

    ax1 = sns.lineplot(x="step", y="snr_normMax", data=df_snr, label="SNR")
    ax2 = sns.lineplot(x="step", y="cnr_normMax", data=df_cnr, label="CNR")

    plt.legend(bbox_to_anchor=(0.95, 0.2))

    # plt.axhline(y=1., color='k', linestyle='--', linewidth=.5)

    # Create and store figure
    # fig = a_snr_train.get_figure()
    fig.savefig(os.path.join(dirs["to_dir"], "a_test_metric.png"))


def generate_plots(dirs: dict[str, str]):
    """Generate plots"""

    # Training curves A - SNR
    plot_a_snr_train_loss(dirs)

    # Training curves A - CNR
    plot_a_cnr_train_loss(dirs)

    # Test curve A
    plot_a_metrics_test(dirs)

    # Training curve B
    # df_b_train = pd.read_csv(
    #     os.path.join(dirs["to_dir"], "b_train", "train_episodes.csv")
    # )

    # Test curves A - SNR

    # Test curves A - CNR


def extract_tb_logs(dirs: dict[str, str], overwrite=False):
    """Extract appropriate log files"""

    # Define dirnames
    dirnames = [
        "a_snr_train", "a_snr_test", "a_cnr_train",
        "a_cnr_test", "b_train", "b_test"
    ]

    # Extract proper event files (if not already done)
    tb2csv.main(
        [
            dirs[dirname] for dirname in dirnames
            if not os.path.isdir(os.path.join(dirs["to_dir"], dirname)) or overwrite
        ],
        [
            os.path.join(dirs["to_dir"], dirname) for dirname in dirnames
            if not os.path.isdir(os.path.join(dirs["to_dir"], dirname)) or overwrite
        ]
    )


if __name__ == '__main__':
    # Add root to path
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if root not in sys.path: sys.path.append(root)

    # Setup directories we wish to get the data from
    dirs = {
        "a_snr_train": "logs/final_a_snr_rdpg",
        "a_snr_test": "logs/final_a_snr_rdpg_scan",
        "a_cnr_train": "logs/final_a_cnr_rdpg",
        "a_cnr_test": "logs/final_a_cnr_rdpg_scan",
        "b_train": "logs/final_b_cnr",
        "b_test": "logs/final_b_cnr",
        "to_dir": "tmp/figures_for_publication"
    }

    # Setup output directory for our figures
    if not os.path.isdir(dirs["to_dir"]): os.mkdir(dirs["to_dir"])

    # Extract tensorboard event files
    extract_tb_logs(dirs)

    # Generate plots
    generate_plots(dirs)
