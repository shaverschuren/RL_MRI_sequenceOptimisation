"""Module used for creating plots for the paper from .csv files"""

import os
import sys
import numpy as np
import pandas as pd
import pickle
import imutils
import matplotlib.pyplot as plt
import seaborn as sns
import tb2csv
import compare_with_standard


def plot_a_snr_train_loss(dirs):
    """Plot loss curve for experiment A (SNR)"""

    # Retrieve appropriate dataframe
    df_a_snr_train = pd.read_csv(
        os.path.join(dirs["to_dir"], "a_snr_train", "train_episodes.csv")
    )

    # Calculate rolling averages + CI bars
    df_a_snr_train['policy_loss_ave'] = df_a_snr_train.policy_loss.rolling(100).mean().shift(-99)
    df_a_snr_train['critic_loss_ave'] = df_a_snr_train.critic_loss.rolling(100).mean().shift(-99)
    df_a_snr_train["reward"] = df_a_snr_train["reward"] / 10.  # Average reward per step
    df_a_snr_train['reward_ave'] = df_a_snr_train.reward.rolling(100).mean().shift(-99)

    df_a_snr_train["policy_loss_res"] = abs(df_a_snr_train["policy_loss"] - df_a_snr_train["policy_loss_ave"]).rolling(50).mean().shift(-49)
    policy_loss_CI_max = df_a_snr_train['policy_loss_ave'] + df_a_snr_train["policy_loss_res"]
    policy_loss_CI_min = df_a_snr_train['policy_loss_ave'] - df_a_snr_train["policy_loss_res"]

    df_a_snr_train["critic_loss_res"] = abs(df_a_snr_train["critic_loss"] - df_a_snr_train["critic_loss_ave"]).rolling(50).mean().shift(-49)
    critic_loss_CI_max = df_a_snr_train['critic_loss_ave'] + df_a_snr_train["critic_loss_res"]
    critic_loss_CI_min = df_a_snr_train['critic_loss_ave'] - df_a_snr_train["critic_loss_res"]

    df_a_snr_train["reward_res"] = abs(df_a_snr_train["reward"] - df_a_snr_train["reward_ave"]).rolling(200).mean().shift(-199)
    reward_CI_max = (df_a_snr_train['reward_ave'] + df_a_snr_train["reward_res"]).rolling(100).mean().shift(-99)
    reward_CI_min = (df_a_snr_train['reward_ave'] - df_a_snr_train["reward_res"]).rolling(100).mean().shift(-99)

    # Create plots
    fig, ax = plt.subplots(figsize=(4, 3))
    plt.xlim(0, len(df_a_snr_train) - 400)
    plt.ylim(-8, 15)
    plt.ylabel(" ")
    plt.yticks([0], [0])
    plt.xlabel("Episodes")
    plt.xticks([0, len(df_a_snr_train) - 400], [0, "10k"])

    ax1 = sns.lineplot(x="step", y="policy_loss_ave", data=df_a_snr_train, label="Policy loss")
    ax2 = sns.lineplot(x="step", y="critic_loss_ave", data=df_a_snr_train, label="Critic loss")
    ax3 = sns.lineplot(x="step", y="reward_ave", data=df_a_snr_train, label="Reward")

    plt.fill_between(df_a_snr_train.step, policy_loss_CI_min, policy_loss_CI_max, alpha=.3)
    plt.fill_between(df_a_snr_train.step, critic_loss_CI_min, critic_loss_CI_max, alpha=.3)
    plt.fill_between(df_a_snr_train.step, reward_CI_min, reward_CI_max, alpha=.3)

    plt.axhline(y=0., color='k', linestyle='--', linewidth=.5)

    plt.legend(frameon=False)

    # Create and store figure
    plt.savefig(os.path.join(dirs["to_dir"], "a_snr_train.png"), bbox_inches="tight")
    plt.close()


def plot_a_cnr_train_loss(dirs):
    """Plot loss curve for experiment A (CNR)"""

    # Retrieve appropriate dataframe
    df_a_cnr_train = pd.read_csv(
        os.path.join(dirs["to_dir"], "a_cnr_train", "train_episodes.csv")
    )

    # Calculate rolling averages + CI bars
    df_a_cnr_train['policy_loss_ave'] = df_a_cnr_train.policy_loss.rolling(100).mean().shift(-99)
    df_a_cnr_train['critic_loss_ave'] = df_a_cnr_train.critic_loss.rolling(100).mean().shift(-99)
    df_a_cnr_train["reward"] = df_a_cnr_train["reward"] / 10.  # Average reward per step
    df_a_cnr_train['reward_ave'] = df_a_cnr_train.reward.rolling(100).mean().shift(-99)

    df_a_cnr_train["policy_loss_res"] = abs(df_a_cnr_train["policy_loss"] - df_a_cnr_train["policy_loss_ave"]).rolling(50).mean().shift(-49)
    policy_loss_CI_max = df_a_cnr_train['policy_loss_ave'] + df_a_cnr_train["policy_loss_res"]
    policy_loss_CI_min = df_a_cnr_train['policy_loss_ave'] - df_a_cnr_train["policy_loss_res"]

    df_a_cnr_train["critic_loss_res"] = abs(df_a_cnr_train["critic_loss"] - df_a_cnr_train["critic_loss_ave"]).rolling(50).mean().shift(-49)
    critic_loss_CI_max = df_a_cnr_train['critic_loss_ave'] + df_a_cnr_train["critic_loss_res"]
    critic_loss_CI_min = df_a_cnr_train['critic_loss_ave'] - df_a_cnr_train["critic_loss_res"]

    df_a_cnr_train["reward_res"] = abs(df_a_cnr_train["reward"] - df_a_cnr_train["reward_ave"]).rolling(200).mean().shift(-199)
    reward_CI_max = (df_a_cnr_train['reward_ave'] + df_a_cnr_train["reward_res"]).rolling(100).mean().shift(-99)
    reward_CI_min = (df_a_cnr_train['reward_ave'] - df_a_cnr_train["reward_res"]).rolling(100).mean().shift(-99)

    # Create plots
    fig, ax = plt.subplots(figsize=(4, 3))
    plt.xlim(0, len(df_a_cnr_train) - 400)
    plt.ylim(-12, 22.5)
    plt.ylabel(" ")
    plt.yticks([0], [0])
    plt.xlabel("Episodes")
    plt.xticks([0, len(df_a_cnr_train) - 400], [0, "10k"])

    ax1 = sns.lineplot(x="step", y="policy_loss_ave", data=df_a_cnr_train, label="Policy loss")
    ax2 = sns.lineplot(x="step", y="critic_loss_ave", data=df_a_cnr_train, label="Critic loss")
    ax3 = sns.lineplot(x="step", y="reward_ave", data=df_a_cnr_train, label="Reward")

    plt.fill_between(df_a_cnr_train.step, policy_loss_CI_min, policy_loss_CI_max, alpha=.3)
    plt.fill_between(df_a_cnr_train.step, critic_loss_CI_min, critic_loss_CI_max, alpha=.3)
    plt.fill_between(df_a_cnr_train.step, reward_CI_min, reward_CI_max, alpha=.3)

    plt.axhline(y=0., color='k', linestyle='--', linewidth=.5)

    plt.legend(frameon=False)


    # Create and store figure
    plt.savefig(os.path.join(dirs["to_dir"], "a_cnr_train.png"), bbox_inches="tight")
    plt.close()


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

    # print(df_snr[df_snr["step"] == 9]["snr_normMax"].mean())
    # print(df_snr[df_snr["step"] == 9]["snr_normMax"].std())

    # print(df_cnr[df_cnr["step"] == 9]["cnr_normMax"].mean())
    # print(df_cnr[df_cnr["step"] == 9]["cnr_normMax"].std())

    # Create plots
    fig, ax = plt.subplots(figsize=(8, 6))
    plt.xlim(-1, 9)
    plt.ylim(0.5, 1)
    plt.ylabel("Performance")  # ("Percentage of maximum SNR/CNR")
    plt.yticks([0.5, 0.75, 1], ["50%", "75%", "100%"])
    plt.xlabel("Timesteps")
    plt.xticks([-1, 9], [0, 10])

    ax1 = sns.lineplot(x="step", y="snr_normMax", data=df_snr, label="SNR: $\mu \pm \sigma$")
    ax2 = plt.plot(
        df_snr["step"].unique(), df_snr[df_snr["run"] == 4]["snr_normMax"],
        color="tab:blue", linestyle="dashed", linewidth=.8, label="SNR: example"
    )
    ax3 = sns.lineplot(x="step", y="cnr_normMax", data=df_cnr, label="CNR: $\mu \pm \sigma$")
    ax4 = plt.plot(
        df_cnr["step"].unique(), df_cnr[df_cnr["run"] == 3]["cnr_normMax"],
        color="tab:orange", linestyle="dashed", linewidth=.8, label="CNR: example"
    )

    plt.legend(bbox_to_anchor=(1., 0.25), frameon=False)

    # plt.axhline(y=1., color='k', linestyle='--', linewidth=.5)

    # Create and store figure
    plt.savefig(os.path.join(dirs["to_dir"], "a_test_metric.png"), bbox_inches="tight")
    plt.close()


def plot_a_snr_images(dirs):
    """Plot series of images for illustrative purposes (experiment A)"""

    # Create new dir if applicable
    to_img_dir = os.path.join(dirs["to_dir"], "a_snr_test", "imgs")
    if not os.path.isdir(to_img_dir): os.mkdir(to_img_dir)

    # Get data
    with open(os.path.join(dirs["to_dir"], "a_snr_test", "img.pickle"), 'rb') as f:
        imgs_dict = pickle.load(f)

    # Loop over tags
    for tag in imgs_dict.keys():
        # Retrieve name of tag
        name = tag[20:]

        # Retrieve image series and steps
        imgs = [row[0] for row in imgs_dict[tag]]
        steps = [row[1] for row in imgs_dict[tag]]

        # Concatenate images to side-by-side
        if len(imgs) > 1:
            # img_concat = np.concatenate([
            #     np.concatenate(imgs[:len(imgs) // 2], axis=1),
            #     np.concatenate(imgs[len(imgs) // 2 : -1], axis=1)
            # ], axis=0
            # )
            img_concat = np.concatenate([imgs[i] for i in range(len(imgs)) if i % 2 == 0], axis=1)
        else:
            img_concat = imgs[0]

        # Yield slightly more contrast intra- and inter-phantom via windowing
        img_concat = np.array(img_concat, dtype=np.float64)
        img_concat -= 40.  # np.percentile(img_concat, 55.)
        img_concat *= 256. / np.percentile(img_concat, 99.)
        img_concat = np.array(np.clip(img_concat, 0, 255), dtype=np.uint8)

        # Store image row
        plt.imshow(img_concat, cmap="gray")
        plt.axis('off')
        plt.savefig(os.path.join(to_img_dir, f"{name}.png"), dpi=500, bbox_inches="tight")
        plt.close()


def plot_a_cnr_images(dirs):
    """Plot series of images for illustrative purposes (experiment A)"""

    # Create new dir if applicable
    to_img_dir = os.path.join(dirs["to_dir"], "a_cnr_test", "imgs")
    if not os.path.isdir(to_img_dir): os.mkdir(to_img_dir)

    # Get data
    with open(os.path.join(dirs["to_dir"], "a_cnr_test", "img.pickle"), 'rb') as f:
        imgs_dict = pickle.load(f)

    # Loop over tags
    for tag in imgs_dict.keys():
        # Retrieve name of tag
        name = tag[20:]

        # Retrieve image series and steps
        imgs = [row[0] for row in imgs_dict[tag]]
        steps = [row[1] for row in imgs_dict[tag]]

        # Rotate and crop images
        imgs = [imutils.rotate(img, angle=-23.5)[95: 143, 93: 195].transpose() for img in imgs]

        # Concatenate images to side-by-side
        img_concat = np.concatenate(imgs, axis=1)

        # Yield slightly more contrast intra- and inter-phantom via windowing
        img_concat = np.array(img_concat, dtype=np.float64)
        img_concat -= 40.  # np.percentile(img_concat, 55.)
        img_concat *= 256. / np.percentile(img_concat, 99.)
        img_concat = np.array(np.clip(img_concat, 0, 255), dtype=np.uint8)

        # Store image row
        plt.imshow(img_concat, cmap="gray")
        plt.axis('off')
        plt.savefig(os.path.join(to_img_dir, f"{name}.png"), dpi=500, bbox_inches="tight")
        plt.close()


def plot_b_train_loss(dirs):
    """Plot loss curve for experiment A (CNR)"""

    # Retrieve appropriate dataframe
    df_b_train = pd.read_csv(
        os.path.join(dirs["to_dir"], "b_train", "train_episodes.csv")
    )

    # Calculate rolling averages + CI bars
    df_b_train['policy_loss_ave'] = df_b_train.policy_loss.rolling(100).mean().shift(-99)
    df_b_train['critic_loss_ave'] = df_b_train.critic_loss.rolling(100).mean().shift(-99)
    df_b_train["reward"] = df_b_train["reward"] / 10.
    df_b_train['reward_ave'] = df_b_train.reward.rolling(200).mean().shift(-199)

    df_b_train["policy_loss_res"] = abs(df_b_train["policy_loss"] - df_b_train["policy_loss_ave"]).rolling(50).mean().shift(-49)
    policy_loss_CI_max = df_b_train['policy_loss_ave'] + df_b_train["policy_loss_res"]
    policy_loss_CI_min = df_b_train['policy_loss_ave'] - df_b_train["policy_loss_res"]

    df_b_train["critic_loss_res"] = abs(df_b_train["critic_loss"] - df_b_train["critic_loss_ave"]).rolling(50).mean().shift(-49)
    critic_loss_CI_max = df_b_train['critic_loss_ave'] + df_b_train["critic_loss_res"]
    critic_loss_CI_min = df_b_train['critic_loss_ave'] - df_b_train["critic_loss_res"]

    df_b_train["reward_res"] = abs(df_b_train["reward"] - df_b_train["reward_ave"]).rolling(100).mean().shift(-99)
    reward_CI_max = (df_b_train['reward_ave'] + df_b_train["reward_res"]).rolling(200).mean().shift(-99)
    reward_CI_min = (df_b_train['reward_ave'] - df_b_train["reward_res"]).rolling(200).mean().shift(-99)

    # Create plots
    fig, ax = plt.subplots(figsize=(6, 4.5))
    plt.xlim(100, len(df_b_train) - 400)
    plt.ylim(-.8, 1)
    plt.ylabel(" ")
    plt.yticks([0], [0])
    plt.xlabel("Episodes")
    plt.xticks([100, len(df_b_train) - 400], [0, "10k"])

    ax1 = sns.lineplot(x="step", y="policy_loss_ave", data=df_b_train, label="Policy loss")
    ax2 = sns.lineplot(x="step", y="critic_loss_ave", data=df_b_train, label="Critic loss")
    ax3 = sns.lineplot(x="step", y="reward_ave", data=df_b_train, label="Reward")

    plt.fill_between(df_b_train.step, policy_loss_CI_min, policy_loss_CI_max, alpha=.3)
    plt.fill_between(df_b_train.step, critic_loss_CI_min, critic_loss_CI_max, alpha=.3)
    plt.fill_between(df_b_train.step, reward_CI_min, reward_CI_max, alpha=.3)

    plt.axhline(y=0., color='k', linestyle='--', linewidth=.5)
    # plt.axvline(x=8800, color='r', linestyle='--', linewidth=.5)

    plt.legend(frameon=False, bbox_to_anchor=(1., 1.))

    # Create and store figure
    plt.savefig(os.path.join(dirs["to_dir"], "b_train.png"), bbox_inches="tight")
    plt.close()


def plot_b_test(dirs):
    """Plot test curve for experiment B"""

    # Retrieve appropriate dataframes
    df = pd.concat([pd.read_csv(
        os.path.join(dirs["to_dir"], "b_test", f"test_episode_{i}.csv")
    ).assign(run=[i] * 11) for i in range(1, 73)]).reset_index()

    # Retrieve "standard" sequence data for comparison
    with open(os.path.join(dirs["to_dir"], "b_eval", "img.pickle"), "rb") as f:
        data_standard = pickle.load(f)
    cnrs_standard = np.array([data_standard[key][1] for key in data_standard.keys()])

    # print(np.mean(cnrs_standard))

    # Create new columns for plotting
    df["cnr_normMax"] = df["cnr"] / np.mean(cnrs_standard)  # Normalised by average performance of average sequence

    # Create plots
    fig, ax = plt.subplots(figsize=(6, 4.5))
    plt.xlim(-1, 9)
    plt.ylim(0.5, 1.25)
    plt.ylabel("Performance")  # ("Percentage of maximum cnr/CNR")
    plt.yticks([0.5, 0.75, 1.994 / np.mean(cnrs_standard), 1., 1.25], ["-50%", "-25%", "*", "-", "+25%"])
    plt.xlabel("Timesteps")
    plt.xticks([-1, 9], [0, 10])

    ax1 = sns.lineplot(x="step", y="cnr_normMax", data=df, color="gray", label="$\mu \pm \sigma$")
    ax2 = plt.plot(
        df["step"].unique(), df[df["run"] == 69]["cnr_normMax"],
        color="tab:blue", linestyle="dashed", linewidth=.8, label="Example #1"
    )
    ax2 = plt.plot(
        df["step"].unique(), df[df["run"] == 20]["cnr_normMax"],
        color="tab:orange", linestyle="dashed", linewidth=.8, label="Example #2"
    )

    print(df[df["step"] == -1]["cnr_normMax"].mean())
    print(df[df["step"] == -1]["cnr_normMax"].std())

    plt.legend(frameon=False)

    plt.axhline(y=1., color='k', linestyle='--', linewidth=.5)
    plt.axhline(y=1.994 / np.mean(cnrs_standard), color='gray', linestyle='--', linewidth=.5)

    # Create and store figure
    plt.savefig(os.path.join(dirs["to_dir"], "b_test_metric.png"), bbox_inches="tight")
    plt.close()


def plot_b_pulsetrain(dirs):
    """Plot pulsetrain curves for experiment B"""

    # Retrieve appropriate dataframes
    df_theta_1 = pd.read_csv(
        os.path.join(dirs["to_dir"], "b_test", f"test_theta_episode_69.csv")
    )
    df_Mz_1 = pd.read_csv(
        os.path.join(dirs["to_dir"], "b_test", f"test_Mz_episode_69.csv")
    )
    df_F0_1 = pd.read_csv(
        os.path.join(dirs["to_dir"], "b_test", f"test_F0_episode_69.csv")
    )
    df_theta_2 = pd.read_csv(
        os.path.join(dirs["to_dir"], "b_test", f"test_theta_episode_71.csv")
    )
    df_Mz_2 = pd.read_csv(
        os.path.join(dirs["to_dir"], "b_test", f"test_Mz_episode_71.csv")
    )
    df_F0_2 = pd.read_csv(
        os.path.join(dirs["to_dir"], "b_test", f"test_F0_episode_71.csv")
    )

    # Do some required calculations
    df_F0_1["diff"] = abs(df_F0_1["F0_a"] - df_F0_1["F0_b"])
    df_F0_2["diff"] = abs(df_F0_2["F0_a"] - df_F0_2["F0_b"])

    # ########## Create plot 1 ############
    fig, ax = plt.subplots(figsize=(4, 3))
    plt.xlim(-5, 63)
    plt.ylim(0., .2)
    plt.ylabel("Signal [a.u.]")  # ("Percentage of maximum cnr/CNR")
    plt.yticks([0], [0])

    plt.xlabel("Pulses")
    plt.xticks([0, 63], [0, "64"])

    ax1 = sns.lineplot(x="step", y="F0_a", data=df_F0_1, color="tab:blue", label="Example #1 - $\mu_{F0}$ (GM)")
    ax2 = sns.lineplot(x="step", y="F0_b", data=df_F0_1, color="tab:blue", linestyle="dashed", label="Example #1 - $\mu_{F0}$ (WM)")
    ax3 = sns.lineplot(x="step", y="F0_a", data=df_F0_2, color="tab:orange", label="Example #2 - $\mu_{F0}$ (GM)")
    ax4 = sns.lineplot(x="step", y="F0_b", data=df_F0_2, color="tab:orange", linestyle="dashed", label="Example #2 - $\mu_{F0}$ (WM)")

    plt.legend(frameon=False)

    plt.axvline(x=0., color='k', linestyle='--', linewidth=.5)

    # Create and store figure
    plt.savefig(os.path.join(dirs["to_dir"], "b_test_F0.png"), bbox_inches="tight")
    plt.close()

    # ########## Create plot 2 ############
    fig, ax = plt.subplots(figsize=(4, 3))
    plt.xlim(-5, 63)
    plt.ylim(0., 0.2)
    plt.ylabel("$M_z^-$ [fraction of $M_0$]") 
    plt.yticks([0, 0.2], [0, "0.2"])

    plt.xlabel("Pulses")
    plt.xticks([0, 63], [0, "64"])

    ax1 = sns.lineplot(x="step", y="Mz_a", data=df_Mz_1, color="tab:blue", label="Example #1 - $\mu_{Mz}$ (GM)")
    ax2 = sns.lineplot(x="step", y="Mz_b", data=df_Mz_1, color="tab:blue", linestyle="dashed", label="Example #1 - $\mu_{Mz}$ (WM)")
    ax3 = sns.lineplot(x="step", y="Mz_a", data=df_Mz_2, color="tab:orange", label="Example #2 - $\mu_{Mz}$ (GM)")
    ax4 = sns.lineplot(x="step", y="Mz_b", data=df_Mz_2, color="tab:orange", linestyle="dashed", label="Example #2 - $\mu_{Mz}$ (WM)")

    plt.legend(frameon=False)

    plt.axvline(x=0., color='k', linestyle='--', linewidth=.5)

    # Create and store figure
    plt.savefig(os.path.join(dirs["to_dir"], "b_test_Mz.png"), bbox_inches="tight")
    plt.close()

    # ########## Create plot 3 ############

    # Get average pulsetrain
    df_test = pd.concat([pd.read_csv(
        os.path.join("tmp/figures_for_publication", "b_test", f"test_theta_episode_{i}.csv")
    ).assign(run=[i] * 69) for i in range(1, 73)]).reset_index()

    theta_avg = []
    theta_std = []
    for step in range(-5, 64):
        df_step = df_test[df_test["step"] == step]

        alpha = df_step["theta"].mean()
        std = df_step["theta"].std()
        theta_avg.append(alpha)
        theta_std.append(std)
    
    theta_avg = np.array(theta_avg)
    theta_std = np.array(theta_std)

    CI_max = theta_avg + theta_std
    CI_min = theta_avg - theta_std

    fig, ax = plt.subplots(figsize=(4, 3))
    plt.xlim(-5, 63)
    plt.ylim(70, 95)
    plt.ylabel("Flip angle [deg]")  # ("Percentage of maximum cnr/CNR")
    plt.yticks([70, 80, 90], [70, 80, 90])

    plt.xlabel("Pulses")
    plt.xticks([0, 63], [0, "64"])

    ax3 = plt.plot(df_theta_1["step"].unique(), theta_avg, color="tab:green", label="Baseline ($\pm \sigma$)")
    ax1 = sns.lineplot(x="step", y="theta", data=df_theta_1, color="tab:blue", linestyle="dashed", label="Example #1")
    ax2 = sns.lineplot(x="step", y="theta", data=df_theta_2, color="tab:orange", linestyle="dashed", label="Example #2")

    plt.fill_between(df_theta_1["step"].unique(), CI_min, CI_max, color="tab:green", alpha=.3)

    plt.legend(frameon=False)

    plt.axvline(x=0., color='k', linestyle='--', linewidth=.5)

    # Create and store figure
    plt.savefig(os.path.join(dirs["to_dir"], "b_test_theta.png"), bbox_inches="tight")
    plt.close()


def plot_b_images(dirs):
    """Plot series of images for illustrative purposes (experiment B)"""

    # Create new dir if applicable
    to_img_dir = os.path.join(dirs["to_dir"], "b_test", "imgs")
    if not os.path.isdir(to_img_dir): os.mkdir(to_img_dir)

    # Get data
    with open(os.path.join(dirs["to_dir"], "b_test", "img.pickle"), 'rb') as f:
        imgs_dict = pickle.load(f)

    # Loop over tags
    for tag in imgs_dict.keys():
        # Retrieve name of tag
        name = tag[20:]

        # Retrieve image series and steps
        imgs = [row[0] for row in imgs_dict[tag]]
        steps = [row[1] for row in imgs_dict[tag]]

        # Window images
        imgs = [(img[:, 7:-7] - img.min()) * 256. / (img.max() - img.min()) for img in imgs]

        # Concatenate images to side-by-side
        if len(imgs) > 1:
            img_concat = np.concatenate(imgs, axis=1)
        else:
            img_concat = np.concatenate(imgs, axis=1)

        # Yield slightly more contrast intra- and inter-phantom via windowing
        img_concat = np.array(img_concat, dtype=np.float64)
        # img_concat -= 40.  # np.percentile(img_concat, 55.)
        # img_concat *= 256. / np.percentile(img_concat, 99.)
        img_concat = np.array(np.clip(img_concat, 0, 255), dtype=np.uint8)

        # Store image row
        plt.imshow(img_concat, cmap="gray")
        plt.axis('off')
        plt.savefig(os.path.join(to_img_dir, f"{name}.png"), dpi=500, bbox_inches="tight")
        plt.close()


def plot_eval_images(dirs):
    """Plot evaluation images"""

    # Create new dir if applicable
    to_img_dir = os.path.join(dirs["to_dir"], "b_eval", "imgs")
    if not os.path.isdir(to_img_dir): os.mkdir(to_img_dir)

    # Get data
    with open(os.path.join(dirs["to_dir"], "b_eval", "img.pickle"), 'rb') as f:
        imgs_dict = pickle.load(f)

    # Loop over tags
    for tag in imgs_dict.keys():
        # Retrieve name of tag
        name = str(tag)

        # Retrieve image series and steps
        img = imgs_dict[tag][0]
        cnr = imgs_dict[tag][1]

        # Window images
        img = (img - img.min()) * 256. / (img.max() - img.min())

        # Yield slightly more contrast intra- and inter-phantom via windowing
        # img_concat = np.array(img, dtype=np.float64)
        # img_concat -= 40.  # np.percentile(img_concat, 55.)
        # img_concat *= 256. / np.percentile(img_concat, 99.)
        img = np.array(np.clip(img, 0, 255), dtype=np.uint8)

        # Store image row
        plt.imshow(img, cmap="gray")
        plt.axis('off')
        plt.title(f"{tag} - CNR={cnr:.2f}")
        plt.savefig(os.path.join(to_img_dir, f"{name}.png"), dpi=500, bbox_inches="tight")
        plt.close()


def generate_plots(dirs: dict[str, str]):
    """Generate plots"""

    # Training curves experiment A - SNR & CNR
    plot_a_snr_train_loss(dirs)
    plot_a_cnr_train_loss(dirs)

    # Test curve experiment A
    plot_a_metrics_test(dirs)

    # Images experiment A - SNR & CNR
    plot_a_snr_images(dirs)
    plot_a_cnr_images(dirs)

    # Training curve B
    plot_b_train_loss(dirs)

    # Test curve B
    plot_b_test(dirs)

    # Theta/F0/Mz curves
    plot_b_pulsetrain(dirs)

    # Images experiment B
    plot_b_images(dirs)

    # Images experiment B evaluation
    plot_eval_images(dirs)


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
    src = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    root = os.path.dirname(src)
    if root not in sys.path: sys.path.append(root)

    # Setup directories we wish to get the data from
    dirs = {
        "a_snr_train": "logs/final_a_snr_rdpg",
        "a_snr_test": "logs/final_a_snr_rdpg_scan",
        "a_cnr_train": "logs/final_a_cnr_rdpg",
        "a_cnr_test": "logs/final_a_cnr_rdpg_scan",
        "b_train": "logs/final_b",
        "b_test": "logs/final_b",
        "to_dir": "tmp/figures_for_publication"
    }

    # Setup output directory for our figures
    if not os.path.isdir(dirs["to_dir"]): os.mkdir(dirs["to_dir"])

    # Extract tensorboard event files
    extract_tb_logs(dirs)

    # Generate comparison data
    compare_with_standard.main()

    # Generate plots
    generate_plots(dirs)
