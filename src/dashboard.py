# src/dashboard.py

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from env import HighwayEnv
from trajectories import generate_trajectories
from risk import estimate_risk_for_all
from planner import select_best_trajectory, compute_score
from visualize import animate_scene

DT = 0.1
HORIZON = 3.0
N_SAMPLES = 50

TRAJECTORY_NAMES = [
    "Keep Lane",
    "Brake",
    "Lane Left",
    "Lane Right"
]


def run_planning_pipeline(seed=None):
    """Run the environment, trajectory, risk and planning pipeline."""

    # Optional seed for reproducible demonstrations
    if seed is not None:
        np.random.seed(seed)

    rng = np.random.default_rng(seed)

    env = HighwayEnv()
    ego_state, others_state = env.reset()

    ego_trajs = generate_trajectories(
        ego_state,
        dt=DT,
        horizon=HORIZON
    )

    risks = estimate_risk_for_all(
        ego_trajs,
        others_initial_state=others_state,
        n_samples=N_SAMPLES,
        dt=DT,
        horizon=HORIZON,
        rng=rng
    )

    scores = [
        compute_score(r, ego_trajs[i])
        for i, r in enumerate(risks)
    ]

    best_idx, _ = select_best_trajectory(
        ego_trajs,
        risks
    )

    return (
        ego_state,
        others_state,
        ego_trajs,
        risks,
        scores,
        best_idx
    )


def plot_dashboard(ego_state, others_state, ego_trajs,
                   risks, scores, best_idx, save_path=None):
    """Create a four-panel planning dashboard."""

    best_traj = ego_trajs[best_idx]

    # Trajectory states begin after the first simulation step
    time = (np.arange(best_traj.shape[0]) + 1) * DT

    fig, axs = plt.subplots(2, 2, figsize=(12, 8))
    axs = axs.flatten()

    fig.suptitle(
        "Risk-Aware Highway Trajectory Planner\n"
        f"Selected Manoeuvre: {TRAJECTORY_NAMES[best_idx]}",
        fontsize=14,
        fontweight="bold"
    )

    # ---------------------------------------------
    # Panel 1 - Speed profile
    # ---------------------------------------------

    axs[0].plot(
        time,
        best_traj[:, 2],
        color="tab:red",
        linewidth=2
    )

    axs[0].set_title("Ego Speed Profile")
    axs[0].set_xlabel("Time (s)")
    axs[0].set_ylabel("Speed (m/s)")
    axs[0].grid(True, alpha=0.3)

    # ---------------------------------------------
    # Panel 2 - Risk summary
    # ---------------------------------------------

    cp = risks[best_idx]["collision_prob"]
    avgd = risks[best_idx]["avg_min_distance"]
    worst = risks[best_idx]["worst_min_distance"]

    bars = axs[1].bar(
        ["Average Min Distance", "Worst Min Distance"],
        [avgd, worst],
        color=["tab:blue", "tab:green"]
    )

    axs[1].set_title(
        "Risk Indicators (Chosen Trajectory)\n"
        f"Estimated Collision Probability: {cp:.1%}"
    )

    axs[1].set_ylabel("Center-to-center distance (m)")
    axs[1].set_ylim(0, max(avgd, worst, 1.0) * 1.2)

    for bar in bars:
        height = bar.get_height()

        axs[1].annotate(
            f"{height:.2f} m",
            xy=(
                bar.get_x() + bar.get_width() / 2,
                height
            ),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom"
        )

    axs[1].grid(axis="y", alpha=0.3)

    # ---------------------------------------------
    # Panel 3 - Planner scores
    # ---------------------------------------------

    bar_colors = ["tab:gray"] * len(scores)
    bar_colors[best_idx] = "tab:red"

    bars = axs[2].bar(
        TRAJECTORY_NAMES,
        scores,
        color=bar_colors
    )

    axs[2].set_title("Trajectory Scores (Lower Is Better)")
    axs[2].set_ylabel("Score")
    axs[2].set_xlabel("Candidate Manoeuvre")

    axs[2].set_ylim(
        0,
        max(max(scores), 0.001) * 1.2
    )

    for bar in bars:
        height = bar.get_height()

        axs[2].annotate(
            f"{height:.3f}",
            xy=(
                bar.get_x() + bar.get_width() / 2,
                height
            ),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=9
        )

    axs[2].grid(axis="y", alpha=0.3)

    # ---------------------------------------------
    # Panel 4 - Initial traffic state
    # ---------------------------------------------

    axs[3].set_title("Top-Down Snapshot (Initial State)")
    axs[3].set_xlabel("Longitudinal position (m)")
    axs[3].set_ylabel("Lateral position (m)")

    lane_w = 3.5
    n_lanes = 3

    # Draw corrected lane boundaries
    for i in range(n_lanes + 1):
        boundary = (i - 0.5) * lane_w

        axs[3].axhline(
            boundary,
            color="gray",
            linestyle="-" if i in (0, n_lanes) else "--",
            linewidth=0.8,
            alpha=0.6
        )

    axs[3].set_ylim(
        -lane_w / 2 - 1,
        (n_lanes - 0.5) * lane_w + 1
    )

    # Ego vehicle
    ex0, ey0, _ = ego_state

    axs[3].scatter(
        [ex0],
        [ey0],
        color="tab:red",
        s=60,
        label="Ego vehicle"
    )

    # Surrounding vehicles
    for i, (ox, oy, _) in enumerate(others_state):

        axs[3].scatter(
            [ox],
            [oy],
            color="tab:gray",
            s=40,
            label="Other vehicles" if i == 0 else None
        )

    axs[3].legend(loc="upper right", fontsize=9)

    plt.tight_layout(rect=[0, 0, 1, 0.93])

    # Optional dashboard export
    if save_path is not None:
        output_path = Path(save_path)
        output_path.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        fig.savefig(
            output_path,
            dpi=150,
            bbox_inches="tight"
        )

        print(f"Dashboard saved to {output_path}")

    plt.show()

    return fig


def run_dashboard(seed=None):
    """Run the planner, animation and dashboard."""

    (
        ego_state,
        others_state,
        ego_trajs,
        risks,
        scores,
        best_idx
    ) = run_planning_pipeline(seed=seed)

    print("Scores:", scores)

    print(
        "Chosen manoeuvre:",
        TRAJECTORY_NAMES[best_idx]
    )

    # Open animation window
    animate_scene(
        ego_state,
        others_state,
        ego_trajs[best_idx],
        step_dt=DT
    )

    # Open dashboard window
    plot_dashboard(
        ego_state,
        others_state,
        ego_trajs,
        risks,
        scores,
        best_idx
    )


if __name__ == "__main__":
    run_dashboard()
