# src/visualize.py

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from pathlib import Path


def animate_scene(ego_state, others_state, ego_traj, others_trajs=None,
                  step_dt=0.1, save_path=None, fps=10):
    """
    Animate the ego vehicle and surrounding traffic.

    Surrounding vehicles follow constant-speed trajectories
    unless predicted trajectories are explicitly supplied.
    """

    T = ego_traj.shape[0]
    lane_w = 3.5
    n_lanes = 3

    # Build deterministic other trajectories if none given
    if others_trajs is None:
        others_trajs = []

        for ox0, oy0, ov0 in others_state:
            traj = []
            x = ox0

            for _ in range(T):
                x += ov0 * step_dt
                traj.append([x, oy0, ov0])

            others_trajs.append(np.array(traj))

    # Determine x-range for plotting
    max_x = ego_traj[:, 0].max()
    min_x = min(ego_state[0], ego_traj[:, 0].min())

    if others_trajs:
        max_x = max(max_x, max(ot[:, 0].max()
                              for ot in others_trajs))
        min_x = min(min_x, min(ot[:, 0].min()
                              for ot in others_trajs))

    xlim = (min_x - 5, max_x + 10)

    # Setup figure
    fig, ax = plt.subplots(figsize=(10, 4))

    ax.set_xlim(xlim)
    ax.set_ylim(-lane_w / 2 - 1,
                (n_lanes - 0.5) * lane_w + 1)

    ax.set_xlabel("Longitudinal position (m)")
    ax.set_ylabel("Lateral position (m)")
    ax.set_title("Autonomous Vehicle Trajectory Animation")

    # Draw lane boundaries
    for i in range(n_lanes + 1):
        boundary = (i - 0.5) * lane_w

        ax.axhline(
            boundary,
            color="gray",
            linestyle="-" if i in (0, n_lanes) else "--",
            linewidth=0.8,
            alpha=0.6
        )

    # Draw selected trajectory outline
    ax.plot(
        ego_traj[:, 0],
        ego_traj[:, 1],
        color="red",
        alpha=0.35,
        linewidth=2,
        label="Selected trajectory"
    )

    # Create ego and other vehicle patches
    ego_rect = plt.Rectangle(
        (0, 0), 4.5, 2.0,
        color="tab:red",
        alpha=0.9,
        label="Ego vehicle"
    )

    ax.add_patch(ego_rect)

    other_rects = []

    for i, _ in enumerate(others_trajs):
        r = plt.Rectangle(
            (0, 0), 4.5, 2.0,
            color="tab:gray",
            alpha=0.9,
            label="Other vehicles" if i == 0 else None
        )

        ax.add_patch(r)
        other_rects.append(r)

    time_text = ax.text(
        0.02, 0.95, "",
        transform=ax.transAxes
    )

    ax.legend(loc="upper right", fontsize=8)

    # Initial animation frame
    def init():
        ex0, ey0, _ = ego_traj[0]

        ego_rect.set_xy(
            (ex0 - 4.5 / 2, ey0 - 2.0 / 2)
        )

        for ot, r in zip(others_trajs, other_rects):
            ox0, oy0, _ = ot[0]

            r.set_xy(
                (ox0 - 4.5 / 2, oy0 - 2.0 / 2)
            )

        time_text.set_text(f"t = {step_dt:.2f} s")

        return [ego_rect] + other_rects + [time_text]

    # Frame update
    def animate_frame(t_idx):
        ex, ey, _ = ego_traj[t_idx]

        ego_rect.set_xy(
            (ex - 4.5 / 2, ey - 2.0 / 2)
        )

        for ot, r in zip(others_trajs, other_rects):
            ox, oy, _ = ot[t_idx]

            r.set_xy(
                (ox - 4.5 / 2, oy - 2.0 / 2)
            )

        time_text.set_text(
            f"t = {(t_idx + 1) * step_dt:.2f} s"
        )

        return [ego_rect] + other_rects + [time_text]

    # Build animation
    anim = animation.FuncAnimation(
        fig,
        animate_frame,
        init_func=init,
        frames=T,
        interval=1000 / fps,
        blit=False
    )

    plt.tight_layout()

    # Save animation before opening the display window
    if save_path is not None:
        output_path = Path(save_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            if output_path.suffix.lower() == ".gif":
                writer = animation.PillowWriter(fps=fps)

            elif output_path.suffix.lower() == ".mp4":
                writer = animation.FFMpegWriter(fps=fps)

            else:
                raise ValueError("Use a .gif or .mp4 extension.")

            anim.save(str(output_path), writer=writer)

            print(f"Saved animation to {output_path}")

        except Exception as e:
            print("Could not save animation:", e)

    plt.show()

    return anim
