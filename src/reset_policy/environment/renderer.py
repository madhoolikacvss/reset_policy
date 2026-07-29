"""
Visualization for Reset Policy environment.

Displays:
- board
- cube position
- occupancy grid
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np


class BoardRenderer:

    def __init__(self,board_width=30,board_height=20,cell_size=1,):

        self.width = board_width
        self.height = board_height
        self.cell_size = cell_size

        # Interactive mode
        plt.ion()

        # Create ONE persistent figure
        self.fig, self.ax = plt.subplots(figsize=(8, 5))

    def render(self,cube_x,cube_y,occupancy_grid=None,coverage=None,action=None,mode="human",):

        ax = self.ax
        fig = self.fig

        # Clear previous frame
        ax.clear()

        # Board
        ax.set_xlim(0, self.width)
        ax.set_ylim(0, self.height)

        ax.set_aspect("equal")

        ax.set_xlabel("X (cm)")
        ax.set_ylabel("Y (cm)")
        ax.set_title("Reset Policy Environment")

        # Occupancy Grid
        if occupancy_grid is not None:

            rows, cols = occupancy_grid.shape

            for r in range(rows):
                for c in range(cols):

                    count = occupancy_grid[r, c]

                    if count > 0:
                        alpha = min(count / 5.0, 1.0)
                        rect = patches.Rectangle(
                            (
                                c * self.cell_size,
                                r * self.cell_size,
                            ),
                            self.cell_size,
                            self.cell_size,
                            alpha=alpha,
                        )

                        ax.add_patch(rect)

        # cube
        ax.scatter(cube_x,cube_y,s=120,marker="s",zorder=10,)

        ax.text(cube_x + 0.3,cube_y + 0.3,"Cube",fontsize=10,)
        #information
        info_lines = []

        if coverage is not None:
            info_lines.append(f"Coverage: {coverage:.2%}")

        if action is not None:
            info_lines.append(
                "Action: "
                + np.array2string(np.asarray(action),precision=2,)
            )

        if len(info_lines) > 0:

            ax.text(
                0.01,
                0.99,
                "\n".join(info_lines),
                transform=ax.transAxes,
                verticalalignment="top",
                bbox=dict(facecolor="white", alpha=0.8),
            )

        if mode == "human":

            fig.canvas.draw_idle()
            fig.canvas.flush_events()
            plt.pause(0.001)

            return None

        elif mode == "rgb_array":

            fig.canvas.draw()
            image = np.asarray(fig.canvas.buffer_rgba())[:, :, :3]
            return image

        else:
            raise ValueError(f"Unknown render mode: {mode}")

    def close(self):
        plt.close(self.fig)