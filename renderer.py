"""
Visualization for Reset Policy environment.

Displays:
- board
- cube position
- occupancy grid
"""


import matplotlib.pyplot as plt
import numpy as np



class BoardRenderer:


    def __init__(
        self,
        board_width=30,
        board_height=20,
        cell_size=1,
    ):


        self.width = board_width
        self.height = board_height
        self.cell_size = cell_size


        # plt.ion()

        # self.fig, self.ax = plt.subplots(
        #     figsize=(8,5)
        # )


    def render(
        self,
        cube_x,
        cube_y,
        occupancy_grid=None,
        coverage=None,
        action=None,
        mode="human",
    ):


        fig, ax = plt.subplots(
            figsize=(8,5)
        )


        ax.set_xlim(
            0,
            self.width
        )

        ax.set_ylim(
            0,
            self.height
        )


        ax.set_aspect("equal")


        # -----------------------------
        # Occupancy
        # -----------------------------

        if occupancy_grid is not None:

            rows, cols = occupancy_grid.shape

            for r in range(rows):

                for c in range(cols):

                    if occupancy_grid[r,c] > 0:

                        ax.add_patch(
                            plt.Rectangle(
                                (
                                    c*self.cell_size,
                                    r*self.cell_size,
                                ),
                                self.cell_size,
                                self.cell_size,
                            )
                        )


        # -----------------------------
        # Cube
        # -----------------------------

        ax.scatter(
            cube_x,
            cube_y,
            s=100,
        )


        ax.text(
            cube_x+0.5,
            cube_y+0.5,
            "Cube",
        )


        # -----------------------------
        # Output
        # -----------------------------


        if mode == "human":

            plt.show(block=False)

            plt.pause(0.001)

            plt.close(fig)

            return None



        elif mode == "rgb_array":

            fig.canvas.draw()


            image = np.asarray(
                fig.canvas.buffer_rgba()
            )


            image = image[:,:,:3]


            plt.close(fig)


            return image