from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class GridCell:
    row: int
    col: int


class OccupancyGrid:

    def __init__(self,board_width_m: float = 0.30,board_height_m: float = 0.20, cell_size_m: float = 0.01,      # 1 cm cells
    ):

        self.board_width = board_width_m
        self.board_height = board_height_m
        self.cell_size = cell_size_m

        self.rows = int(np.ceil(board_height_m / cell_size_m))
        self.cols = int(np.ceil(board_width_m / cell_size_m))

        self.grid = np.zeros((self.rows, self.cols),dtype=np.int32,)

    def reset(self):
        self.grid.fill(0)

    def position_to_cell(self, x: float, y: float) -> GridCell:
        """
        x,y are directly in the AprilTag world frame.

        Origin = world tag.
        Units = meters.
        """

        if not (0 <= x <= self.board_width):
            raise ValueError(f"x={x:.3f} outside board")

        if not (0 <= y <= self.board_height):
            raise ValueError(f"y={y:.3f} outside board")

        col = min(int(x / self.cell_size),self.cols - 1,)

        row = min(int(y / self.cell_size),self.rows - 1,)
        return GridCell(row, col)


    def visit(self, x: float, y: float):
        cell = self.position_to_cell(x, y)
        self.grid[cell.row, cell.col] += 1
        return self.grid[cell.row, cell.col]


    def visitation_count(self, x: float, y: float):
        cell = self.position_to_cell(x, y)
        return self.grid[cell.row, cell.col]
    
    def is_new_cell(self, x: float, y: float):
        return self.visitation_count(x, y) == 0

    def coverage(self):
        return np.count_nonzero(self.grid) / self.grid.size

    def as_numpy(self):

        return self.grid.copy()