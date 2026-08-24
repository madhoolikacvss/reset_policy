from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class GridCell:
    row: int
    col: int


class OccupancyGrid:

    def __init__(
        self,
        x_min: float = 0.035,
        x_max: float = 0.830,
        y_min: float = 0.140,
        y_max: float = 0.654,
        cell_size_m: float = 0.01,      # 1 cm cells
    ):
        """
        Initialize occupancy grid with board boundaries.
        
        Args:
            x_min: Minimum x coordinate in meters (world frame)
            x_max: Maximum x coordinate in meters (world frame)
            y_min: Minimum y coordinate in meters (world frame)
            y_max: Maximum y coordinate in meters (world frame)
            cell_size_m: Size of each cell in meters
        """
        self.x_min = x_min
        self.x_max = x_max
        self.y_min = y_min
        self.y_max = y_max
        
        # Calculate board dimensions from bounds
        self.board_width = x_max - x_min
        self.board_height = y_max - y_min
        self.cell_size = cell_size_m

        # Calculate number of rows and columns
        self.rows = int(np.ceil(self.board_height / cell_size_m))
        self.cols = int(np.ceil(self.board_width / cell_size_m))

        self.grid = np.zeros((self.rows, self.cols), dtype=np.int32)

    def reset(self):
        """Reset all visitation counts to zero."""
        self.grid.fill(0)

    def position_to_cell(self, x: float, y: float) -> GridCell:
        """
        Convert world coordinates to grid cell indices.
        
        If coordinates are outside board boundaries, clamps to nearest edge.
        
        Args:
            x: x coordinate in meters (AprilTag world frame)
            y: y coordinate in meters (AprilTag world frame)
            
        Returns:
            GridCell: Row and column indices
        """
        # Clamp coordinates to board bounds (instead of raising error)
        x_clamped = max(self.x_min, min(self.x_max, x))
        y_clamped = max(self.y_min, min(self.y_max, y))

        # Normalize coordinates relative to board origin (x_min, y_min)
        x_norm = x_clamped - self.x_min
        y_norm = y_clamped - self.y_min

        # Calculate cell indices
        col = min(int(x_norm / self.cell_size), self.cols - 1)
        row = min(int(y_norm / self.cell_size), self.rows - 1)
        
        return GridCell(row, col)

    def visit(self, x: float, y: float) -> int:
        """
        Record a visit to a cell and increment its count.
        
        If coordinates are outside board bounds, clamps to nearest edge cell.
        
        Args:
            x: x coordinate in meters
            y: y coordinate in meters
            
        Returns:
            int: Updated visitation count for the cell
        """
        cell = self.position_to_cell(x, y)
        self.grid[cell.row, cell.col] += 1
        return self.grid[cell.row, cell.col]

    def visitation_count(self, x: float, y: float) -> int:
        """
        Get the visitation count for a cell.
        
        If coordinates are outside board bounds, clamps to nearest edge cell.
        
        Args:
            x: x coordinate in meters
            y: y coordinate in meters
            
        Returns:
            int: Number of times the cell has been visited
        """
        cell = self.position_to_cell(x, y)
        return self.grid[cell.row, cell.col]
    
    def is_new_cell(self, x: float, y: float) -> bool:
        """
        Check if a cell has never been visited.
        
        If coordinates are outside board bounds, clamps to nearest edge cell.
        
        Args:
            x: x coordinate in meters
            y: y coordinate in meters
            
        Returns:
            bool: True if cell has never been visited
        """
        return self.visitation_count(x, y) == 0

    def coverage(self) -> float:
        """
        Calculate the percentage of cells that have been visited at least once.
        
        Returns:
            float: Coverage ratio between 0 and 1
        """
        return np.count_nonzero(self.grid) / self.grid.size

    def get_bounds(self) -> dict:
        """
        Get the board boundaries.
        
        Returns:
            dict: Dictionary with x_min, x_max, y_min, y_max
        """
        return {
            'x_min': self.x_min,
            'x_max': self.x_max,
            'y_min': self.y_min,
            'y_max': self.y_max
        }

    def as_numpy(self) -> np.ndarray:
        """
        Get a copy of the grid as a numpy array.
        
        Returns:
            np.ndarray: Copy of the grid
        """
        return self.grid.copy()