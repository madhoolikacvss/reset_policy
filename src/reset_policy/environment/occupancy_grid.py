"""
Occupancy grid for tracking cube positions on the board.

Coordinate system:
- Y-axis: horizontal (left/right)
- X-axis: vertical (up/down)
"""

from __future__ import annotations

import numpy as np


class OccupancyGrid:
    """Tracks visitation counts on a 2D grid."""
    
    def __init__(
        self,
        x_min: float,
        x_max: float,
        y_min: float,
        y_max: float,
        cell_size_m: float = 0.01,
    ):
        """
        Initialize occupancy grid.
        
        Args:
            x_min: Minimum X coordinate (vertical axis)
            x_max: Maximum X coordinate (vertical axis)
            y_min: Minimum Y coordinate (horizontal axis)
            y_max: Maximum Y coordinate (horizontal axis)
            cell_size_m: Cell size in meters
        """
        self.x_min = x_min
        self.x_max = x_max
        self.y_min = y_min
        self.y_max = y_max
        self.cell_size = cell_size_m
        
        # Calculate grid dimensions
        self.board_width = x_max - x_min  # Vertical extent
        self.board_height = y_max - y_min  # Horizontal extent
        
        # Rows = X axis (vertical), Cols = Y axis (horizontal)
        self.rows = int(np.ceil(self.board_width / cell_size_m))
        self.cols = int(np.ceil(self.board_height / cell_size_m))
        
        self.grid = np.zeros((self.rows, self.cols), dtype=np.int32)
    
    def reset(self):
        """Reset all visitation counts."""
        self.grid.fill(0)
    
    def position_to_cell(self, x: float, y: float):
        """
        Convert world coordinates to grid indices.
        Clamps to board bounds if outside.
        
        Args:
            x: X coordinate in meters (vertical)
            y: Y coordinate in meters (horizontal)
            
        Returns:
            Tuple of (row, col)
        """
        # Clamp to bounds
        x_clamped = np.clip(x, self.x_min, self.x_max)
        y_clamped = np.clip(y, self.y_min, self.y_max)
        
        # Convert to grid indices
        # Row corresponds to X axis (vertical)
        # Col corresponds to Y axis (horizontal)
        row = min(int((x_clamped - self.x_min) / self.cell_size), self.rows - 1)
        col = min(int((y_clamped - self.y_min) / self.cell_size), self.cols - 1)
        
        return row, col
    
    def visit(self, x: float, y: float) -> int:
        """Record a visit and return updated count."""
        row, col = self.position_to_cell(x, y)
        self.grid[row, col] += 1
        return self.grid[row, col]
    
    def visitation_count(self, x: float, y: float) -> int:
        """Get visitation count for a position."""
        row, col = self.position_to_cell(x, y)
        return self.grid[row, col]
    
    def is_new_cell(self, x: float, y: float) -> bool:
        """Check if position has never been visited."""
        return self.visitation_count(x, y) == 0
    
    def coverage(self) -> float:
        """Calculate coverage ratio (0-1)."""
        return np.count_nonzero(self.grid) / self.grid.size
    
    def as_numpy(self) -> np.ndarray:
        """Get copy of grid."""
        return self.grid.copy()
