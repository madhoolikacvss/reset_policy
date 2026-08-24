"""
Visualization for Reset Policy environment.

Displays:
- board
- cube trajectory (path taken during episode)
- occupancy heatmap (density of visits)
- current cube position
"""

from __future__ import annotations

import os
import time
import matplotlib
# Use Agg backend for headless saving (no display needed)
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from pathlib import Path
from collections import deque


class BoardRenderer:

    def __init__(
        self,
        board_width=30,
        board_height=20,
        cell_size=1,
        save_dir=None,
        save_every_n=5,           # Save every N episodes
        max_saved_plots=200,
        max_trajectory_points=500, # Max points to keep in trajectory
    ):

        self.width = board_width
        self.height = board_height
        self.cell_size = cell_size

        # Setup save directory
        if save_dir is None:
            save_dir = Path(__file__).resolve().parent.parent.parent / "logs" / "renderings"
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        self.save_every_n = save_every_n
        self.max_saved_plots = max_saved_plots
        self.max_trajectory_points = max_trajectory_points
        
        # Episode tracking
        self.episode_count = 0
        self.step_count = 0
        self.trajectory = []  # List of (x, y) positions in cm
        self.visit_counts = {}  # Dict mapping (row, col) -> count
        
        # For interactive mode (if available)
        self.fig = None
        self.ax = None
        self.interactive = False
        
        # Try to create interactive figure if display is available
        try:
            if os.environ.get('DISPLAY') or os.name == 'nt':
                matplotlib.use('TkAgg')
                import matplotlib.pyplot as plt
                self.fig, self.ax = plt.subplots(figsize=(10, 8))
                plt.ion()
                self.fig.show()
                self.fig.canvas.draw()
                self.interactive = True
                print("Renderer: Interactive mode enabled")
            else:
                print("Renderer: Using headless mode (saving to disk)")
        except Exception as e:
            print(f"Renderer: Using headless mode (saving to disk) - {e}")
        
        # For tracking last save
        self.last_saved_episode = -1

    def update_episode(self, episode):
        """Start a new episode - reset trajectory and counts."""
        self.episode_count = episode
        self.step_count = 0
        self.trajectory = []  # Reset trajectory for new episode
        self.visit_counts = {}  # Reset visit counts for new episode

    def update_step(self, step, cube_x_cm, cube_y_cm):
        """Update current step and record position."""
        self.step_count = step
        
        # Add to trajectory (in cm coordinates)
        self.trajectory.append((cube_x_cm, cube_y_cm))
        
        # Limit trajectory size
        if len(self.trajectory) > self.max_trajectory_points:
            # Keep every other point to reduce memory
            self.trajectory = self.trajectory[::2]
        
        # Update visit counts for heatmap (convert to grid cell)
        cell_x = int(cube_x_cm / self.cell_size)
        cell_y = int(cube_y_cm / self.cell_size)
        
        # Only count if within bounds
        if 0 <= cell_x < int(self.width / self.cell_size) and 0 <= cell_y < int(self.height / self.cell_size):
            key = (cell_x, cell_y)
            self.visit_counts[key] = self.visit_counts.get(key, 0) + 1

    def render(self, cube_x, cube_y, occupancy_grid=None, coverage=None, action=None, mode="human"):
        """
        Render the current state.
        
        Shows:
        - Occupancy heatmap (density of visits)
        - Cube trajectory (path taken)
        - Current cube position
        - Coverage percentage
        """
        
        # Clamp cube position
        cube_x_clamped = max(0, min(self.width, cube_x))
        cube_y_clamped = max(0, min(self.height, cube_y))
        
        out_of_bounds = (
            cube_x < 0 or cube_x > self.width or
            cube_y < 0 or cube_y > self.height
        )

        # ============================================================
        # Create the plot
        # ============================================================
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Board
        ax.set_xlim(0, self.width)
        ax.set_ylim(0, self.height)
        ax.set_aspect("equal")
        ax.set_xlabel("X (cm)")
        ax.set_ylabel("Y (cm)")
        ax.set_title(f"Episode {self.episode_count}, Step {self.step_count}")

        # ============================================================
        # 1. OCCUPANCY HEATMAP (from visit_counts)
        # ============================================================
        
        if self.visit_counts:
            # Create heatmap grid
            grid_cols = int(self.width / self.cell_size)
            grid_rows = int(self.height / self.cell_size)
            heatmap = np.zeros((grid_rows, grid_cols))
            
            max_visits = max(self.visit_counts.values()) if self.visit_counts else 1
            
            for (col, row), count in self.visit_counts.items():
                if 0 <= row < grid_rows and 0 <= col < grid_cols:
                    # Normalize by max visits for color intensity
                    heatmap[row, col] = count / max_visits
            
            # Display heatmap
            im = ax.imshow(
                heatmap,
                origin='lower',
                extent=[0, self.width, 0, self.height],
                cmap='hot',
                alpha=0.6,
                aspect='auto',
                vmin=0,
                vmax=1,
            )
            
            # Add colorbar
            cbar = fig.colorbar(im, ax=ax, shrink=0.8)
            cbar.set_label('Visit Density (relative)')

        # ============================================================
        # 2. TRAJECTORY PATH
        # ============================================================
        
        if len(self.trajectory) > 1:
            # Extract x and y coordinates
            traj_x = [p[0] for p in self.trajectory]
            traj_y = [p[1] for p in self.trajectory]
            
            # Draw the path as a line
            ax.plot(
                traj_x, traj_y,
                color='green',
                linewidth=2,
                alpha=0.7,
                label='Path Trajectory',
            )
            
            # Draw start point (green circle)
            ax.scatter(
                traj_x[0], traj_y[0],
                s=150,
                marker='o',
                color='lime',
                edgecolor='darkgreen',
                linewidth=2,
                zorder=8,
                label='Start',
            )
            
            # Draw intermediate points (small dots)
            if len(self.trajectory) > 2:
                # Sample points to avoid overcrowding
                step = max(1, len(self.trajectory) // 50)
                sample_x = traj_x[::step]
                sample_y = traj_y[::step]
                ax.scatter(
                    sample_x, sample_y,
                    s=20,
                    color='green',
                    alpha=0.3,
                    zorder=5,
                )

        # ============================================================
        # 3. CURRENT CUBE POSITION
        # ============================================================
        
        ax.scatter(
            cube_x_clamped,
            cube_y_clamped,
            s=300,
            marker="s",
            zorder=10,
            color='red' if out_of_bounds else 'blue',
            edgecolor='black',
            linewidth=2,
            label='Current Position',
        )

        # Cube label
        label = "Cube"
        if out_of_bounds:
            label += " (OUT OF BOUNDS!)"
        ax.text(
            cube_x_clamped + 0.5,
            cube_y_clamped + 0.5,
            label,
            fontsize=10,
            color='red' if out_of_bounds else 'black',
            weight='bold',
        )

        # ============================================================
        # 4. INFORMATION PANEL
        # ============================================================
        
        info_lines = []
        
        if coverage is not None:
            info_lines.append(f"Coverage: {coverage:.2%}")
        
        if self.visit_counts:
            total_cells = len(self.visit_counts)
            max_visits = max(self.visit_counts.values()) if self.visit_counts else 0
            info_lines.append(f"Cells Visited: {total_cells}")
            info_lines.append(f"Max Visits: {max_visits}")
        
        if out_of_bounds:
            info_lines.append(f"⚠️ Out of bounds! ({cube_x:.2f}, {cube_y:.2f})")
        
        if action is not None:
            info_lines.append("Action: " + np.array2string(np.asarray(action), precision=2))
        
        if len(self.trajectory) > 0:
            info_lines.append(f"Path Length: {len(self.trajectory)} steps")

        if len(info_lines) > 0:
            ax.text(
                0.02,
                0.98,
                "\n".join(info_lines),
                transform=ax.transAxes,
                verticalalignment="top",
                bbox=dict(facecolor="white", alpha=0.9, edgecolor='gray'),
                fontsize=10,
            )

        # ============================================================
        # 5. LEGEND
        # ============================================================
        
        ax.legend(loc='upper right')

        # ============================================================
        # 6. SAVE PLOT (only if it's time to save)
        # ============================================================
        
        saved_path = None
        
        # Check if we should save this episode
        should_save = (
            self.episode_count % self.save_every_n == 0 and
            self.episode_count != self.last_saved_episode
        )
        
        if should_save:
            save_path = self.save_dir / f"render_ep{self.episode_count:04d}_step{self.step_count:04d}.png"
            fig.savefig(save_path, dpi=150, bbox_inches='tight')
            saved_path = str(save_path)
            self.last_saved_episode = self.episode_count
            print(f"Saved render: {save_path}")
            
            # Clean up old plots
            self._cleanup_old_plots()

        # ============================================================
        # 7. INTERACTIVE DISPLAY (if available)
        # ============================================================
        
        if self.interactive and self.fig is not None and mode == "human":
            try:
                # Transfer plot to interactive figure
                self.ax.clear()
                # Copy the plot elements (simplified for performance)
                self.ax.set_xlim(0, self.width)
                self.ax.set_ylim(0, self.height)
                self.ax.set_aspect("equal")
                self.ax.set_xlabel("X (cm)")
                self.ax.set_ylabel("Y (cm)")
                self.ax.set_title(f"Episode {self.episode_count}, Step {self.step_count}")
                
                # Add heatmap if available
                if self.visit_counts:
                    grid_cols = int(self.width / self.cell_size)
                    grid_rows = int(self.height / self.cell_size)
                    heatmap = np.zeros((grid_rows, grid_cols))
                    max_visits = max(self.visit_counts.values()) if self.visit_counts else 1
                    for (col, row), count in self.visit_counts.items():
                        if 0 <= row < grid_rows and 0 <= col < grid_cols:
                            heatmap[row, col] = count / max_visits
                    self.ax.imshow(
                        heatmap,
                        origin='lower',
                        extent=[0, self.width, 0, self.height],
                        cmap='hot',
                        alpha=0.6,
                        aspect='auto',
                        vmin=0,
                        vmax=1,
                    )
                
                # Add trajectory
                if len(self.trajectory) > 1:
                    traj_x = [p[0] for p in self.trajectory]
                    traj_y = [p[1] for p in self.trajectory]
                    self.ax.plot(traj_x, traj_y, color='green', linewidth=2, alpha=0.7)
                    self.ax.scatter(traj_x[0], traj_y[0], s=150, marker='o', color='lime', edgecolor='darkgreen', linewidth=2)
                
                # Add cube
                self.ax.scatter(
                    cube_x_clamped,
                    cube_y_clamped,
                    s=300,
                    marker="s",
                    zorder=10,
                    color='red' if out_of_bounds else 'blue',
                    edgecolor='black',
                    linewidth=2,
                )
                
                # Add info
                if len(info_lines) > 0:
                    self.ax.text(
                        0.02,
                        0.98,
                        "\n".join(info_lines),
                        transform=self.ax.transAxes,
                        verticalalignment="top",
                        bbox=dict(facecolor="white", alpha=0.9, edgecolor='gray'),
                        fontsize=10,
                    )
                
                self.fig.canvas.draw_idle()
                self.fig.canvas.flush_events()
                plt.pause(0.001)
            except Exception as e:
                print(f"Renderer: Interactive mode failed - {e}")
                self.interactive = False

        plt.close(fig)
        
        if mode == "rgb_array" and saved_path:
            try:
                img = plt.imread(saved_path)
                return img
            except:
                return None

        return saved_path

    def save_final_render(self, occupancy_grid=None, coverage=None):
        """Save a final render at episode end."""
        if self.trajectory:
            # Use the last position
            last_x, last_y = self.trajectory[-1]
            return self.render(
                cube_x=last_x,
                cube_y=last_y,
                occupancy_grid=occupancy_grid,
                coverage=coverage,
            )
        return None

    def _cleanup_old_plots(self):
        """Remove old plots to prevent disk overflow."""
        try:
            files = sorted(self.save_dir.glob("render_ep*.png"))
            if len(files) > self.max_saved_plots:
                for f in files[:-self.max_saved_plots]:
                    f.unlink()
        except Exception as e:
            print(f"Warning: Could not clean up old plots - {e}")

    def close(self):
        try:
            if self.fig is not None:
                plt.close(self.fig)
        except:
            pass