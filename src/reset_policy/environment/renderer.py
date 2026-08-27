"""
Visualization for Reset Policy environment.

Displays:
- Board with proper coordinate system (Y horizontal, X vertical)
- Cube trajectory (path taken during episode)
- Occupancy heatmap (from OccupancyGrid)
- Current cube position
"""

from __future__ import annotations

import matplotlib
matplotlib.use('Agg')  # Headless backend

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


class BoardRenderer:
    """Renders board state to PNG files."""
    
    def __init__(
        self,
        board_width_cm=80,   # Horizontal extent (Y axis)
        board_height_cm=55,  # Vertical extent (X axis)
        cell_size_cm=1,
        save_dir=None,
        save_every_n=5,
        max_saved_plots=200,
        max_trajectory_points=500,
    ):
        self.width = board_width_cm    # Horizontal (Y)
        self.height = board_height_cm  # Vertical (X)
        self.cell_size = cell_size_cm
        
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
        self.trajectory = []  # List of (y_cm, x_cm) tuples
        
        # Track saved episodes
        self.saved_episodes = set()
        
        print(f"Renderer: Headless mode (saving to {self.save_dir})")
        print(f"  Save every {save_every_n} episodes")
        print(f"  Board: {board_width_cm}x{board_height_cm} cm (Y x X)")
    
    def update_episode(self, episode):
        """Start a new episode."""
        self.episode_count = episode
        self.step_count = 0
        self.trajectory = []
    
    def update_step(self, step, cube_y_cm, cube_x_cm):
        """
        Update current step and record position.
        
        Args:
            cube_y_cm: Y position in cm (horizontal)
            cube_x_cm: X position in cm (vertical)
        """
        self.step_count = step
        self.trajectory.append((cube_y_cm, cube_x_cm))
        
        # Limit trajectory size
        if len(self.trajectory) > self.max_trajectory_points:
            self.trajectory = self.trajectory[::2]
    
    def render(self, cube_y_cm, cube_x_cm, occupancy_grid=None, coverage=None, 
               action=None, mode="human"):
        """
        Render current state.
        
        Args:
            cube_y_cm: Y position in cm (horizontal)
            cube_x_cm: X position in cm (vertical)
        """
        # Clamp for display
        cube_y_clamped = max(0, min(self.width, cube_y_cm))
        cube_x_clamped = max(0, min(self.height, cube_x_cm))
        
        out_of_bounds = (
            cube_y_cm < 0 or cube_y_cm > self.width or
            cube_x_cm < 0 or cube_x_cm > self.height
        )
        
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Board setup
        ax.set_xlim(0, self.width)    # Horizontal (Y)
        ax.set_ylim(0, self.height)   # Vertical (X)
        ax.set_aspect("equal")
        ax.set_xlabel("Y (cm) - Horizontal")
        ax.set_ylabel("X (cm) - Vertical")
        ax.set_title(f"Episode {self.episode_count}, Step {self.step_count}")
        
        # 1. Occupancy heatmap
        if occupancy_grid is not None and occupancy_grid.size > 0:
            # Transpose if needed: grid is [rows=X, cols=Y], imshow wants [Y, X]
            # occupancy_grid shape: (rows, cols) = (X_cells, Y_cells)
            # For imshow with extent=[0, width, 0, height], we need (Y_cells, X_cells)
            heatmap_display = occupancy_grid.T  # Transpose to (Y, X)
            
            max_visits = np.max(heatmap_display) if np.max(heatmap_display) > 0 else 1
            heatmap_normalized = heatmap_display / max_visits
            
            im = ax.imshow(
                heatmap_normalized,
                origin='lower',
                extent=[0, self.width, 0, self.height],
                cmap='hot',
                alpha=0.6,
                aspect='auto',
                vmin=0,
                vmax=1,
            )
            
            cbar = fig.colorbar(im, ax=ax, shrink=0.8)
            cbar.set_label('Visit Density')
        
        # 2. Trajectory path
        if len(self.trajectory) > 1:
            traj_y = [p[0] for p in self.trajectory]  # Horizontal
            traj_x = [p[1] for p in self.trajectory]  # Vertical
            
            ax.plot(
                traj_y, traj_x,
                color='green',
                linewidth=2,
                alpha=0.7,
                label='Path',
            )
            
            # Start position
            ax.scatter(
                traj_y[0], traj_x[0],
                s=150,
                marker='o',
                color='lime',
                edgecolor='darkgreen',
                linewidth=2,
                zorder=8,
                label='Start',
            )
            
            # Debug print
            print(f"[RENDER] Trajectory points: {len(self.trajectory)}")
            print(f"[RENDER] First: (y={traj_y[0]:.1f}, x={traj_x[0]:.1f})")
            print(f"[RENDER] Last: (y={traj_y[-1]:.1f}, x={traj_x[-1]:.1f})")
        
        # 3. Current position
        ax.scatter(
            cube_y_clamped, cube_x_clamped,
            s=300,
            marker="s",
            zorder=10,
            color='red' if out_of_bounds else 'blue',
            edgecolor='black',
            linewidth=2,
            label='Current',
        )
        
        # Debug print
        print(f"[RENDER] Cube: y={cube_y_cm:.1f}, x={cube_x_cm:.1f}, OOB={out_of_bounds}")
        
        # 4. Information panel
        info_lines = []
        if coverage is not None:
            info_lines.append(f"Coverage: {coverage:.2%}")
        if out_of_bounds:
            info_lines.append(f"OUT OF BOUNDS! (Y:{cube_y_cm:.1f}, X:{cube_x_cm:.1f})")
        if action is not None:
            info_lines.append(f"Action: [{action[0]:+.2f}, {action[1]:+.2f}, {action[2]:+.2f}, {action[3]:+.2f}]")
        info_lines.append(f"Steps: {len(self.trajectory)}")
        
        if info_lines:
            ax.text(
                0.02, 0.98,
                "\n".join(info_lines),
                transform=ax.transAxes,
                verticalalignment="top",
                bbox=dict(facecolor="white", alpha=0.9, edgecolor='gray'),
                fontsize=10,
            )
        
        # 5. Legend
        ax.legend(loc='upper right')
        
        # 6. Save plot
        saved_path = None
        should_save = (
            self.episode_count % self.save_every_n == 0 and
            self.episode_count not in self.saved_episodes
        )
        
        if should_save:
            save_path = self.save_dir / f"render_ep{self.episode_count:04d}.png"
            fig.savefig(save_path, dpi=150, bbox_inches='tight')
            saved_path = str(save_path)
            self.saved_episodes.add(self.episode_count)
            print(f"Saved render: {save_path}")
            self._cleanup_old_plots()
        
        plt.close(fig)
        
        if mode == "rgb_array" and saved_path:
            try:
                return plt.imread(saved_path)
            except:
                return None
        
        return saved_path
    
    def save_final_render(self, occupancy_grid=None, coverage=None):
        """Save final render at episode end."""
        if not self.trajectory:
            print("[RENDER] No trajectory to render")
            return None
        
        last_y, last_x = self.trajectory[-1]
        print(f"[RENDER] Final: y={last_y:.1f}, x={last_x:.1f}")
        
        return self.render(
            cube_y_cm=last_y,
            cube_x_cm=last_x,
            occupancy_grid=occupancy_grid,
            coverage=coverage,
        )
    
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
        """Close all figures."""
        plt.close('all')
