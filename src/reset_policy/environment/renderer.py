"""
Visualization for Reset Policy environment.
"""

from __future__ import annotations

import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


class BoardRenderer:
    """Renders board state to PNG files."""
    
    def __init__(
        self,
        board_width_cm=80,
        board_height_cm=55,
        cell_size_cm=1,
        save_dir=None,
        save_every_n=5,
        max_saved_plots=200,
        max_trajectory_points=500,
    ):
        self.width = board_width_cm
        self.height = board_height_cm
        self.cell_size = cell_size_cm
        
        if save_dir is None:
            save_dir = Path(__file__).resolve().parent.parent.parent / "logs" / "renderings"
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        self.save_every_n = save_every_n
        self.max_saved_plots = max_saved_plots
        self.max_trajectory_points = max_trajectory_points
        
        self.episode_count = 0
        self.step_count = 0
        self.trajectory = []  # List of (cube_x_cm, cube_y_cm) tuples
        self.saved_episodes = set()
        
        print(f"Renderer: Headless mode (saving to {self.save_dir})")
        print(f"  Save every {save_every_n} episodes")
        print(f"  Board: {board_width_cm}x{board_height_cm} cm")
    
    def update_episode(self, episode):
        """Start a new episode."""
        self.episode_count = episode
        self.step_count = 0
        self.trajectory = []
    
    def update_step(self, step, cube_x_cm, cube_y_cm):
        """
        Update current step and record position.
        
        Args:
            cube_x_cm: X position in cm (vertical)
            cube_y_cm: Y position in cm (horizontal)
        """
        self.step_count = step
        self.trajectory.append((cube_x_cm, cube_y_cm))
        
        if len(self.trajectory) > self.max_trajectory_points:
            self.trajectory = self.trajectory[::2]
    
    def render(self, cube_x_cm, cube_y_cm, occupancy_grid=None, coverage=None, 
               action=None, mode="human"):
        """Render current state."""
        
        # Clamp for display
        cube_x_clamped = max(0, min(self.height, cube_x_cm))
        cube_y_clamped = max(0, min(self.width, cube_y_cm))
        
        out_of_bounds = (
            cube_x_cm < 0 or cube_x_cm > self.height or
            cube_y_cm < 0 or cube_y_cm > self.width
        )
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Setup axes
        ax.set_xlim(0, self.width)
        ax.set_ylim(0, self.height)
        ax.set_aspect("equal")
        ax.set_xlabel("Y (cm) - Horizontal")
        ax.set_ylabel("X (cm) - Vertical")
        ax.set_title(f"Episode {self.episode_count}, Step {self.step_count}")
        
        # 1. Heatmap
        if occupancy_grid is not None and occupancy_grid.size > 0:
            max_visits = np.max(occupancy_grid) if np.max(occupancy_grid) > 0 else 1
            heatmap_normalized = occupancy_grid / max_visits
            
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
        
        # 2. Trajectory
        if len(self.trajectory) > 1:
            traj_x = [p[0] for p in self.trajectory]  # Vertical
            traj_y = [p[1] for p in self.trajectory]  # Horizontal
            
            ax.plot(
                traj_y, traj_x,
                color='green',
                linewidth=2,
                alpha=0.7,
                label='Path',
            )
            
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
        
        # 4. Info panel
        info_lines = []
        if coverage is not None:
            info_lines.append(f"Coverage: {coverage:.2%}")
        if out_of_bounds:
            info_lines.append(f"OUT OF BOUNDS! (X:{cube_x_cm:.1f}, Y:{cube_y_cm:.1f})")
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
        
        # 6. SAVE SECTION (WAS MISSING!)
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
        
        last_x, last_y = self.trajectory[-1]  # Unpack (x, y)
        
        return self.render(
            cube_x_cm=last_x,
            cube_y_cm=last_y,
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