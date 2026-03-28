"""
Simple simulation of a robot navigating with a Stimuli-Induced Equilibrium Point (SIEP)-like
potential field. The robot moves toward a goal while avoiding humans/obstacles using
virtual forces. This code is for educational purposes and demonstrates how to set up
and run a basic navigation loop.

References:
- SIEP concept models virtual forces induced by stimuli perceived from the environment;
  the resultant motion dynamics are obtained by summing those forces【555555923291556†L104-L120】.
- Social navigation research stresses that respecting personal space and modelling human
  behaviour is essential for socially acceptable robot navigation【530254435115083†L274-L287】.

Usage:
  python siep_navigation_example.py
This will run the simulation and display a plot showing the robot trajectory.

"""

import numpy as np
import matplotlib.pyplot as plt

# Simulation parameters
TIME_STEP = 0.1  # s
TOTAL_TIME = 30.0  # s

# Robot parameters
robot_pos = np.array([0.0, 0.0])
robot_vel = np.array([0.0, 0.0])

# Goal position
goal_pos = np.array([10.0, 8.0])

# Humans/obstacles: list of (position, orientation (radians), social radius)
humans = [
    (np.array([5.0, 4.0]), 0.0, 0.8),   # human 1 at (5,4) with orientation 0
    (np.array([7.0, 6.5]), np.pi/4, 1.0) # human 2 at (7,6.5) with orientation 45 degrees
]

# Force parameters
k_attr = 1.0   # attractive force constant to goal
k_rep = 2.0    # repulsive force constant from humans
personal_space_scale = 0.5  # scaling factor for social space

# Lists to store trajectory for plotting
traj = []

# Helper function: compute repulsive force from a human

def repulsive_force(robot_pos: np.ndarray, human_pos: np.ndarray, social_radius: float) -> np.ndarray:
    """
    Compute a repulsive force on the robot from a human. The force magnitude decays
    with distance and becomes strong when the robot enters the human's personal space.
    """
    diff = robot_pos - human_pos
    dist = np.linalg.norm(diff)
    if dist < 1e-5:
        dist = 1e-5  # avoid division by zero
    direction = diff / dist
    # Repulsive force magnitude using an exponential decay outside personal space
    force_mag = k_rep * np.exp(-(dist - social_radius) / personal_space_scale)
    return force_mag * direction

# Simulation loop
num_steps = int(TOTAL_TIME / TIME_STEP)
for step in range(num_steps):
    # Compute attractive force toward the goal
    to_goal = goal_pos - robot_pos
    dist_to_goal = np.linalg.norm(to_goal)
    if dist_to_goal < 0.2:
        # Robot reached the goal
        print(f"Reached goal at time {step * TIME_STEP:.2f} s")
        break
    attr_force = k_attr * to_goal / (dist_to_goal + 1e-5)

    # Compute repulsive forces from humans
    rep_force = np.array([0.0, 0.0])
    for human_pos, human_ori, social_radius in humans:
        rep_force += repulsive_force(robot_pos, human_pos, social_radius)

    # Total force (virtual stimuli)
    total_force = attr_force + rep_force

    # Update robot velocity and position (simple Euler integration)
    robot_vel = total_force  # treat total force as velocity command
    robot_pos = robot_pos + robot_vel * TIME_STEP

    # Save trajectory
    traj.append(robot_pos.copy())

# Convert trajectory to numpy array for plotting
traj = np.array(traj)

# Plotting
plt.figure(figsize=(8, 6))
# Plot humans and their personal space circles
for human_pos, human_ori, social_radius in humans:
    for i, (human_pos, human_ori, social_radius) in enumerate(humans):
        plt.scatter(human_pos[0], human_pos[1], c='red', label='human' if i == 0 else None)
        
    circle = plt.Circle(human_pos, social_radius, color='red', alpha=0.2)
    plt.gca().add_patch(circle)
# Plot goal and robot trajectory
plt.scatter(goal_pos[0], goal_pos[1], c='green', label='goal')
plt.plot(traj[:, 0], traj[:, 1], '-b', label='robot trajectory')
plt.scatter(traj[0, 0], traj[0, 1], c='blue', label='start')
plt.title('SIEP-like Social Navigation Simulation')
plt.xlabel('x position')
plt.ylabel('y position')
plt.legend()
plt.grid(True)
plt.axis('equal')
plt.tight_layout()
plt.savefig('/inspire/ssd/tenant_predefaa-9a1b-4522-bb10-8850f313be13/global_user/8717-pengyan/Siep_navigation/siep_navigation_example_plot.png', dpi=300)
print('Simulation finished. Plot saved to siep_navigation_example_plot.png')
