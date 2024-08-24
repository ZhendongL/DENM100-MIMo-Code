""" Simple script to view the showroom. We perform no training and MIMo takes no actions.
"""
import gymnasium as gym
import time
import numpy as np
import mimoEnv

def main():
    """ Creates the environment and takes 200 time steps. MIMo takes no actions.
    The environment is rendered to an interactive window.
    """
    env = gym.make("MIMoSelfBody-v0")

    _ = env.reset()

    start = time.time()

    for step in range(12000):
        action = env.action_space.sample()
        # action = np.zeros(env.action_space.shape)
        obs, reward, done, trunc, info = env.step(action)
        # env.mujoco_renderer.render(render_mode="human")
        if done or trunc:
            env.reset()

    print("Elapsed time: ", time.time() - start, "Simulation time:", 12000*env.dt)
    # env.close()

if __name__ == "__main__":
    main()

