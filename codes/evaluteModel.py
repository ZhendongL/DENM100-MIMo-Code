import os
import time
import cv2
import mimoEnv
import gymnasium as gym
from mimoEnv.envs.mimo_env import MIMoEnv
from mimoActuation.actuation import SpringDamperModel
from mimoActuation.muscle import MuscleModel
from stable_baselines3.common.evaluation import evaluate_policy

algorithm = 'PPO'  # PPO  TD3
actuation_model = MuscleModel
env = gym.make('MIMoSelfBody-v0', actuation_model=actuation_model)

if algorithm == 'PPO':
    from modify.PPOdiy import PPO as RL
elif algorithm == 'TD3':
    from modify.TD3diy import TD3 as RL

num='5.2_1'
load_model = 'models/selfbody/selfbody'+algorithm+'version'+num+'/model_1'
model = RL.load(load_model, env)
mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=20,render=False)
print(f"Mean reward version{num}: {mean_reward} +/- {std_reward}")

vec_env = model.get_env()
obs = vec_env.reset()

start = time.time()

# Evaluate
for i in range(12000):
    action, _states = model.predict(obs, deterministic=True)
    obs, rewards, dones, info = vec_env.step(action)
    # vec_env.render("human")
    # env.mujoco_renderer.render(render_mode="human")

# print("Elapsed time: ", time.time() - start, "Simulation time:", 12000*env.dt)


