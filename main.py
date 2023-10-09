import gymnasium as gym
import panda_gym
from stable_baselines3 import HerReplayBuffer
from sb3_contrib import TQC
import time

env = gym.make("PandaPickAndPlace-v3", render_mode="human")

model_class = TQC
goal_selection_strategy = "future"
policy_kwargs = dict(n_critics=2, n_quantiles=25)
model = model_class(policy="MultiInputPolicy",env=env, replay_buffer_class=HerReplayBuffer,verbose=1,    
    replay_buffer_kwargs=dict(n_sampled_goal=4, goal_selection_strategy=goal_selection_strategy),
    top_quantiles_to_drop_per_net=2, policy_kwargs=policy_kwargs)

model.learn(total_timesteps=10_000, log_interval=4, progress_bar=True)
model.save("./zip_model/tqc_her_panda")

del model # remove to demonstrate saving and loading
model = TQC.load("./zip_model/tqc_her_panda", env=env)

obs, _ = env.reset()
for i in range(100):
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)
    env.render()
    if terminated or truncated:
      obs, info = env.reset()
    time.sleep(0.1)