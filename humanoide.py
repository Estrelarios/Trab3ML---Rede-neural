import gymnasium as gym
import time


env = gym.make("Humanoid-v5", render_mode="human", xml_file="humanoid.xml")

# Reset environment to start a new episode
env.action_space.seed(1)  # Set a seed for reproducibility
observation, info = env.reset()
env.action_space.seed(1)  # Set a seed for reproducibility





time.sleep(3)
env.close()