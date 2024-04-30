import cellworld_gym
import gymnasium as gym

if __name__ == "__main__":
    def reward(obs):
        return 1
    env = gym.make("CellworldBotEvade-v0",
                   world_name="21_05",
                   use_lppos=False,
                   use_predator=True,
                   reward_function=reward,
                   render=True,
                   real_time=True)
    env.reset()
    for i in range(100):
        action = env.action_space.sample()
        for j in range(10):
            env.step(action=action)
    print("yes")
