import cellworld_gym
import gymnasium as gym

if __name__ == "__main__":
    def reward():
        pass
    env = gym.make("CellworldBotEvade",
                   world_name="21_05",
                   use_lppos=True,
                   use_predator=True,
                   reward_function=reward)
    print("yes")
