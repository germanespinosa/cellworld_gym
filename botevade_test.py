import time
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
                   render=False,
                   real_time=False,
                   predator_prey_forward_speed_ratio=1.5,
                   predator_prey_turning_speed_ratio=1.5)
    env.reset()
    start_time = time.time()
    step_count = 0
    for i in range(100):
        action = env.action_space.sample()
        for j in range(10):
            env.step(action=action)
            step_count += 1

    total_time = time.time() - start_time

    print("not real time steps per second: {:.2f}".format(step_count / total_time))

    env = gym.make("CellworldBotEvade-v0",
                   world_name="21_05",
                   use_lppos=False,
                   use_predator=True,
                   reward_function=reward,
                   render=True,
                   real_time=True)
    env.reset()
    start_time = time.time()
    step_count = 0
    for i in range(10):
        action = env.action_space.sample()
        for j in range(10):
            env.step(action=action)
            step_count += 1

    total_time = time.time() - start_time

    print("real steps per second: {:.2f}".format(step_count / total_time))
