import time
import cellworld_gym
import cellworld_belief as belief
import gymnasium as gym

if __name__ == "__main__":
    def reward(*obs):
        return 1

    DB = belief.DecreasingBeliefComponent(rate=.05)
    NB = belief.NoBeliefComponent()
    LOS = belief.LineOfSightComponent()
    V = belief.VisibilityComponent()
    D = belief.DiffusionComponent()
    GD = belief.GaussianDiffusionComponent()
    DD = belief.DirectedDiffusionComponent()
    O = belief.OcclusionsComponent()
    A = belief.ArenaComponent()
    M = belief.MapComponent()
    NL = belief.ProximityComponent()

    components = [V, LOS, M]

    env = gym.make("CellworldBotEvadeBelief-v0",
                   world_name="21_05",
                   use_lppos=False,
                   use_predator=True,
                   reward_function=reward,
                   render=True,
                   real_time=True,
                   belief_state_components=components)
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
