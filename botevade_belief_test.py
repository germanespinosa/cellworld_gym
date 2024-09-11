import time
import cellworld_game as game
import cellworld_gym
import cellworld_belief as belief
import gymnasium as gym



if __name__ == "__main__":
    def reward(*obs):
        print(obs)
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

    env: cellworld_gym.BotEvadeBeliefEnv = gym.make("CellworldBotEvadeBelief-v0",
                                                    world_name="21_05",
                                                    use_lppos=False,
                                                    use_predator=True,
                                                    reward_function=reward,
                                                    render=True,
                                                    real_time=True,
                                                    belief_state_components=components,
                                                    prey_max_forward_speed=.25,
                                                    prey_max_turning_speed=2.0,
                                                    predator_prey_forward_speed_ratio=1.0,
                                                    predator_prey_turning_speed_ratio=1.0)


    def render(screen, coordinate_converter: game.CoordinateConverter):
        import numpy as np
        import pygame
        obs: np.ndarray = env.get_observation()
        factor = max(obs.max(), 1 / obs.size)
        values = (obs * 255 / factor).astype(int)
        heatmap_surface = pygame.Surface(obs.shape[::-1], pygame.SRCALPHA)
        pix_array = pygame.PixelArray(heatmap_surface)
        for y in range(obs.shape[0]):
            for x in range(obs.shape[1]):
                v = values[y, x]
                if v < 0:
                    pix_array[x, obs.shape[0] - 1 - y] = (0, 0, 255, 255)
                else:
                    pix_array[x, obs.shape[0] - 1 - y] = (255, 0, 0, v)
        # Delete the pixel array to unlock the surface
        del pix_array

        # Scale the surface to the window size
        scaled_heatmap = pygame.transform.scale(heatmap_surface,
                                                size=(coordinate_converter.screen_width,
                                                      coordinate_converter.screen_height))
        screen.blit(scaled_heatmap, (0, 0))


    env.model.view.add_render_step(render_step=render, z_index=500)
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
