import typing
from ..core import Environment
import cellworld_belief as belief
import numpy as np
import cellworld_game as cwgame
from gymnasium import spaces
from gymnasium import Env


class BotEvadeBeliefEnv(Environment):
    def __init__(self,
                 world_name: str,
                 use_lppos: bool,
                 use_predator: bool,
                 max_step: int = 300,
                 reward_function: typing.Callable[[cwgame.Model], float] = lambda x: 0,
                 time_step: float = .25,
                 render: bool = False,
                 real_time: bool = False,
                 belief_state_components: typing.List[belief.BeliefStateComponent] = None,
                 belief_state_definition: int = 100,
                 belief_state_probability: float = 1.0):

        self.max_step = max_step
        self.reward_function = reward_function
        self.time_step = time_step
        self.loader = cwgame.CellWorldLoader(world_name=world_name)

        if use_lppos:
            self.action_list = self.loader.tlppo_action_list
        else:
            self.action_list = self.loader.full_action_list

        self.model = cwgame.BotEvade(world_name=world_name,
                                     real_time=real_time,
                                     render=render,
                                     use_predator=use_predator)

        self.belief_state = belief.BeliefState(model=self.model,
                                               agent_name="prey",
                                               other_name="predator",
                                               definition=belief_state_definition,
                                               components=belief_state_components,
                                               probability=belief_state_probability)

        self.observation = self.belief_state.probability_distribution
        self.observation_space = spaces.Box(0, 1, self.belief_state.shape, dtype=np.float32)
        self.action_space = spaces.Discrete(len(self.action_list))

        self.prey_trajectory_length = 0
        self.predator_trajectory_length = 0
        self.episode_reward = 0
        self.step_count = 0
        Environment.__init__(self)

    def set_action(self, action: int):
        self.model.prey.set_destination(self.action_list[action])

    def step(self, action: int):
        self.set_action(action=action)
        model_t = self.model.time + self.time_step
        while self.model.running and self.model.time < model_t:
            self.model.step()
        Environment.step(self, action=action)
        self.step_count += 1
        truncated = (self.step_count >= self.max_step)
        reward = self.reward_function(self.model)
        self.episode_reward += reward

        if self.model.puffed:
            self.model.puffed = False
        if not self.model.running or truncated:
            survived = 1 if not self.model.running and self.model.puff_count == 0 else 0
            info = {"captures": self.model.puff_count,
                    "reward": self.episode_reward,
                    "is_success": survived,
                    "survived": survived,
                    "agents": {}}
        else:
            info = {}
        self.belief_state.tick()
        return self.belief_state.probability_distribution, reward, not self.model.running, truncated, info

    def reset(self,
              options: typing.Optional[dict] = None,
              seed=None):
        Environment.reset(self, options=options, seed=seed)
        self.model.reset()
        self.episode_reward = 0
        self.step_count = 0
        return self.belief_state.probability_distribution, {}

    def close(self):
        self.model.close()
        Env.close(self=self)
