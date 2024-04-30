import typing
import cellworld_game as cwgame
import numpy as np
import math

from ..core import Observation
from cellworld_game import AgentState
from gymnasium import Env
from gymnasium import spaces


class BotEvadeObservation(Observation):
    fields = ["prey_x",
              "prey_y",
              "prey_direction",
              "predator_x",
              "predator_y",
              "predator_direction",
              "prey_goal_distance",
              "predator_prey_distance",
              "puffed",
              "puff_cooled_down",
              "finished"]


class BotEvadeEnv(Env):
    def __init__(self,
                 world_name: str,
                 use_lppos: bool,
                 use_predator: bool,
                 max_step: int = 300,
                 reward_function: typing.Callable[[BotEvadeObservation], float] = lambda x: 0,
                 time_step: float = .25,
                 render: bool = False,
                 real_time: bool = False):

        self.max_step = max_step
        self.reward_function = reward_function
        self.time_step = time_step
        self.loader = cwgame.CellWorldLoader(world_name=world_name)
        self.observation = BotEvadeObservation()
        self.observation_space = spaces.Box(-np.inf, np.inf, (len(self.observation),), dtype=np.float32)
        if use_lppos:
            self.action_list = self.loader.tlppo_action_list
        else:
            self.action_list = self.loader.full_action_list

        self.action_space = spaces.Discrete(len(self.action_list))

        self.model = cwgame.BotEvade(world_name="21_05",
                                     real_time=real_time,
                                     render=render,
                                     use_predator=use_predator)
        self.prey_trajectory_length = 0
        self.predator_trajectory_length = 0
        self.episode_reward = 0
        self.step_count = 0

    def __update_observation__(self):
        self.observation.prey_x = self.model.prey.state.location[0]
        self.observation.prey_y = self.model.prey.state.location[1]
        self.observation.prey_direction = math.radians(self.model.prey.state.direction)

        if self.model.use_predator and self.model.predator_visible:
            self.observation.predator_x = self.model.predator.state.location[0]
            self.observation.predator_y = self.model.predator.state.location[1]
            self.observation.predator_direction = math.radians(self.model.predator.state.direction)
        else:
            self.observation.predator_x = 0
            self.observation.predator_y = 0
            self.observation.predator_direction = 0

        self.observation.prey_goal_distance = self.model.prey_goal_distance
        self.observation.predator_prey_distance = self.model.predator_prey_distance
        self.observation.puffed = self.model.puffed
        self.observation.puff_cooled_down = self.model.puff_cool_down
        self.observation.finished = not self.model.running
        return self.observation

    def set_action(self, action: int):
        self.model.prey.set_destination(self.action_list[action])

    def __step__(self):
        truncated = (self.step_count >= self.max_step)
        obs = self.__update_observation__()
        reward = self.reward_function(obs)
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
            for agent_name, agent in self.model.agents.items():
                info["agents"][agent_name] = {}
                info["agents"][agent_name] = agent.get_stats()
        else:
            info = {}
        self.step_count += 1
        return obs, reward, not self.model.running, truncated, info

    def replay_step(self, agents_state: typing.Dict[str, AgentState]):
        self.model.set_agents_state(agents_state=agents_state,
                                    delta_t=self.time_step)
        return self.__step__()

    def step(self, action: int):
        self.set_action(action=action)
        model_t = self.model.time + self.time_step
        while self.model.time < model_t:
            self.model.step()
        return self.__step__()

    def __reset__(self):
        self.episode_reward = 0
        self.step_count = 0
        return self.__update_observation__(), {}

    def reset(self,
              options={},
              seed=None):
        self.model.reset()
        return self.__reset__()

    def replay_reset(self, agents_state: typing.Dict[str, AgentState]):
        self.model.reset()
        self.model.set_agents_state(agents_state=agents_state)
        return self.__reset__()

    def close(self):
        self.model.close()
        Env.close(self=self)
