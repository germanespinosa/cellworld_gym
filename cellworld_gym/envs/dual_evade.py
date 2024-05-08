import random
import typing
import cellworld_game as cwgame
import numpy as np
import math

from ..core import Observation
from cellworld_game import AgentState
from gymnasium import Env
from gymnasium import spaces


class DualEvadeObservation(Observation):
    fields = ["self_x",
              "self_y",
              "self_direction",
              "predator_x",
              "predator_y",
              "predator_direction",
              "prey_goal_distance",
              "predator_prey_distance",
              "puffed",
              "puff_cooled_down",
              "finished"]


class DualEvadeObservationWithOther(DualEvadeObservation):
    fields = ["self_x",
              "self_y",
              "self_direction",
              "other_x",
              "other_y",
              "other_direction",
              "predator_x",
              "predator_y",
              "predator_direction",
              "prey_goal_distance",
              "predator_prey_distance",
              "puffed",
              "puff_cooled_down",
              "finished"]


class DualEvadeEnv(Env):

    def __init__(self,
                 world_name: str,
                 use_lppos: bool,
                 use_predator: bool,
                 max_step: int = 300,
                 reward_function: typing.Callable[[DualEvadeObservation], float] = lambda x: 0,
                 time_step: float = .25,
                 render: bool = False,
                 real_time: bool = False,
                 end_on_pov_goal: bool = True,
                 use_other: bool = True):
        self.max_step = max_step
        self.reward_function = reward_function
        self.time_step = time_step
        self.loader = cwgame.CellWorldLoader(world_name=world_name)
        self.use_other = use_other
        if use_other:
            self.observation = DualEvadeObservationWithOther()
            self.other_observation = DualEvadeObservationWithOther()
        else:
            self.other_observation = DualEvadeObservation()
            self.observation = DualEvadeObservation()
        self.observation_space = spaces.Box(-np.inf, np.inf, (len(self.observation),), dtype=np.float32)
        self.end_on_pov_goal = end_on_pov_goal
        if use_lppos:
            self.action_list = self.loader.tlppo_action_list
        else:
            self.action_list = self.loader.full_action_list

        self.action_space = spaces.Discrete(len(self.action_list))

        self.model = cwgame.DualEvade(world_name=world_name,
                                      real_time=real_time,
                                      render=render,
                                      use_predator=use_predator)
        self.prey_trajectory_length = 0
        self.predator_trajectory_length = 0
        self.episode_reward = 0
        self.step_count = 0
        self.prey = self.model.prey_1
        self.other = self.model.prey_2
        self.prey_data = self.model.prey_data_1
        self.other_data = self.model.prey_data_2
        self.other_policy = lambda x: random.randint(0, len(self.action_list) - 1)

    def set_other_policy(self, other_policy: typing.Callable[[DualEvadeObservation], int]):
        self.other_policy = other_policy

    def __update_observation__(self, observation: DualEvadeObservation, prey, prey_data, other):
        observation.self_x = prey.state.location[0]
        observation.self_y = prey.state.location[1]
        observation.self_direction = math.radians(prey.state.direction)

        if self.use_other:
            if self.model.mouse_visible:
                observation.other_x = other.state.location[0]
                observation.other_y = other.state.location[1]
                observation.other_direction = math.radians(other.state.direction)
            else:
                observation.other_x = 0
                observation.other_y = 0
                observation.other_direction = 0

        if self.model.use_predator and prey_data.predator_visible:
            observation.predator_x = self.model.predator.state.location[0]
            observation.predator_y = self.model.predator.state.location[1]
            observation.predator_direction = math.radians(self.model.predator.state.direction)
        else:
            observation.predator_x = 0
            observation.predator_y = 0
            observation.predator_direction = 0

        observation.prey_goal_distance = prey_data.prey_goal_distance
        observation.predator_prey_distance = prey_data.predator_prey_distance
        observation.puffed = prey_data.puffed
        observation.puff_cooled_down = self.model.puff_cool_down
        observation.finished = not self.model.running
        return observation

    def set_actions(self, action: int, other_action: int):
        self.prey.set_destination(self.action_list[action])
        self.other.set_destination(self.action_list[other_action])

    def __step__(self):
        self.step_count += 1
        truncated = (self.step_count >= self.max_step)

        obs = self.__update_observation__(observation=self.observation,
                                          prey=self.prey,
                                          prey_data=self.prey_data,
                                          other=self.other)
        reward = self.reward_function(obs)
        self.episode_reward += reward

        if self.prey_data.puffed:
            self.prey_data.puffed = False

        if self.prey_data.goal_achieved:
            if self.end_on_pov_goal or self.other_data.goal_achieved:
                self.model.stop()

        if not self.model.running or truncated:
            survived = 1 if not self.model.running and self.prey_data.puff_count == 0 else 0
            info = {"captures": self.prey_data.puff_count,
                    "reward": self.episode_reward,
                    "is_success": survived,
                    "survived": survived,
                    "agents": {}}
            for agent_name, agent in self.model.agents.items():
                info["agents"][agent_name] = {}
                info["agents"][agent_name] = agent.get_stats()
        else:
            info = {}
        return obs, reward, not self.model.running, truncated, info

    def replay_step(self, agents_state: typing.Dict[str, AgentState]):
        self.model.set_agents_state(agents_state=agents_state,
                                    delta_t=self.time_step)
        return self.__step__()

    def step(self, action: int):
        other_obs = self.__update_observation__(observation=self.other_observation,
                                                prey=self.other,
                                                prey_data=self.other_data,
                                                other=self.prey)
        other_action = self.other_policy(other_obs)
        self.set_actions(action=action,
                         other_action=other_action)
        model_t = self.model.time + self.time_step
        while self.model.running and self.model.time < model_t:
            self.model.step()
        return self.__step__()

    def __reset__(self):
        self.episode_reward = 0
        self.step_count = 0
        return self.__update_observation__(observation=self.observation,
                                           prey=self.prey,
                                           prey_data=self.prey_data,
                                           other=self.other), {}

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
