import typing
from ..core import Environment
import cellworld_belief as belief
import numpy as np
import cellworld_game as cwgame
import math
from gymnasium import spaces
from gymnasium import Env
from .bot_evade import BotEvadeObservation


class BotEvadeBeliefEnv(Environment):
    def __init__(self,
                 world_name: str,
                 use_lppos: bool,
                 use_predator: bool,
                 max_step: int = 300,
                 reward_function: typing.Callable[[BotEvadeObservation], float] = lambda x: 0,
                 time_step: float = .25,
                 render: bool = False,
                 real_time: bool = False,
                 belief_state_components: typing.List[belief.BeliefStateComponent] = None,
                 belief_state_definition: int = 100,
                 belief_state_probability: float = 1.0):
        if belief_state_components is None:
            belief_state_components = []
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

        self.observation_space = spaces.Box(-1, 1, self.belief_state.shape, dtype=np.float32)
        self.action_space = spaces.Discrete(len(self.action_list))

        self.prey_trajectory_length = 0
        self.predator_trajectory_length = 0
        self.episode_reward = 0
        self.step_count = 0
        self.prev_agents_state: typing.Dict[str, cwgame.AgentState] = {}
        Environment.__init__(self)

    def __get_observation__(self) -> BotEvadeObservation:
        obs = BotEvadeObservation()
        obs.prey_x = self.model.prey.state.location[0]
        obs.prey_y = self.model.prey.state.location[1]
        obs.prey_direction = math.radians(self.model.prey.state.direction)

        if self.model.use_predator and self.model.predator_visible:
            obs.predator_x = self.model.predator.state.location[0]
            obs.predator_y = self.model.predator.state.location[1]
            obs.predator_direction = math.radians(self.model.predator.state.direction)
        else:
            obs.predator_x = 0
            obs.predator_y = 0
            obs.predator_direction = 0

        obs.prey_goal_distance = self.model.prey_goal_distance
        obs.predator_prey_distance = self.model.predator_prey_distance
        obs.puffed = self.model.puffed
        obs.puff_cooled_down = self.model.puff_cool_down
        obs.finished = not self.model.running
        return obs

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
        reward = self.reward_function(self.__get_observation__())
        self.prev_agents_state = self.model.get_agents_state()
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
        return self.get_observation(), reward, not self.model.running, truncated, info

    def get_observation(self):
        probability_matrix = self.belief_state.probability_distribution.cpu().numpy()
        i, j, _, _, _, _ = self.belief_state.get_location_indices(self.model.prey.state.location)
        probability_matrix[i, j] = -1
        return probability_matrix

    def reset(self,
              options: typing.Optional[dict] = None,
              seed=None):
        Environment.reset(self, options=options, seed=seed)
        self.model.reset()
        self.episode_reward = 0
        self.step_count = 0
        self.prev_agents_state = self.model.get_agents_state()
        return self.get_observation(), {}

    def close(self):
        self.model.close()
        Env.close(self=self)
