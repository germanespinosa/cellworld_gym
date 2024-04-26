from gymnasium.envs.registration import register
from .bot_evade import BotEvadeEnv, BotEvadeObservation
from .oasis import OasisEnv, OasisObservation
from .dual_evade import DualEvadeEnv, DualEvadeObservation

register(
    id='CellworldBotEvade-v0',
    entry_point='cellworld_gym.envs:BotEvadeEnv'
)

register(
    id='CellworldOasis-v0',
    entry_point='cellworld_gym.envs:OasisEnv'
)

register(
    id='CellworldDualEvade-v0',
    entry_point='cellworld_gym.envs:DualEvadeEnv'
)
