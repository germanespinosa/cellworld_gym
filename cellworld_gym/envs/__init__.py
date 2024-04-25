from gymnasium.envs.registration import register
from .bot_evade import BotEvadeEnv, BotEvadeObservation
from .oasis import Oasis, OasisObservation, OasisGoalSequence, OasisGoalRandomSequence

register(
    id='CellworldBotEvade-v0',
    entry_point='cellworld_gym.envs:BotEvadeEnv'
)

register(
    id='CellworldOasis-v0',
    entry_point='cellworld_gym.envs:Oasis'
)
