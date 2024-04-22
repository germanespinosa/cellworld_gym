from gymnasium.envs.registration import register
from .bot_evade import BotEvade, BotEvadeObservation, BotEvadeReward
from .oasis import Oasis, OasisObservation, OasisReward, OasisGoalSequence, OasisGoalRandomSequence

register(
    id='CellworldBotEvade-v0',
    entry_point='cellworld_gym.envs:BotEvade'
)

register(
    id='CellworldOasis-v0',
    entry_point='cellworld_gym.envs:Oasis'
)
