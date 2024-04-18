from gymnasium.envs.registration import register
from .bot_evade import BotEvade, BotEvadeObservation, BotEvadeReward

register(
    id='CellworldBotEvade-v0',
    entry_point='cellworld_gym.envs:BotEvade'
)