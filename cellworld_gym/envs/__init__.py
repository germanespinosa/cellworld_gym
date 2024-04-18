from gymnasium.envs.registration import register
from .bot_evade import BotEvade

register(
    id='CellworldBotEvade-v0',
    entry_point='cellworld_gym.envs:BotEvade'
)