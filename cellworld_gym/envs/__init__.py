from gym.envs.registration import register

register(
    id='CellworldBotEvade-v0',
    entry_point='cellworld_gym.envs:BotEvadeEnv',
)