from gymnasium.envs.registration import register

register(
     id="hc10dt_gym/HC10DTRL-v0",
     entry_point="hc10dt_gym.envs:HC10DTCPEnv",
     max_episode_steps=1000,
)
