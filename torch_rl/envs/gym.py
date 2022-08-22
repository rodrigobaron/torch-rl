import gym


def make_gym_env(env_id:str, seed:int, idx:int, capture_path:str = None):
    def thunk():
        env = gym.make(env_id)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        if capture_path is not None:
            if idx == 0:
                env = gym.wrappers.RecordVideo(env, capture_path)
        env.seed(seed)
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
        return env

    return thunk