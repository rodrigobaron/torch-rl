
dataset, env = d3rlpy.datasets.get_atari('HopperBulletEnv-v0'')
train_episodes, test_episodes = train_test_split(dataset, test_size=0.1)

# env = gym.make('HopperBulletEnv-v0')
eval_env = gym.make('HopperBulletEnv-v0')

agent = Agent(
    encoder=ConvEncoder(), 
    algo=DiscreteDQN(
        buffer = ReplayBuffer(maxlen=1000000)
    )
)

agent.fit(
    train_episodes, 
    eval_episodes=test_episodes, 
    n_epochs=100, 
    log_path='.logs/', 
    use_gpu=True
)

agent.train(
    env=env, 
    max_steps=100000, 
    log_path='.logs/', 
    record=True, 
    use_gpu=True
)


agent.save(path=".models/hopper")
# agent.load_state(path=".models/hopper")

eval_agent = Agent.load(path=".models/hopper")


env_evaluator = EnvEvaluator(env=eval_env)

env_evaluator.play(
    eval_agent,
    trials=10,
    log_path='.logs/', 
    record=True, 
    use_gpu=True
)
