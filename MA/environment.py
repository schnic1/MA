from tensorforce.environments import Environment


# TODO: define environment

class TradeEnv(Environment):

    def __init__(self):
        super().__init__()

    # creating the state space of the environment as input for the agent
    def states(self):
        return state_space

    # creating action space with all actions agent is able to take
    def actions(self):
        return action_space

    def reset(self):
        return

    def step(self):
        return