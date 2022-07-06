from tensorforce.environments import Environment


# TODO: define environment

class TradeEnv(Environment):

    def __init__(self):
        super().__init__()

    # creating the state space of the environment as input for the agent
    def states(self):
        return state_space

    # creating action space with all actions agent is able to take
    """make that actions are only possible for available contracts, 
    since the data for the two contracts are not equally long, and have some different indices,
    the agent then should only be able to trade the one that is available"""
    def actions(self):
        return action_space

    def reset(self):
        return

    def step(self):
        return