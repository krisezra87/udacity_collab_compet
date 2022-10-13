# Deep Deterministic Policy Gradient (DDPG) approach to train tennis rackets
# ---

import torch
from unityagents import UnityEnvironment
from ddpg import Agent
import numpy as np

# First configure the environment
# NOTE: I have configured for LINUX x86_64
env = UnityEnvironment(file_name="Tennis_Linux/Tennis.x86_64")

# get the default brain
brain_name = env.brain_names[0]
brain = env.brains[brain_name]

state_size = brain.vector_observation_space_size
action_size = brain.vector_action_space_size

n_agents = len(env.reset(train_mode=False)[brain_name].agents)

agents = []

# Set up the agents generically for state and action sizes and number of agents
for idx in range(n_agents):
    # Create the agents and let them share a replay buffer
    if not agents:
        memory = None
    else:
        memory = agents[-1].memory

    new_agent = Agent(state_size=state_size,
                      action_size=action_size,
                      memory=memory)

    network_types = ['actor', 'critic']
    network_scopes = ['local', 'target']
    for ntype in network_types:
        network = torch.load(f'checkpoint-{ntype}{idx}.pth')
        for scope in network_scopes:
            net_obj = getattr(new_agent, f'{ntype}_{scope}')
            net_obj.load_state_dict(network)
    agents.append(new_agent)


def get_state(states):
    # Note here that the states are stacks of 3 observations...
    # These observations go back in time, and we are interested in
    # the last one because it's the current state
    # The state itself is the racket position and velocity, then
    # ball position and velocity

    states_out = []
    for state in states:
        state_set = np.reshape(state, (3, state_size))
        current_pos = state_set[-1]
        states_out.append(current_pos)
    return states_out


# demo the agents
def ddpg_demo(max_t=1200):
    """ Deep Deterministic Policy Gradient (DDPG)

    Params
    ======
        max_t (int): maximum number of timesteps per episode
    """
    # Reset the environment in training mode according to the standard
    # brain
    env_info = env.reset(train_mode=False)[brain_name]

    # Get the states from the environment
    states = env_info.vector_observations

    # Get the agents ready for this new episode
    for agent in agents:
        agent.reset()

    # Initialize the episode score to zero for each agent
    agent_scores = np.zeros(n_agents)
    # We will leave a maximum time in here for now
    for _ in range(max_t):
        # Go get the actions for each agent, given the current state
        actions = []
        for agent, state in zip(agents, get_state(states)):
            # Note here that the states are stacks of 3 observations...
            # These observations go back in time, and we are interested in
            # the last one because it's the current state
            # The state itself is the racket position and velocity, then
            # ball position and velocity
            action = agent.act(state)
            if len(actions):
                actions = np.append(actions, action, axis=0)
            else:
                # The first time
                actions = action

        env_info = env.step(actions)[brain_name]
        next_states = env_info.vector_observations
        rewards = env_info.rewards
        dones = env_info.local_done

        # # Do the next step for all agents
        # for agent, state, action, reward, next_state, done in zip(
        #         agents, get_state(states), actions, rewards, get_state(next_states), dones):
        #     agent.step(state, action, reward, next_state, done)

        # Increment the state for the next loop
        states = next_states
        agent_scores += rewards
        if np.any(dones):
            break
        # save most recent episode score in moving average and to the complete
        # list. Recall that the episode score is the maximum score of any agent
        episode_score = np.max(agent_scores)
    env.close()
    return episode_score


# Do a run and output the score
score = ddpg_demo()
print(f"Score: {score}")
