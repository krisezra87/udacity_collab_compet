# Deep Deterministic Policy Gradient (DDPG) approach to move a double-jointed
# arm (Project Option 1)
# ---

import torch
import numpy as np
from collections import deque
from unityagents import UnityEnvironment
import matplotlib.pyplot as plt
from ddpg import Agent


# First configure the environment
# NOTE: I have configured for LINUX x86_64
env = UnityEnvironment(file_name="Tennis_Linux/Tennis.x86_64")

# get the default brain
brain_name = env.brain_names[0]
brain = env.brains[brain_name]

state_size = brain.vector_observation_space_size
action_size = brain.vector_action_space_size

n_agents = len(env.reset(train_mode=True)[brain_name].agents)

agents = []

# Set up the agents generically for state and action sizes and number of agents
for _ in range(n_agents):
    # Create the agents and let them share a replay buffer
    if not agents:
        memory = None
    else:
        memory = agents[-1].memory

    new_agent = Agent(state_size=state_size,
                      action_size=action_size,
                      memory=memory)

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


# Train the agent
def ddpg(n_episodes=6500, max_t=1000):
    """ Deep Deterministic Policy Gradient (DDPG)

    Params
    ======
        n_episodes (int): maximum number of training episodes
        max_t (int): maximum number of timesteps per episode
    """
    # Initialize the list of episode max scores
    scores = []

    # Initialize a moving average of last 100 scores
    scores_window = deque(maxlen=100)  # last 100 scores

    for i_episode in range(1, n_episodes+1):
        # Reset the environment in training mode according to the standard
        # brain
        env_info = env.reset(train_mode=True)[brain_name]

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

            # Put all the states into a list of lists and feed it to the
            # environment
            # actions = np.random.randn(n_agents, action_size)
            # actions = np.clip(actions, -1, 1)

            env_info = env.step(actions)[brain_name]
            next_states = env_info.vector_observations
            rewards = env_info.rewards
            dones = env_info.local_done

            # Do the next step for all agents
            for agent, state, action, reward, next_state, done in zip(
                    agents, get_state(states), actions, rewards, get_state(next_states), dones):
                agent.step(state, action, reward, next_state, done)

            # Increment the state for the next loop
            states = next_states
            agent_scores += rewards
            if np.any(dones):
                break
        # save most recent episode score in moving average and to the complete
        # list. Recall that the episode score is the maximum score of any agent
        episode_score = np.max(agent_scores)
        scores_window.append(episode_score)
        scores.append(episode_score)

        if i_episode % 100 == 0:
            print(
                  (f'\rEpisode {i_episode}',
                   f'\tAverage Score: {np.mean(scores_window):.2f}')
                  )

        if np.mean(scores_window) >= 0.5:
            print(
                  (f'\nEnvironment solved in {i_episode-100:d} episodes!',
                   f'\tAverage Score: {np.mean(scores_window):.2f}')
                  )
            # If we win, we need to save the checkpoint
            for i, agent in enumerate(agents):
                torch.save(agent.actor_local.state_dict(),
                           F'checkpoint-actor{i}.pth')
                torch.save(agent.critic_local.state_dict(),
                           F'checkpoint-critic{i}.pth')
            break

    # Close the environment
    env.close()
    return scores


# Train the agent and output the scores per episode
score_array = ddpg()

# Plot the scores over each episode
fig = plt.figure()
ax = fig.add_subplot(111)
plt.plot(np.arange(len(score_array)), score_array)
plt.ylabel('Score')
plt.xlabel('Episode #')
plt.show()
