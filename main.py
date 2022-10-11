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

# Set up the agent generically for state and action sizes.  Can't ever be TOO
# portable!
# TODO is there a way to see how many agents there are? Generically?  Need to put some generic looping in here.
agent1 = Agent(state_size=brain.vector_observation_space_size,
               action_size=brain.vector_action_space_size)

agent2 = Agent(state_size=brain.vector_observation_space_size,
               action_size=brain.vector_action_space_size, memory=agent1.memory)

# Train the agent
def ddpg(n_episodes=6500, max_t=1000):
    """ Deep Deterministic Policy Gradient (DDPG)

    Params
    ======
        n_episodes (int): maximum number of training episodes
        max_t (int): maximum number of timesteps per episode
    """
    scores = []  # list containing scores from each episode
    scores_window = deque(maxlen=100)  # last 100 scores
    for i_episode in range(1, n_episodes+1):
        # Reset the environment in training mode according to the standard
        # brain
        env_info = env.reset(train_mode=True)[brain_name]
        num_agents = len(env_info.agents)

        # Get the states from the environment
        states = env_info.vector_observations

        # Get the agents ready for this new episode
        agent1.reset()
        agent2.reset()

        # Initialize the score to zero
        score = np.zeros(2)
        # We will leave a maximum time in here for now
        for _ in range(max_t):
            state1 = states[0]
            state2 = states[1]
            action1 = agent1.act(state1)
            action2 = agent2.act(state2)
            actions = np.stack((action1,action2))
            env_info = env.step(actions)[brain_name]
            next_states = env_info.vector_observations
            rewards = env_info.rewards
            dones = env_info.local_done
            agent1.step(state1, action1, rewards[0], next_states[0], dones[0])
            agent2.step(state2, action2, rewards[1], next_states[1], dones[0])
            states = next_states
            score += rewards
            if np.any(dones):
                break
        max_score = np.max(score)
        scores_window.append(max_score)       # save most recent max score
        scores.append(max_score)              # save most recent score

        if i_episode % 100 == 0:
            print(
                    '\rEpisode {}\tAverage Score: {:.2f}'.format(
                        i_episode, np.mean(scores_window)))

        if np.mean(scores_window) >= 30:
            print(
                    '\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(
                        i_episode-100, np.mean(scores_window)))
            # If we win, we need to save the checkpoint
            torch.save(agent1.actor_local.state_dict(), 'checkpoint-actor1.pth')
            torch.save(agent1.critic_local.state_dict(), 'checkpoint-critic1.pth')
            torch.save(agent2.actor_local.state_dict(), 'checkpoint-actor2.pth')
            torch.save(agent2.critic_local.state_dict(), 'checkpoint-critic2.pth')
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
