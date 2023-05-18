import numpy as np
import gym

# Define the Q-learning function
def q_learning(env, num_episodes, alpha, gamma, epsilon):
    # Initialize the Q-table to zeros
    Q = np.zeros((env.observation_space.n, env.action_space.n))
    
    # Loop over episodes
    for episode in range(num_episodes):
        # Reset the environment
        state = env.reset()
        
        # Initialize the total reward for this episode
        total_reward = 0
        
        # Loop over time steps in this episode
        done = False
        while not done:
            # Choose an action using an epsilon-greedy policy
            if np.random.random() < epsilon:
                action = env.action_space.sample()  # explore
            else:
                action = np.argmax(Q[state])  # exploit
            
            # Take the chosen action and observe the next state and reward
            next_state, reward, done, _ = env.step(action)
            
            # Update the Q-table
            Q[state, action] += alpha * (reward + gamma * np.max(Q[next_state]) - Q[state, action])
            
            # Update the total reward
            total_reward += reward
            
            # Update the state for the next iteration
            state = next_state
        
        # Print the total reward for this episode
        print(f"Episode {episode + 1}: Total reward = {total_reward}")
        
    return Q


# Create the environment
env = gym.make('Taxi-v3')

# Set the hyperparameters
num_episodes = 1000
alpha = 0.1
gamma = 0.99
epsilon = 0.1

# Run the Q-learning algorithm
Q = q_learning(env, num_episodes, alpha, gamma, epsilon)

# Close the environment
env.close()
