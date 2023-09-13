from models import LexDNN, DNN
import numpy as np
from collections import deque, namedtuple
import torch.optim as optim
import random
import torch
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
from params import TrainingParameters
import gym
import os
import copy

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning, module="gym")
warnings.filterwarnings("ignore", category=RuntimeWarning, module="numpy")



#===================#
#   ReplayBuffer    #
#===================#
# A queue containing the agent's memory in the named tuple: (state, action, reward, next_state, done)
#=================================================================================
class ReplayBuffer:

    def __init__(self, buffer_size, batch_size, device):

        self.device = device
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.experiences = namedtuple("Experience",
                            field_names=["state",
                                           "action",
                                           "reward",
                                           "next_state",
                                           "done"])

    #Add informations to queue
    def add(self, state, action, reward, next_state, done):

        e = self.experiences(state, action, reward, next_state, done)
        self.memory.append(e)

    #Randomly sample batch_size experiences from the queue
    def sample(self, sample_all=False):

        if sample_all:

            experiences = self.memory

        else:

            experiences = random.sample(self.memory, k=self.batch_size)

        #Cast the arrays from the namedtuple to tensors and host them on the preferred device
        states = torch.from_numpy(np.vstack([e.state.cpu() for e in experiences if e is not None])).float().to(self.device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).float().to(self.device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(self.device)
        next_states = torch.from_numpy(np.vstack([e.next_state.cpu() for e in experiences if e is not None])).float().to(self.device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(self.device)        

        return (states, actions, rewards, next_states, dones)

    #Len method definition
    def __len__(self):

        return len(self.memory)


#===================#
#       LexCDQN     # =====================================================================================
#===================#
# Lexicographic Continuous DQN class; contains all the parameters for the trainin and the network itself
#   
#       Arguments:
#           - train_params          ->A TrainingParameters dataclass
#           - env                   ->A gym environment; Currently only supports Ant
#           - state_size            ->Network's input size
#           - action_size           ->Action size for the environment
#===========================================================================================================
class LexCDQN:

    def __init__(self, train_params, env, state_size, action_size):

        self.device = torch.device("cpu")
        
        self.t = 0  # total number of frames observed
        self.gamma: float = train_params.gamma  # discount

        #Epsilon parameters
        self.epsilon: float = train_params.epsilon
        self.epsilon_decay: float = train_params.epsilon_decay
        self.epsilon_min: float = train_params.epsilon_min
        self.epsilon_decay_start: int = train_params.epsilon_decay_start

        #Lexicographic parameters
        self.slack: float = train_params.slack
        self.loss_threshold: float = train_params.loss_threshold

        self.update_every: int = train_params.update_every
        self.batch_size: int = train_params.batch_size
        self.buffer_size: int = train_params.buffer_size
        self.sample_size: int = train_params.sample_size
        self.hidden_size: int = train_params.hidden_size
        self.reward_size: int = train_params.reward_size


        self.nohid: bool = train_params.nohid
        self.bias: bool = train_params.bias

        #Environment parameters
        self.env = env
        self.action_low = self.env.action_space.low
        self.action_high = self.env.action_space.high
        self.sample_step = (self.action_high[0] - self.action_low[0]) / self.sample_size
        self.action_range = np.round(np.arange(-1, 1, self.sample_step), 3)

        self.action_size: int = action_size
        self.model = LexDNN(state_size, self.action_size, self.sample_size, self.hidden_size, self.reward_size, self.nohid, self.bias)
        self.target_model = LexDNN(state_size, self.action_size, self.sample_size, self.hidden_size, self.reward_size, self.nohid, self.bias)

        self.memory = ReplayBuffer(self.buffer_size, self.batch_size, self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=train_params.learning_rate)
        self.criterion = torch.nn.MSELoss() 

    #===========#
    #   act()   # ==============================================================================================================
    #===========#
    # Given the state as the input, returns the action chosen by the network.
    # Using an epsilon-greedy approach, with a probability of epsilon, random actions are chosen among the sampled action space
    # if test = True, the network is in test mode and only actions from the network are chosen.
    #===========================================================================================================================
    def act(self, state, test = False):

        if test == False:

            #Choose a random action in the sampled action space with a probability of epsilon
            if np.random.choice([True, False], p=[self.epsilon, 1 - self.epsilon]):

                action = np.array([random.choice(self.action_range) for i in range(self.action_size)])
                return action

        #Retrieve the q_values from the network and get the permissible actions
        #> See function defined below
        q_values = self.model(state)[0]
        action = self.get_permissible_actions(q_values)   

        #Choose randomly one among permissible actions overwritinga action array
        for i, a in enumerate(action):

            action[i] = self.action_range[random.choice(a)]

        return np.array(action)

    #===============================#
    #   get_permissible_actions()   # ========================================================================================================
    #===============================#
    # Instead of using the argmax to choose the action, in this lexicographic version, the slack variable is used as a tolerance
    # to see which actions are closer to the max q_value.
    # This is done for every group of q_values for every reward. Then, actions which maximize more than one objective are chosen, if present.
    #=========================================================================================================================================
    def get_permissible_actions(self, Q):

        permissible_actions = []

        #For every action, and for every reward:
        for i in range(self.action_size):

            for j in range(self.reward_size):

                #Get the q_values for the correct reward and action and store its max
                rew_Q = Q[i,:,j]
                m = rew_Q.max().item()

                #Store the actions from the sampled action space whoose q_value is near the max (for the first objective)
                if j == 0:
                    
                    first_objective_actions = [idx for idx, a in enumerate(self.action_range) if  rew_Q[idx] >= m - self.slack * abs(m)]
                    permissible_actions.append(copy.deepcopy(first_objective_actions))                   

                #Delete any action that does not maximize any other objective
                else: 

                    if len(first_objective_actions) != 0:

                        for k, action_index in enumerate(first_objective_actions):

                            if not rew_Q[action_index] >= m - self.slack * abs(m):

                                first_objective_actions.pop(k)

            #If no actions are left, use the first objective's ones
            if len(first_objective_actions) != 0:

                permissible_actions[i] = first_objective_actions

        multiple_actions = np.array([len(permissible_actions[i])>1 for i in range(len(permissible_actions))])
        return permissible_actions

    #===========#
    #   step()  # ==========================================================================================
    #===========#
    # Function that gets executed every timestep by the agent. Haldles internal time and update of networks
    #=======================================================================================================
    def step(self, state, action, reward, next_state, done):

        #Increase internal time count and add current interaction to the Replay Buffer
        self.t += 1
        self.memory.add(state, action, reward, next_state, done)      

        #Perform a network update 
        if self.t % self.update_every == 0 and len(self.memory) > self.batch_size:

            experience = self.memory.sample()
            avg_losses = self.update(experience)

            return avg_losses

        else:

            return np.array([np.nan, np.nan, np.nan])

    #===============#
    #   decay()     # ==============================================================
    #===============#
    # Decreases the epsilon value every episode, starting from epsilon_decay_start
    # down to a minimum of epsilon_min
    # ==============================================================================
    def decay(self, episode):

        if episode < self.epsilon_decay_start:

            return

        else:

            if self.epsilon > self.epsilon_min:

                self.epsilon *= self.epsilon_decay

    #===============#
    #   update()    # ==========================================================================================================
    #===============#
    # Netowork update function. It calculates the loss for the first reward: if it is lower than loss_threshold, it calculates
    # the loss relative to the second reward and backpropagates it. The same is done for the third objective.
    # ==========================================================================================================================
    def update(self, experiences):

        #Initialize loss vectors
        loss_vec = np.empty(3)
        loss_vec[:] = np.nan

        #Set the model in train mode
        self.model.train()


        loss = self.calculate_loss(experiences, 0)        

        if abs(loss) <= self.loss_threshold:

            loss = self.calculate_loss(experiences, 1) 

            if abs(loss) <= self.loss_threshold:

                # print('Minimizing for third')
                loss = self.calculate_loss(experiences, 2)
                loss_vec[2] = loss

            else:

                # print("Minimizing for second")
                loss_vec[1] = loss

        else:

            # print("Minimizing for first")
            loss_vec[0] = loss

    
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.model.eval()

        return loss_vec

    #=======================#
    #   calculate_loss()    # ==========================================
    #=======================#
    # Function that calculates the loss relative to the given objective
    # ==================================================================
    def calculate_loss(self, experiences, objective):

        #Get the list of states, actions, rewards, next_states and dones from the sampled ReplayBuffer 
        states, actions, rewards, next_states, done = experiences  

        q_batch = np.empty((self.action_size, self.sample_size,0))
        target_q_batch = np.empty((self.action_size, self.sample_size,0))

        #For every sampled interaction:
        for i in range(len(states)):

            #Get the q_values and next_q_values reative to the current objective from the model and target model, using actual state and next state
            q_values = self.model(states[i]).detach().squeeze().numpy()[:,:,objective]

            with torch.no_grad():
                next_q_values = self.target_model(next_states[i]).detach().squeeze().numpy()[:,:,objective]
         
            #Initialize the target q_values as the next_q_values
            target_q_values = copy.deepcopy(q_values)

            #For every action:
            for j in range(0, self.action_size):

                #Get the index from the sampled action space of the choosen action sample
                indexes = [z for z, a in enumerate(self.action_range) if a == round(actions[i][j].squeeze().detach().item(), 3)]

                # Change the target_q_value at the correct q_value and update if with the Q-learning formula
                target_q_values[j, indexes] = rewards[i][objective] + ( (1-done[i]) * self.gamma * next_q_values.max()) 

            q_batch = np.append(q_batch, q_values)
            target_q_batch = np.append(target_q_batch, target_q_values)

        #Calculate loss using the target_q_values and the q_values with MSE
        loss = self.criterion(torch.from_numpy(q_batch).requires_grad_(True), torch.from_numpy(target_q_batch).requires_grad_(True)).to(self.device)
        return loss

    #============================
    #   update_target_model()   # ===================
    #============================
    # Updates target model by copying the actual one
    # ===============================================
    def update_target_model(self):

        self.target_model = copy.deepcopy(self.model)

    #===================
    #   save_model()   #
    #===================
    # Saves the model
    # ==================
    def save_model(self, root):

        torch.save(self.model.state_dict(), '{}-model.pt'.format(root))
        
    #===================
    #   load_model()   #
    #===================
    # Loads the model
    # ==================
    def load_model(self, root):

        self.model.load_state_dict(torch.load('{}-model.pt'.format(root)))


    @classmethod
    #================
    #   train()     # ==========================================================================================================
    #================
    # Handles the training and testing loop.
    #   
    #   Arguments:  
    #       - env               ->A gym environment. Currently only supports 'Ant'
    #       - seed              ->An int to be used as seed for random, torch and numpy
    #       - train_params      ->TrainingParameters dataclass; will be used to retrieve training and network parameters
    #       - session_pref      ->An str with the path where the training will be saved
    #       - show_prog_bar     ->Wether or not to show progress bars in the console
    #       - rew_mode          ->The reward mode for compute_rewards; will be used to distribute different rewards
    #       - test              ->Wether or not to perform testing at the end of the training
    #       - render            ->Wether or not render the tests
    #==========================================================================================================================
    def train(self, env, seed,  train_params: TrainingParameters, session_pref: str, show_prog_bar=True, rew_mode = 0, test = True, render = False):

        self.device = torch.device("cpu")
        negative_rewards = [3]

        #Initialize agent
        agent = LexCDQN(train_params, env, env.observation_space.shape[0], env.action_space.shape[0])

        #Set seed for repeatability
        random.seed(seed)
        torch.manual_seed(seed)
        np.random.seed(seed)

        #Create training directories
        run_dir = os.path.join(session_pref, train_params.env_name, train_params.agent_name + "-" + str(seed))
        os.makedirs(run_dir, exist_ok=True)

        #Initialize TensorBoard logger and save the training parameters
        writer = SummaryWriter(log_dir=run_dir)
        train_params.render_to_file(run_dir + ".params")

        #Initialize iterator for episodes loop
        interact_iter = range(train_params.num_episodes)
        if show_prog_bar:

            interact_iter = tqdm(interact_iter, colour="green", desc="Episode")

        #Best reward array, used for best model saving and average episode array
        best_reward = -1000
        avg_array = np.zeros(7)
        
        #Training loop
        for episode in interact_iter:

            #Reset state and cast to device
            state = env.reset()[0]
            state = torch.tensor(state).float().to(self.device)          

            #Register spawn location ad append to x_pos and y_pos   
            spawn = env.step([0.0 for i in range(0, env.action_space.shape[0])])[4]
            spawn_x, spawn_y = spawn["x_position"], spawn["y_position"]
            x_pos = [spawn_x]
            y_pos = [spawn_y]

            #Initialize cumulative reward array, cumulative network rewards array, and average episode loss array, 
            rew_array = np.zeros(5)
            totrew = np.zeros(3)
            avg_losses = np.empty(3)
            avg_losses[:] = np.nan

            #Termination condition 
            done = False

            #Interaction loop
            while not done:

                #Get the action from the netowrk and interact with the environment
                action = agent.act(state)          
                next_state, reward, done, _, info = env.step(action)

                #Compute interaction reward and cumulative rewards
                reward, rew_array = LexCDQN.compute_rewards(rew_array, info, rew_mode)
                
                #Update cumulative network reward
                totrew += np.array(reward)

                #Check if the Ant is healthy
                if state[0]<=0.26 or state[0]>1.0:
              
                    done = True            
                
                #Update network and stack average losses
                action = torch.from_numpy(action).squeeze().cpu().float()         
                next_state = torch.tensor(next_state).float().to(self.device)           
                avg_interaction_loss = agent.step(state, action, reward, next_state, done)
                avg_losses = np.vstack((avg_losses, avg_interaction_loss))

                #Go to next state
                state = next_state

                #Update Ant's path
                x_pos.append(info["x_position"])
                y_pos.append(info["y_position"])

            #===============
            #END OF EPISODE#
            #===============
            
            #Decay epsilon parameter and update target model
            agent.decay(episode)
            agent.update_target_model()

            # Create the array to calculate the average reward per episode anc average episode loss
            # Net rew 1 - Net rew 2 - Net rew 3 - CTRL - ORG - HLT - FWD
            avg_array = np.vstack((avg_array, [np.append(totrew, rew_array[1:])]))     
            average_episode_loss = [np.mean(avg_losses[:,i][~np.isnan(avg_losses[:,i])]) for i in range(len(avg_losses[0,:]))]

            
            #Calculate the average reward over 100 epochs
            if episode>=100: 

                avg = [np.mean(avg_array[episode-99:episode,i]) for i in range(0,avg_array.shape[1])]         
                    
            else:

                avg = [np.mean(avg_array[0:episode,i]) for i in range(0,avg_array.shape[1])]

            #Save the model if the average is higher than the best average value
            if rew_mode not in negative_rewards:

                if totrew[0] > best_reward and episode > 10:

                    agent.save_model(os.path.join(run_dir, "best"))
                    best_reward = totrew[0]

            else:

                if totrew[0] > best_reward and episode > 50:

                    agent.save_model(os.path.join(run_dir, "best"))
                    best_reward = totrew[0]

            
            #Save the model every save_every_n epochs
            if train_params.save_every_n is not None and episode % train_params.save_every_n == 0:
                agent.save_model(run_dir)            

            #Save a plot of the Ant's path every save_path_every_n epochs
            if episode%(train_params.num_episodes/train_params.save_path_every_n) == 0 and episode != 0:
                fig = plt.figure()
                path = fig.add_subplot(1,1,1)
                path.plot(x_pos, y_pos)
                path.plot(spawn_x, spawn_y, marker = '*')
                path.set_title("Path")

                fig.canvas.draw()
                img = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
                img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
                img = img / 255.0
                img = np.swapaxes(img, 0, 2)
                
                writer.add_figure(f"{train_params.env_name}/Path:", fig, episode)
                writer.add_image(f"{train_params.env_name}/Path:", img, episode)

            #Write metrics with TensorBoard writer
            writer.add_scalar(f"{train_params.env_name}/Average Episode Loss - 1:", average_episode_loss[0], episode)
            writer.add_scalar(f"{train_params.env_name}/Average Episode Loss - 2:", average_episode_loss[1], episode)
            writer.add_scalar(f"{train_params.env_name}/Average Episode Loss - 3:", average_episode_loss[2], episode)
            writer.add_scalar(f"{train_params.env_name}/Network Reward - 1:", totrew[0], episode)
            writer.add_scalar(f"{train_params.env_name}/Network Reward - 2:", totrew[1], episode)
            writer.add_scalar(f"{train_params.env_name}/Network Reward - 3:", totrew[2], episode)
            writer.add_scalar(f"{train_params.env_name}/Reward Forward:", rew_array[4], episode)
            writer.add_scalar(f"{train_params.env_name}/Cost:", rew_array[1], episode)
            writer.add_scalar(f"{train_params.env_name}/Distance from origin:", rew_array[2], episode)
            writer.add_scalar(f"{train_params.env_name}/Survive:", rew_array[3], episode)

            writer.add_scalar(f"{train_params.env_name}-Avg/ Network Reward - 1:", avg[0], episode)
            writer.add_scalar(f"{train_params.env_name}-Avg/ Network Reward - 2:", avg[1], episode)
            writer.add_scalar(f"{train_params.env_name}-Avg/ Network Reward - 3:", avg[2], episode)
            writer.add_scalar(f"{train_params.env_name}-Avg/ Reward Forward:", avg[6], episode)
            writer.add_scalar(f"{train_params.env_name}-Avg/ Cost:", avg[3], episode)
            writer.add_scalar(f"{train_params.env_name}-Avg/ Distance from origin:", avg[4], episode)
            writer.add_scalar(f"{train_params.env_name}-Avg/ Survive:", avg[5], episode)
            
        #Save last model
        agent.save_model(run_dir)

        #Run test procedure
        if test:

            LexCDQN.test(agent, writer, run_dir, rew_mode, train_params, show_prog_bar, render)

        writer.flush()

    @classmethod
    #================
    #   test()      # =========================================================================================================
    #================
    # Handles the testing loop.
    #   
    #   Arguments:  
    #       - env               ->A gym environment. Currently only supports 'Ant'
    #       - agent             ->The LexCDQN class already initialized
    #       - writer            ->The TensorBoard SummaryWriter object used during training
    #       - run_dir           ->A str containing the model's path
    #       - rew_mode          ->The reward mode for compute_rewards; will be used to distribute different rewards
    #       - train_params      ->TrainingParameters dataclass; will be used to retrieve training and network parameters
    #       - show_prog_bar     ->Wether or not to show progress bars in the console
    #       - render            ->Wether or not render the tests
    #==========================================================================================================================
    def test(self, agent, writer, run_dir, rew_mode, train_params: TrainingParameters, show_prog_bar=True, render = False, infinite = False):

        self.device = torch.device("cpu")

        if type(writer) == str:

            writer = SummaryWriter(log_dir=writer)

        #Initialize env
        if render:
            test_env = gym.make('Ant-v4', render_mode = "human")
        else:
            test_env = gym.make('Ant-v4')

        #Load best model
        agent.load_model(os.path.join(run_dir, "best"))
        agent.model.eval()   

        #Initialize test loop iterator 
        test_iter = range(train_params.num_test)
        if show_prog_bar:
            test_iter = tqdm(test_iter, colour="blue", desc="Episode")
        
        #Initialize average episode reward array
        avg_array = np.zeros(7)

        #Test loop
        for episode in test_iter:       
   
            #Initialize state and save spawn location
            state = test_env.reset()[0]
            spawn = test_env.step([0.0 for i in range(0, test_env.action_space.shape[0])])[4]
            spawn_x, spawn_y = spawn["x_position"], spawn["y_position"]
            x_pos = [spawn_x]
            y_pos = [spawn_y]

            state = torch.tensor(state).float().to(self.device) 

            #Initialize cumulative rewards and network rewards array
            rew_array = np.zeros(5)
            totrew = np.array([0.0, 0.0, 0.0])

            #Initialize last action
            last_action = np.zeros(test_env.action_space.shape[0])
            t=0

            #Initialize termination condition
            done = False

            #Interaction loop
            while not done:

                #Increase timer
                t+=1

                #Render environment
                if render:
                    test_env.render()

                #Interact with the environment
                action = agent.act(state, test = True)
                next_state, reward, done, _, info = test_env.step(action)
                next_state = torch.tensor(next_state).float().to(self.device)
                reward, rew_array = LexCDQN.compute_rewards(rew_array, info, rew_mode)
                
                #Update cumulative rewards, state and path
                totrew += reward                            
                state = next_state
                x_pos.append(info["x_position"])
                y_pos.append(info["y_position"])

                #Termination condition if the agent gets stuck in an infinite loop
                #>In one of the training, the Ant started "vibrating" (performing alternately the same two slightly
                #  different actions) in place and moving slowly but infinitely
                if not infinite:

                    if abs(reward[0]) > 6000:

                        done = True

                    #Check if the Ant is healthy
                    if state[0]<=0.26 or state[0]>1.0:
                        
                        done = True

                    #Over a certain time, check if the agent is moving at all
                    if t > 1000:

                        if (last_action == action).all():
                            done = True
                            writer.add_scalar(f"{train_params.env_name}-[Point]Test/Inactive Terminations", 1, episode)

                        else:

                            last_action = action

                    #Termination condition for time taken
                    if t > 100000:

                        done = True        
            

            avg_array = np.vstack((avg_array, [np.append(totrew, rew_array[1:])]))

            #Save a plot of the Ant's path every save_path_every_n_test epochs
            if episode%(train_params.num_episodes/train_params.save_path_every_n_test) == 0 and episode != 0:
                fig = plt.figure()
                path = fig.add_subplot(1,1,1)
                path.plot(x_pos, y_pos)
                path.plot(spawn_x, spawn_y, marker = '*')
                path.set_title("Path")

                fig.canvas.draw()
                img = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
                img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
                img = img / 255.0
                img = np.swapaxes(img, 0, 2)
                
                writer.add_figure(f"{train_params.env_name}/Path:", fig, episode)
                writer.add_image(f"{train_params.env_name}/Path:", img, episode)

            #Calculate average episode rewards over 100 epochs
            if episode>=100: 

                avg = [np.mean(avg_array[episode-99:episode,i]) for i in range(0,avg_array.shape[1])]         
                    
            else:
                avg = [np.mean(avg_array[0:episode,i]) for i in range(0,avg_array.shape[1])]
            
        
            #Write metrics with TensorBoard writer
            writer.add_scalar(f"{train_params.env_name}-Test/Network Reward - 1:", totrew[0], episode)
            writer.add_scalar(f"{train_params.env_name}-Test/Network Reward - 2:", totrew[1], episode)
            writer.add_scalar(f"{train_params.env_name}-Test/Network Reward - 3:", totrew[2], episode)
            writer.add_scalar(f"{train_params.env_name}-Test/Reward Forward:", rew_array[4], episode)
            writer.add_scalar(f"{train_params.env_name}-Test/Cost:", rew_array[1], episode)
            writer.add_scalar(f"{train_params.env_name}-Test/Distance from origin:", rew_array[2], episode)
            writer.add_scalar(f"{train_params.env_name}-Test/Survive:", rew_array[3], episode)

            writer.add_scalar(f"{train_params.env_name}-Test-Avg/ Network Reward - 1:", avg[0], episode)
            writer.add_scalar(f"{train_params.env_name}-Test-Avg/ Network Reward - 2:", avg[1], episode)
            writer.add_scalar(f"{train_params.env_name}-Test-Avg/ Network Reward - 3:", avg[2], episode)
            writer.add_scalar(f"{train_params.env_name}-Test-Avg/ Reward Forward:", avg[6], episode)
            writer.add_scalar(f"{train_params.env_name}-Test-Avg/ Cost:", avg[3], episode)
            writer.add_scalar(f"{train_params.env_name}-Test-Avg/ Distance from origin:", avg[4], episode)
            writer.add_scalar(f"{train_params.env_name}-Test-Avg/ Survive:", avg[5], episode)

        test_env.close()

    @classmethod
    #=======================#
    #   compute_array()     # ============================================================
    #=======================#
    # Returns different rewards array based on the mode selected as well as the current
    # cumulative rewards earned
    #=====================================================================================
    def compute_rewards(self, rew_array, information, mode = 0):

        rew_array[4] += information["reward_forward"]
        rew_array[3] += information["reward_survive"]
        rew_array[2] += information["distance_from_origin"]
        rew_array[1] += information["reward_ctrl"]


        #default
        if mode == 0:

            network_reward = [5*information["reward_forward"], information["reward_survive"], information["reward_ctrl"]]           # [5FWD, HLT, CTR]

        if mode == 1:

            network_reward = [information["reward_survive"], 5*information["reward_forward"], information["reward_ctrl"]]           # [HLT, 5FWD, CTR]

        if mode == 2:

            network_reward = [information["distance_from_origin"], information["reward_survive"], information["reward_ctrl"]]       # [ORG, HLT, CTR]

        if mode == 3:

            network_reward = [information["reward_ctrl"], information["distance_from_origin"], information["reward_survive"]]       # [CTR, ORG, HLT]

        if mode == 4:

            network_reward = [information["reward_survive"], information["distance_from_origin"], information["reward_ctrl"]]       # [CTR, ORG, HLT]

        return network_reward, rew_array








#=========================#
#       ContinuousDQN     # ==================================================================
#=========================#
# Continuous DQN class; contains all the parameters for the trainin and the network itself
#   
#       Arguments:
#           - train_params          ->A TrainingParameters dataclass
#           - env                   ->A gym environment; Currently only supports Ant
#           - state_size            ->Network's input size
#           - action_size           ->Action size for the environment
#==============================================================================================
class ContinuousDQN:

    def __init__(self, train_params, env, state_size, action_size):

        self.device = torch.device("cpu")
        
        self.t = 0  # total number of frames observed
        self.gamma: float = train_params.gamma  # discount

        #Epsilon parameters
        self.epsilon: float = train_params.epsilon
        self.epsilon_decay: float = train_params.epsilon_decay
        self.epsilon_min: float = train_params.epsilon_min
        self.epsilon_decay_start: int = train_params.epsilon_decay_start

        self.update_every: int = train_params.update_every
        self.batch_size: int = train_params.batch_size
        self.buffer_size: int = train_params.buffer_size
        self.sample_size: int = train_params.sample_size
        self.hidden_size: int = train_params.hidden_size

        self.nohid: bool = train_params.nohid
        self.bias: bool = train_params.bias

        #Environment parameters
        self.env = env
        self.action_low = self.env.action_space.low
        self.action_high = self.env.action_space.high
        self.sample_step = (self.action_high[0] - self.action_low[0]) / self.sample_size
        self.action_range = np.round(np.arange(-1, 1, self.sample_step), 3)

        self.action_size: int = action_size
        self.model = DNN(state_size, self.action_size, self.sample_size, self.hidden_size, self.nohid, self.bias)
        self.target_model = DNN(state_size, self.action_size, self.sample_size, self.hidden_size, self.nohid, self.bias)

        self.memory = ReplayBuffer(self.buffer_size, self.batch_size, self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=train_params.learning_rate)
        self.criterion = torch.nn.MSELoss() 


    #===========#
    #   act()   # ==============================================================================================================
    #===========#
    # Given the state as the input, returns the action chosen by the network.
    # Using an epsilon-greedy approach, with a probability of epsilon, random actions are chosen among the sampled action space
    # if test = True, the network is in test mode and only actions from the network are chosen.
    #===========================================================================================================================
    def act(self, state, test = False):

        if test == False:

            #Choose a random action in the sampled action space with a probability of epsilon
            if np.random.choice([True, False], p=[self.epsilon, 1 - self.epsilon]):

                action = np.array([random.choice(self.action_range) for i in range(self.action_size)])
                return action

        #Retrieve the q_values from the network and choose the samples accordingly using argmax     
        q_values = self.model(state)
        action_indexes = [q_values[0][i].argmax() for i in range(0, self.action_size)]
        action = self.action_range[action_indexes]       

        return action


    #===========#
    #   step()  # ==========================================================================================
    #===========#
    # Function that gets executed every timestep by the agent. Haldles internal time and update of networks
    #=======================================================================================================
    def step(self, state, action, reward, next_state, done):

        #Increase internal time count and add current interaction to the Replay Buffer
        self.t += 1
        self.memory.add(state, action, reward, next_state, done)      

        #Perform a network update 
        if self.t % self.update_every == 0 and len(self.memory) > self.batch_size:

            experience = self.memory.sample()
            self.update(experience)


    #===============#
    #   decay()     # ==============================================================
    #===============#
    # Decreases the epsilon value every episode, starting from epsilon_decay_start
    # down to a minimum of epsilon_min
    # ==============================================================================
    def decay(self, episode):

        if episode < self.epsilon_decay_start:

            return

        else:

            if self.epsilon > self.epsilon_min:

                self.epsilon *= self.epsilon_decay

    #===============#
    #   update()    # ====================
    #===============#
    # Netowork update function.
    # ====================================
    def update(self, experiences):

        states, actions, rewards, next_states, done = experiences
        self.model.train()

        #For every sampled interaction:
        for i in range(len(states)):

            #Get the q_values and next_q_values reative to the current objective from the model and target model, using actual state and next state
            q_values = self.model(states[i]).squeeze().detach()

            with torch.no_grad():
                next_q_values = self.target_model(next_states[i]).squeeze().detach()
            
            #Initialize the target q_values as the next_q_values
            target_q_values = self.model(states[i]).squeeze().detach()

            #For every action:
            for j in range(0, self.action_size):

                #Get the index from the sampled action space of the choosen action sample
                indexes = [z for z, a in enumerate(self.action_range) if a == round(actions[i][j].squeeze().detach().item(), 3)]

                # Change the target_q_value at the correct q_value and update if with the Q-learning formula
                target_q_values[j, indexes] = rewards[i] + ( (1-done[i]) * self.gamma * next_q_values.max()) 

            #Calculate loss using the target_q_values and the q_values with MSE
            loss = self.criterion(q_values.requires_grad_(True), target_q_values.requires_grad_(True)).to(self.device)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        self.model.eval()


    #============================
    #   update_target_model()   # ===================
    #============================
    # Updates target model by copying the actual one
    # ===============================================
    def update_target_model(self):
        self.target_model = copy.deepcopy(self.model)

    #===================
    #   save_model()   #
    #===================
    # Saves the model
    # ==================
    def save_model(self, root):
        torch.save(self.model.state_dict(), '{}-model.pt'.format(root))
        

    #===================
    #   load_model()   #
    #===================
    # Loads the model
    # ==================
    def load_model(self, root):
        self.model.load_state_dict(torch.load('{}-model.pt'.format(root)))


    @classmethod
    #================
    #   train()     # ==========================================================================================================
    #================
    # Handles the training and testing loop.
    #   
    #   Arguments:  
    #       - env               ->A gym environment. Currently only supports 'Ant'
    #       - seed              ->An int to be used as seed for random, torch and numpy
    #       - train_params      ->TrainingParameters dataclass; will be used to retrieve training and network parameters
    #       - session_pref      ->An str with the path where the training will be saved
    #       - show_prog_bar     ->Wether or not to show progress bars in the console
    #       - rew_mode          ->The reward mode for compute_rewards; will be used to distribute different rewards
    #       - test              ->Wether or not to perform testing at the end of the training
    #       - render            ->Wether or not render the tests
    #==========================================================================================================================
    def train(self, env, seed,  train_params: TrainingParameters, session_pref: str, show_prog_bar=True, rew_mode = 0, test = True, render = False):

        self.device = torch.device("cpu")
        negative_rewards = [3]

        #Initialize agent
        agent = ContinuousDQN(train_params, env, env.observation_space.shape[0], env.action_space.shape[0])

        #Set seed for repeatability
        random.seed(seed)
        torch.manual_seed(seed)
        np.random.seed(seed)

        #Create training directories
        run_dir = os.path.join(session_pref, train_params.env_name, train_params.agent_name + "-" + str(seed))
        os.makedirs(run_dir, exist_ok=True)

        #Initialize TensorBoard logger and save the training parameters
        writer = SummaryWriter(log_dir=run_dir)
        train_params.render_to_file(run_dir + ".params")

        #Initialize iterator for episodes loop
        interact_iter = range(train_params.num_episodes)
        if show_prog_bar:

            interact_iter = tqdm(interact_iter, colour="green", desc="Episode")

        #Best reward array, used for best model saving and average episode array
        best_reward = -10
        avg_array = np.zeros(5)
        
        #Training loop
        for episode in interact_iter:

            #Reset state and cast to device
            state = env.reset()[0]
            state = torch.tensor(state).float().to(self.device)          

            #Register spawn location ad append to x_pos and y_pos   
            spawn = env.step([0.0 for i in range(0, env.action_space.shape[0])])[4]
            spawn_x, spawn_y = spawn["x_position"], spawn["y_position"]
            x_pos = [spawn_x]
            y_pos = [spawn_y]

            #Initialize cumulative reward array, cumulative network reward, 
            rew_array = np.zeros(5)
            totrew = 0
     

            #Termination condition 
            done = False

            #Interaction loop
            while not done:

                #Get the action from the netowrk and interact with the environment
                action = agent.act(state)          
                next_state, reward, done, _, info = env.step(action)

                #Compute interaction reward and cumulative rewards
                rew_array = ContinuousDQN.compute_rewards(rew_array, info, rew_mode)
                reward = rew_array[0]
                
                #Update cumulative network reward
                totrew += reward

                #Check if the Ant is healthy
                if state[0]<=0.26 or state[0]>1.0:
              
                    done = True            
                
                #Update network and stack average losses
                action = torch.from_numpy(action).squeeze().cpu().float()         
                next_state = torch.tensor(next_state).float().to(self.device)           
                agent.step(state, action, reward, next_state, done)
              

                #Go to next state
                state = next_state

                #Update Ant's path
                x_pos.append(info["x_position"])
                y_pos.append(info["y_position"])

            #===============
            #END OF EPISODE#
            #===============
            
            #Decay epsilon parameter and update target model
            agent.decay(episode)
            agent.update_target_model()

            # Create the array to calculate the average reward per episode anc average episode loss
            # Net rew 1 - Net rew 2 - Net rew 3 - CTRL - ORG - HLT - FWD
            avg_array = np.vstack((avg_array, [np.append(totrew, rew_array[1:])]))      
            
            #Calculate the average reward over 100 epochs
            if episode>=100: 

                avg = [np.mean(avg_array[episode-99:episode,i]) for i in range(0,avg_array.shape[1])]         
                    
            else:

                avg = [np.mean(avg_array[0:episode,i]) for i in range(0,avg_array.shape[1])]

            #Save the model if the average is higher than the best average value
            if totrew > best_reward and episode > 10:

                agent.save_model(os.path.join(run_dir, "best"))
                best_reward = totrew      

            
            #Save the model every save_every_n epochs
            if train_params.save_every_n is not None and episode % train_params.save_every_n == 0:
                agent.save_model(run_dir)            

            #Save a plot of the Ant's path every save_path_every_n epochs
            if episode%(train_params.num_episodes/train_params.save_path_every_n) == 0 and episode != 0:
                fig = plt.figure()
                path = fig.add_subplot(1,1,1)
                path.plot(x_pos, y_pos)
                path.plot(spawn_x, spawn_y, marker = '*')
                path.set_title("Path")

                fig.canvas.draw()
                img = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
                img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
                img = img / 255.0
                img = np.swapaxes(img, 0, 2)
                
                writer.add_figure(f"{train_params.env_name}/Path:", fig, episode)
                writer.add_image(f"{train_params.env_name}/Path:", img, episode)

            #Write metrics with TensorBoard writer
            writer.add_scalar(f"{train_params.env_name}/Network Reward:", totrew, episode)
            writer.add_scalar(f"{train_params.env_name}/Reward + Costs:", (totrew+rew_array[1]), episode)
            writer.add_scalar(f"{train_params.env_name}/Reward Forward:", rew_array[4], episode)
            writer.add_scalar(f"{train_params.env_name}/Cost:", rew_array[1], episode)
            writer.add_scalar(f"{train_params.env_name}/Distance from origin:", rew_array[2], episode)
            writer.add_scalar(f"{train_params.env_name}/Survive:", rew_array[3], episode)

            writer.add_scalar(f"{train_params.env_name}-Avg/ Network Reward:", avg[0], episode)
            writer.add_scalar(f"{train_params.env_name}-Avg/ Reward + Costs:", (avg[0]+avg[1]), episode)
            writer.add_scalar(f"{train_params.env_name}-Avg/ Reward Forward:", avg[4], episode)
            writer.add_scalar(f"{train_params.env_name}-Avg/ Cost:", avg[1], episode)
            writer.add_scalar(f"{train_params.env_name}-Avg/ Distance from origin:", avg[2], episode)
            writer.add_scalar(f"{train_params.env_name}-Avg/ Survive:", avg[3], episode)
            
        #Save last model
        agent.save_model(run_dir)

        #Run test procedure
        if test:

            ContinuousDQN.test(agent, writer, run_dir, rew_mode, train_params, show_prog_bar, render)

        writer.flush()



    @classmethod
    #================
    #   test()      # =========================================================================================================
    #================
    # Handles the testing loop.
    #   
    #   Arguments:  
    #       - env               ->A gym environment. Currently only supports 'Ant'
    #       - agent             ->The LexCDQN class already initialized
    #       - writer            ->The TensorBoard SummaryWriter object used during training
    #       - run_dir           ->A str containing the model's path
    #       - rew_mode          ->The reward mode for compute_rewards; will be used to distribute different rewards
    #       - train_params      ->TrainingParameters dataclass; will be used to retrieve training and network parameters
    #       - show_prog_bar     ->Wether or not to show progress bars in the console
    #       - render            ->Wether or not render the tests
    #==========================================================================================================================
    def test(self, agent, writer, run_dir, rew_mode, train_params: TrainingParameters, show_prog_bar=True, render = False, infinite = False):

        self.device = torch.device("cpu")

        if type(writer) == str:
            
            writer = SummaryWriter(log_dir=writer)

        #Initialize env
        if render:
            test_env = gym.make('Ant-v4', render_mode = "human")
        else:
            test_env = gym.make('Ant-v4')

        #Load best model
        agent.load_model(os.path.join(run_dir, "best"))
        agent.model.eval()   

        #Initialize test loop iterator 
        test_iter = range(train_params.num_test)
        if show_prog_bar:
            test_iter = tqdm(test_iter, colour="blue", desc="Episode")
        
        #Initialize average episode reward array
        avg_array = np.zeros(5)

        #Test loop
        for episode in test_iter:       

            #Initialize state and save spawn location
            state = test_env.reset()[0]
            spawn = test_env.step([0.0 for i in range(0, test_env.action_space.shape[0])])[4]
            spawn_x, spawn_y = spawn["x_position"], spawn["y_position"]
            x_pos = [spawn_x]
            y_pos = [spawn_y]

            state = torch.tensor(state).float().to(self.device) 

            #Initialize cumulative rewards and network rewards array
            rew_array = np.zeros(5)
            totrew = 0

            #Initialize last action
            last_action = np.zeros(test_env.action_space.shape[0])
            t=0

            #Initialize termination condition
            done = False

            #Interaction loop
            while not done:

                #Increase timer
                t+=1

                #Render environment
                if render:
                    test_env.render()

                #Interact with the environment
                action = agent.act(state, test = True)
                next_state, reward, done, _, info = test_env.step(action)
                next_state = torch.tensor(next_state).float().to(self.device)
                rew_array = ContinuousDQN.compute_rewards(rew_array, info, rew_mode)
                reward = rew_array[0]
                
                #Update cumulative rewards, state and path
                totrew += reward                            
                state = next_state
                x_pos.append(info["x_position"])
                y_pos.append(info["y_position"])

                #Termination condition if the agent gets stuck in an infinite loop
                #>In one of the training, the Ant started "vibrating" (performing alternately the same two slightly
                #  different actions) in place and moving slowly but infinitely
                if not infinite:
                    
                    if abs(reward) > 6000:

                        done = True

                    #Check if the Ant is healthy
                    if state[0]<=0.26 or state[0]>1.0:
                        
                        done = True

                    #Over a certain time, check if the agent is moving at all
                    if t > 1000:

                        if (last_action == action).all():
                            done = True
                            writer.add_scalar(f"{train_params.env_name}-[Point]Test/Inactive Terminations", 1, episode)

                        else:

                            last_action = action

                    #Termination condition for time taken
                    if t > 100000:

                        done = True        
                

            avg_array = np.vstack((avg_array, [np.append(totrew, rew_array[1:])]))

            #Save a plot of the Ant's path every save_path_every_n_test epochs
            if episode%(train_params.num_episodes/train_params.save_path_every_n_test) == 0 and episode != 0:
                fig = plt.figure()
                path = fig.add_subplot(1,1,1)
                path.plot(x_pos, y_pos)
                path.plot(spawn_x, spawn_y, marker = '*')
                path.set_title("Path")

                fig.canvas.draw()
                img = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
                img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
                img = img / 255.0
                img = np.swapaxes(img, 0, 2)
                
                writer.add_figure(f"{train_params.env_name}/Path:", fig, episode)
                writer.add_image(f"{train_params.env_name}/Path:", img, episode)

            #Calculate average episode rewards over 100 epochs
            if episode>=100: 

                avg = [np.mean(avg_array[episode-99:episode,i]) for i in range(0,avg_array.shape[1])]         
                    
            else:
                avg = [np.mean(avg_array[0:episode,i]) for i in range(0,avg_array.shape[1])]
            
        
            #Write metrics with TensorBoard writer
            writer.add_scalar(f"{train_params.env_name}-Test/Network Reward:", totrew, episode)
            writer.add_scalar(f"{train_params.env_name}-Test/Reward + Costs:", (totrew+rew_array[1]), episode)
            writer.add_scalar(f"{train_params.env_name}-Test/Reward Forward:", rew_array[4], episode)
            writer.add_scalar(f"{train_params.env_name}-Test/Cost:", rew_array[1], episode)
            writer.add_scalar(f"{train_params.env_name}-Test/Distance from origin:", rew_array[2], episode)
            writer.add_scalar(f"{train_params.env_name}-Test/Survive:", rew_array[3], episode)


            writer.add_scalar(f"{train_params.env_name}-Test-Avg/ Network Reward:", avg[0], episode)
            writer.add_scalar(f"{train_params.env_name}-Test-Avg/ Reward + Costs:", (avg[0]+avg[1]), episode)
            writer.add_scalar(f"{train_params.env_name}-Test-Avg/ Reward Forward:", avg[4], episode)
            writer.add_scalar(f"{train_params.env_name}-Test-Avg/ Cost:", avg[1], episode)
            writer.add_scalar(f"{train_params.env_name}-Test-Avg/ Distance from origin:", avg[2], episode)
            writer.add_scalar(f"{train_params.env_name}-Test-Avg/ Survive:", avg[3], episode)

        test_env.close()


    @classmethod
    #=======================#
    #   compute_array()     # ============================================================
    #=======================#
    # Returns different rewards array based on the mode selected as well as the current
    # cumulative rewards earned
    #=====================================================================================
    def compute_rewards(self, rew_array, information, mode = 0):

        rew_array[4] += information["reward_forward"]
        rew_array[3] += information["reward_survive"]
        rew_array[2] += information["distance_from_origin"]
        rew_array[1] += information["reward_ctrl"]

        #default
        if mode == 0:

            rew_array[0] = 5*information["reward_forward"] +  information["reward_survive"] + information["reward_ctrl"]

        if mode == 1:

            rew_array[0] = information["reward_survive"] + information["distance_from_origin"]

        if mode == 2:

            rew_array[0] = information["distance_from_origin"] + information["reward_ctrl"]

        if mode == 3:

            rew_array[0] = information["distance_from_origin"] +  information["reward_survive"] + information["reward_ctrl"]

        if mode == 4:

            rew_array[0] = information["distance_from_origin"] +  10 * information["reward_survive"]

        if mode == 5:

            rew_array[0] = information["distance_from_origin"]


        return rew_array