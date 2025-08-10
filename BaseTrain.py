import random
import torch
import torch.optim as optim
import torch.nn.functional as F

from RL_moneyflow.lib import ReplayMemory, Transition
import RL_moneyflow.env
from itertools import count
from tqdm import tqdm
import math
import os
from pathlib import Path

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODEL_FOLDER = f'{os.path.dirname(os.path.realpath(__file__))}/models'

class BaseTrain:
    def __init__(self,
                 data_train,
                 env,
                 ticker, 
                 window_size=1,
                 transaction_cost=0.0,
                 BATCH_SIZE=10,
                 GAMMA=0.7,
                 ReplayMemorySize=50,
                 TARGET_UPDATE=5,
                 n_step=10):
        """
        This class is the base class for training across multiple models in the DeepRLAgent directory.
        @param data_loader: The data loader here is to only access the start_data, end_data and split point in order to
            name the result file of the experiment
        @param data_train: of type DataAutoPatternExtractionAgent
        @param data_test: of type DataAutoPatternExtractionAgent
        @param dataset_name: for using in the name of the result file
        @param state_mode: for using in the name of the result file
        @param window_size: for using in the name of the result file
        @param transaction_cost: for using in the name of the result file
        @param BATCH_SIZE: batch size for batch training
        @param GAMMA: in the algorithm
        @param ReplayMemorySize: size of the replay buffer
        @param TARGET_UPDATE: hard update policy network into target network every TARGET_UPDATE iterations
        @param n_step: for using in the name of the result file
        """
        self.data_train = data_train
        self.env = env
        self.BATCH_SIZE = BATCH_SIZE
        self.GAMMA = GAMMA
        self.ReplayMemorySize = ReplayMemorySize
        self.transaction_cost = transaction_cost
        self.window_size = window_size


        self.TARGET_UPDATE = TARGET_UPDATE
        self.n_step = n_step
        self.memory = ReplayMemory(ReplayMemorySize)
        self.EPS_START = 0.9
        self.EPS_END = 0.05
        self.EPS_DECAY = 500

        self.steps_done = 0
        self.model_dir =  f'{MODEL_FOLDER}/{ticker}-model.pkl'
        self.ticker = ticker

    def select_action(self, state):
        sample = random.random()

        eps_threshold = self.EPS_END + (self.EPS_START - self.EPS_END) * \
                        math.exp(-1. * self.steps_done / self.EPS_DECAY)
        self.steps_done += 1

        if sample > eps_threshold:
            with torch.no_grad():
                # t.max(1) will return largest column value of each row.
                # second column on max result is index of where max element was
                # found, so we pick action with the larger expected reward.
                self.policy_net.eval()
                action = self.policy_net(state).max(1)[1].view(1, 1)
                self.policy_net.train()
                return action
        else:
            return torch.tensor([[random.randrange(len(self.env.code_to_action))]], device=device, dtype=torch.long)

    def optimize_model(self):
        if len(self.memory) < self.BATCH_SIZE:
            return
        transitions = self.memory.sample(self.BATCH_SIZE)
        # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
        # detailed explanation). This converts batch-array of Transitions
        # to Transition of batch-arrays.
        batch = Transition(*zip(*transitions))

        # Compute a mask of non-final states and concatenate the batch elements
        # (a final state would've been the one after which simulation ended)
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                                batch.next_state)), device=device, dtype=torch.bool)
        non_final_next_states = torch.cat([s for s in batch.next_state
                                           if s is not None])

        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken. These are the actions which would've been taken
        # for each batch state according to policy_net

        # Using policy-net, we calculate the action-value of the previous actions we have taken before.
        state_action_values = self.policy_net(state_batch).gather(1, action_batch)

        # Compute V(s_{t+1}) for all next states.
        # Expected values of actions for non_final_next_states are computed based
        # on the "older" target_net; selecting their best reward with max(1)[0].
        # This is merged based on the mask, such that we'll have either the expected
        # state value or 0 in case the state was final.
        next_state_values = torch.zeros(self.BATCH_SIZE, device=device)
        next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1)[0].detach()
        # Compute the expected Q values
        expected_state_action_values = (next_state_values * (self.GAMMA ** self.n_step)) + reward_batch

        # Compute Huber loss
        loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()

    def train(self, num_episodes=100):
        print(f'Training {self.ticker}', '...')
        for i_episode in tqdm(range(num_episodes)):
            # Initialize the environment and state
            self.env.reset()
            state = torch.tensor([self.env.get_current_state()], dtype=torch.float, device=device)
            for t in count():
                # Select and perform an action
                action = self.select_action(state)
                done, reward, next_state = self.env.step(action.item())

                reward = torch.tensor([reward], dtype=torch.float, device=device)

                if next_state is not None:
                    next_state = torch.tensor([next_state], dtype=torch.float, device=device)

                # Store the transition in memory
                self.memory.push(state, action, next_state, reward)

                # Move to the next state
                if not done:
                    state = torch.tensor([self.env.get_current_state()], dtype=torch.float, device=device)

                # Perform one step of the optimization (on the target network)
                self.optimize_model()
                if done:
                    break
            # Update the target network, copying all weights and biases in DQN
            if i_episode % self.TARGET_UPDATE == 0:
                self.target_net.load_state_dict(self.policy_net.state_dict())

        # Only save model if not a temporary training (parallel workers)
        if not self.ticker.endswith('_temp'):
            self.save_model(self.policy_net.state_dict())

        print('Complete')

    def save_model(self, model):
        torch.save(model, self.model_dir)

    def code_to_action(self, action_code_list):
        """
        Provided a list of actions at each time-step, it converts the action to its original name like:
        0 -> Buy
        1 -> None
        2 -> Sell
        @param action_list: ...
        @return: ...
        """
        code_dict = {0: 'buy', 1: 'None', 2: 'sell'}
        i = 1
        action_list = []
        action_list.append('None')
        for a in action_code_list:
            action_list.append(code_dict[a])
            i += 1
        return action_list

    def test(self, data, initial_investment=1000):
        """
        :@param file_name: name of the .pkl file to load the model
        :@param test_type: test results on train data or test data
        :@return returns an Evaluation object to have access to different evaluation metrics.
        """
        try:
            self.test_net.load_state_dict(torch.load(self.model_dir, weights_only=True))
        except RuntimeError as e:
            print(f"Architecture mismatch loading {self.model_dir}: {e}")
            print("Skipping model loading - using randomly initialized weights")
        self.test_net.to(device)

        action_list = []
        #data.__iter__()

        i = 0
        while (i < len(data)):
            batch = torch.from_numpy( data.iloc[ i : i + self.BATCH_SIZE].values).to(device, dtype=torch.float)
            #batch = [torch.tensor([s], dtype=torch.float, device=device) for s in
            #         data.iloc[ i : i + self.BATCH_SIZE].values]
            try: 
                action_batch = self.test_net(batch).max(1)[1]
                action_list += list(action_batch.cpu().numpy())
            except ValueError:
                action_list += [1]                
            i += self.BATCH_SIZE

        action_list =  self.code_to_action(action_list)
        return action_list

