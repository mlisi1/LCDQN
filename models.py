import torch.nn as nn
import torch.nn.functional as F

#===============#
#   LexDNN      # ==========================================================
#===============#
# A lexicographic DNN; has an input size of state_size and and output size
# of action_size * sample_size * reward size.
# Has 3 layers, the first two of which perform relu activation
# The nohid parameter can remove the intermediate layer
#===========================================================================
class LexDNN(nn.Module):

    def __init__(self, state_size, action_size, sample_size, hidden_size, reward_size, nohid, bias):

        super(LexDNN, self).__init__()

        self.state_size = state_size
        self.hidden_size = hidden_size
        self.action_size = action_size
        self.sample_size = sample_size
        self.reward_size = reward_size
        self.nohid = nohid
        self.bias = bias
      
        self.fc1 = nn.Linear(self.state_size, self.hidden_size, bias = bias)
        self.fc2 = nn.Linear(self.hidden_size, self.hidden_size, bias = bias)
        self.fc3 = nn.Linear(self.hidden_size, self.action_size * self.sample_size * self.reward_size, bias = bias)        

    def forward(self, x):

        x = F.relu(self.fc1(x))
        if not self.nohid:
            x = F.relu(self.fc2(x))
        x = self.fc3(x)        
       
        x = x.view(-1, self.action_size, self.sample_size, self.reward_size)       
        
        return x


#============#
#   DNN      # ==========================================================
#============#
# A DNN; has an input size of state_size and and output size
# of action_size * sample_size.
# Has 3 layers, the first two of which perform relu activation
# The nohid parameter can remove the intermediate layer
#===========================================================================
class DNN(nn.Module):

    def __init__(self, state_size, action_size, sample_size, hidden_size, nohid, bias):

        super(DNN, self).__init__()

        self.state_size = state_size
        self.hidden_size = hidden_size
        self.action_size = action_size
        self.sample_size = sample_size
        self.nohid = nohid
        self.bias = bias
      
        self.fc1 = nn.Linear(self.state_size, self.hidden_size, bias = bias)
        self.fc2 = nn.Linear(self.hidden_size, self.hidden_size, bias = bias)
        self.fc3 = nn.Linear(self.hidden_size, self.action_size * self.sample_size, bias = bias)
        

    def forward(self, x):
        x = F.relu(self.fc1(x))
        if not self.nohid:
            x = F.relu(self.fc2(x))
        x = self.fc3(x)        
       
        x = x.view(-1, self.action_size, self.sample_size)        
        
        return x


