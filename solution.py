import torch
import torch.optim as optim
from torch.distributions import Normal
import torch.nn as nn
import numpy as np
from gym.wrappers.monitoring.video_recorder import VideoRecorder
import warnings
from typing import Union
from utils import ReplayBuffer, get_env, run_episode

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# define seeds for reproducibility
import random
random.seed(0)
np.random.seed(0)
torch.manual_seed(0)

class NeuralNetwork(nn.Module):
    '''
    This class implements a neural network with a variable number of hidden layers and hidden units.
    You may use this function to parametrize your policy and critic networks.
    '''
    def __init__(self, input_dim: int, output_dim: int, hidden_size: int, 
                                hidden_layers: int, activation: str):
        super(NeuralNetwork, self).__init__()

        # Defining a neural network 
        # with a variable number of hidden layers and hidden units.
        act_fn = getattr(nn, activation)() 
        layers = [nn.Linear(input_dim,hidden_size),act_fn]
        for _ in range(hidden_layers):
            layers.extend([nn.Linear(hidden_size,hidden_size),act_fn])
        layers.append(nn.Linear(hidden_size,output_dim))

        self.model = nn.Sequential(*layers)         

    def forward(self, s: torch.Tensor) -> torch.Tensor:
        return self.model(s)
    
class Actor:
    def __init__(self,hidden_size: int, hidden_layers: int, actor_lr: float,
                state_dim: int = 3, action_dim: int = 1, device: torch.device = torch.device('cpu')):
        super(Actor, self).__init__()

        self.hidden_size = hidden_size
        self.hidden_layers = hidden_layers
        self.actor_lr = actor_lr
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.device = device
        self.LOG_STD_MIN = -20
        self.LOG_STD_MAX = 2
        self.setup_actor()

    def setup_actor(self):
        '''
        This function sets up the actor network in the Actor class.
        '''
        self.network = NeuralNetwork(self.state_dim, 2, self.hidden_size, self.hidden_layers, "ReLU")
        self.optimizer = optim.Adam(self.network.parameters(), lr=self.actor_lr)
        # to avoid numerical errors when taking the log
        self.EPS = 1e-6

    def clamp_log_std(self, log_std: torch.Tensor) -> torch.Tensor:
        '''
        :param log_std: torch.Tensor, log_std of the policy.
        Returns:
        :param log_std: torch.Tensor, log_std of the policy clamped between LOG_STD_MIN and LOG_STD_MAX.
        '''
        return torch.clamp(log_std, self.LOG_STD_MIN, self.LOG_STD_MAX)

    def get_action_and_log_prob(self, state: torch.Tensor, 
                                deterministic: bool) -> (torch.Tensor, torch.Tensor):
        '''
        :param state: torch.Tensor, state of the agent
        :param deterministic: boolean, if true return a deterministic action 
                                otherwise sample from the policy distribution.
        Returns:
        :param action: torch.Tensor, action the policy returns for the state.
        :param log_prob: log_probability of the the action.
        '''
        assert state.shape == (3,) or state.shape[1] == self.state_dim, 'State passed to this method has a wrong shape'
        action , log_prob = torch.zeros(state.shape[0]), torch.ones(state.shape[0])
        # get out output from model  
        out = self.network(state)
        # retrieve policy distribution parameters from the output
        mu = out[:,0].reshape((out.shape[0],1))
        # assume the model predicts log_std -> ensures numerical stability
        log_std = out[:,1].reshape((out.shape[0],1))
        std = torch.exp(self.clamp_log_std(log_std))
        # use reparameterization trick - implemented via rsample
        normal = torch.distributions.Normal(mu,std)
        raw_action = normal.rsample()
        # squash action (i.e. ensure it is bounded between -1 and 1)
        action = torch.tanh(raw_action)
        # correct for squashing (using the explanation from Appendix C in the SAC paper)
        # D for the action space is 1, so summing over the log terms in this term is redundant
        # however this is still implemented to abide by the formula in the paper
        # and for potential extensions to larger action spaces
        log_prob = normal.log_prob(raw_action) - torch.log(1 - action**2 + self.EPS).sum(1, keepdim=True)
        # experiment with deterministic policy
        if deterministic:
            mu = torch.tanh(mu)
            return mu, log_prob
        assert (action.shape == (state.shape[0], self.action_dim) and \
            log_prob.shape == (state.shape[0], self.action_dim)), 'Incorrect shape for action or log_prob.'
        return action, log_prob


class Critic:
    def __init__(self, hidden_size: int, 
                 hidden_layers: int, critic_lr: int, state_dim: int = 3, 
                    action_dim: int = 1,device: torch.device = torch.device('cpu')):
        super(Critic, self).__init__()
        self.hidden_size = hidden_size
        self.hidden_layers = hidden_layers
        self.critic_lr = critic_lr
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.device = device
        self.setup_critic()

    def setup_critic(self):
        import copy
        # initialize the first Q-function
        self.q1 = NeuralNetwork(self.state_dim+self.action_dim, 1, self.hidden_size, self.hidden_layers, "ReLU")
        # initialize q1_bar=q1
        self.q1_bar = copy.deepcopy(self.q1)
        self.q1_opt = optim.Adam(self.q1.parameters(), lr=self.critic_lr)

        # initialize the second Q-function
        self.q2 = NeuralNetwork(self.state_dim+self.action_dim, 1, self.hidden_size, self.hidden_layers, "ReLU")
        # initialize q2_bar=q2
        self.q2_bar = copy.deepcopy(self.q2)
        self.q2_opt = optim.Adam(self.q2.parameters(), lr=self.critic_lr)

class TrainableParameter:
    '''
    This class could be used to define a trainable parameter in your method. You could find it 
    useful if you try to implement the entropy temerature parameter for SAC algorithm.
    '''
    def __init__(self, init_param: float, lr_param: float, 
                 train_param: bool, device: torch.device = torch.device('cpu')):
        
        self.log_param = torch.tensor(np.log(init_param), requires_grad=train_param, device=device)
        self.optimizer = optim.Adam([self.log_param], lr=lr_param)

    def get_param(self) -> torch.Tensor:
        return torch.exp(self.log_param)

    def get_log_param(self) -> torch.Tensor:
        return self.log_param


class Agent:
    def __init__(self):
        # Environment variables. You don't need to change this.
        self.state_dim = 3  # [cos(theta), sin(theta), theta_dot]
        self.action_dim = 1  # [torque] in[-1,1]
        self.batch_size = 200
        self.min_buffer_size = 1000
        self.max_buffer_size = 100000
        # If your PC possesses a GPU, you should be able to use it for training, 
        # as self.device should be 'cuda' in that case.
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("Using device: {}".format(self.device))
        self.memory = ReplayBuffer(self.min_buffer_size, self.max_buffer_size, self.device)
        
        self.setup_agent()

    def setup_agent(self):
        # Setup off-policy agent with policy and critic classes. 
                
        # define parameters for the actor=policy
        policy_hidden_size = 256
        policy_hidden_layers = 2
        policy_lr = 3*1e-4
        self.policy = Actor(policy_hidden_size, policy_hidden_layers, policy_lr)
        
        # define parameters for the critics
        critic_hidden_size = 256
        critic_hidden_layers = 2
        critic_lr = 3*1e-4
        self.critic = Critic(critic_hidden_size, critic_hidden_layers, critic_lr)
        
        # define additional hyperparameters from the SAC algorithm
        self.gamma = 0.99
        self.tau = 0.005
        self.alpha_init = 0.2
        self.alpha_lr = 3*1e-4
        self.alpha = TrainableParameter(self.alpha_init, self.alpha_lr, True)

    def get_action(self, s: np.ndarray, train: bool) -> np.ndarray:
        """
        :param s: np.ndarray, state of the pendulum. shape (3, )
        :param train: boolean to indicate if you are in eval or train mode. 
                    You can find it useful if you want to sample from deterministic policy.
        :return: np.ndarray,, action to apply on the environment, shape (1,)
        """
        # convert state to torch.tensor and reshape
        state = torch.from_numpy(s).unsqueeze(0)

        # sample action from the policy
        action, _ = self.policy.get_action_and_log_prob(state, False)
        # reshape and convert to numpy array
        action = action.detach().squeeze(-1).numpy()
        
        assert action.shape == (1,), 'Incorrect action shape.'
        assert isinstance(action, np.ndarray ), 'Action dtype must be np.ndarray' 
        return action

    @staticmethod
    def run_gradient_update_step(object: Union[Actor, Critic], loss: torch.Tensor):
        '''
        This function takes in a object containing trainable parameters and an optimizer, 
        and using a given loss, runs one step of gradient update. If you set up trainable parameters 
        and optimizer inside the object, you could find this function useful while training.
        :param object: object containing trainable parameters and an optimizer
        '''
        object.optimizer.zero_grad()
        loss.mean().backward()
        object.optimizer.step()

    def critic_target_update(self, base_net: NeuralNetwork, target_net: NeuralNetwork, 
                             tau: float, soft_update: bool):
        '''
        This method updates the target network parameters using the source network parameters.
        If soft_update is True, then perform a soft update, otherwise a hard update (copy).
        :param base_net: source network
        :param target_net: target network
        :param tau: soft update parameter
        :param soft_update: boolean to indicate whether to perform a soft update or not
        '''
        for param_target, param in zip(target_net.parameters(), base_net.parameters()):
            if soft_update:
                param_target.data.copy_(param_target.data * (1.0 - tau) + param.data * tau)
            else:
                param_target.data.copy_(param.data)

    def train_agent(self):
        '''
        This function represents one training iteration for the agent. It samples a batch 
        from the replay buffer,and then updates the policy and critic networks 
        using the sampled batch.
        '''

        # Batch sampling
        batch = self.memory.sample(self.batch_size)
        s_batch, a_batch, r_batch, s_prime_batch = batch
        
        # Critic(s) update here.
        # get alpha 
        alpha = self.alpha.get_param()

        # update Q-function parameters
        # define J_Q(theta_1) and J_Q(theta_2) using eq. 5
        # note we use the redefinition
        # E_{s_t+1~p}[V_{theta_bar}]=Q_{theta_bar}(s_t+1,a_t+1)-alpha*log(pi_phi(a_t+1|s_t+1))
        # as in the paper
        # compute Q_theta1 and Q_theta2
        q1_val = self.critic.q1(torch.hstack((s_batch,a_batch)))
        q2_val = self.critic.q2(torch.hstack((s_batch,a_batch)))

        # sample a_t+1 using the s_prime_batch
        a_prime_batch, log_prob_prime = self.policy.get_action_and_log_prob(s_prime_batch, False)
        # compute Q_theta_bar(s_t+1,a_t+1) for both Q-functions
        q1_val_prime = self.critic.q1_bar(torch.hstack((s_prime_batch,a_prime_batch)))
        q2_val_prime = self.critic.q2_bar(torch.hstack((s_prime_batch,a_prime_batch)))
        # as mentioned in the paper we use the minimum of the Q-functions for the Q_theta_bar term
        min_q_val_prime = torch.min(q1_val_prime,q2_val_prime)
        q_hat = r_batch + self.gamma * (min_q_val_prime - alpha * log_prob_prime)

        # compute the Q losses and update weights via backpropagation
        # important q_hat does not depend on theta_1
        # and should therefore be detached from the computation graph
        # use eq. 5 from the Soft Actor-Critic Algorithms and Applications paper
        q1_loss = 0.5 * torch.mean((q1_val - q_hat.detach())**2)
        self.critic.q1_opt.zero_grad()
        q1_loss.backward()
        self.critic.q1_opt.step()

        q2_loss = 0.5 * torch.mean((q2_val - q_hat.detach())**2)
        self.critic.q2_opt.zero_grad()
        q2_loss.backward()
        self.critic.q2_opt.step()

        # Policy update
        # compute log pi(a|s)
        # sample actions a_t via reparameterization (done implicitly in the get_action_and_log_prob function)
        a_batch_new, log_prob = self.policy.get_action_and_log_prob(s_batch, False)
        # compute Q(s_t,f_phi(epsilon_t;s_t))
        # i.e. a_t=a_batch_new=f_phi(epsilon_t;s_t)
        q1_val = self.critic.q1(torch.hstack((s_batch,a_batch_new)))
        q2_val = self.critic.q2(torch.hstack((s_batch,a_batch_new)))
        # as mintioned in the paper used the minimum value to compute the policy loss
        min_q_val = torch.min(q1_val, q2_val)
        
        # compute policy loss via eq. 9 from the Soft Actor-Critic Algorithms and Applications paper
        policy_loss = torch.mean(alpha * log_prob - min_q_val)
        self.policy.optimizer.zero_grad()
        policy_loss.backward()
        self.policy.optimizer.step()
        
        # update alpha
        # use H=-dim(A) for the entropy target
        # same as Soft Actor-Critic Algorithms and Applications paper
        # compute alpha loss using eq. 18
        alpha_loss = -torch.mean(self.alpha.get_log_param() * (log_prob + (-self.action_dim)).detach())
        self.alpha.optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha.optimizer.step()
        
        # update target network weights, i.e. q1_bar and q2_bar
        self.critic_target_update(self.critic.q1, self.critic.q1_bar, self.tau, True)
        self.critic_target_update(self.critic.q2, self.critic.q2_bar, self.tau, True)



# This main function is provided here to enable some basic testing. 
# ANY changes here WON'T take any effect while grading.
if __name__ == '__main__':
    TRAIN_EPISODES = 50 # 50
    TEST_EPISODES = 300 # 300

    # You may set the save_video param to output the video of one of the evalution episodes, or 
    # you can disable console printing during training and testing by setting verbose to False.
    save_video = True
    verbose = True

    agent = Agent()
    env = get_env(g=10.0, train=True)

    for EP in range(TRAIN_EPISODES):
        run_episode(env, agent, None, verbose, train=True)

    if verbose:
        print('\n')

    test_returns = []
    env = get_env(g=10.0, train=False)

    if save_video:
        video_rec = VideoRecorder(env, f'pendulum_episode_train+{TRAIN_EPISODES}.mp4')
    
    for EP in range(TEST_EPISODES):
        rec = video_rec if (save_video and EP == TEST_EPISODES - 1) else None
        with torch.no_grad():
            episode_return = run_episode(env, agent, rec, verbose, train=False)
        test_returns.append(episode_return)

    avg_test_return = np.mean(np.array(test_returns))

    print("\n AVG_TEST_RETURN:{:.1f} \n".format(avg_test_return))

    if save_video:
        video_rec.close()
