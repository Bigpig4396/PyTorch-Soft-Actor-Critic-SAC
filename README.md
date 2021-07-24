# PyTorch-Soft-Actor-Critic-SAC


https://arxiv.org/abs/1910.07207v1


Soft actor critic algorithm for discrete action.


If it is instable, you can make the q_loss and p_loss smaller (like divide by 100 or more) and cancel the learning of alpha parameter in train().
