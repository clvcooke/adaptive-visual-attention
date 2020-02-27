import torch.nn as nn
from modules import GlimpseNetwork, DecisionNetwork, ActionNetwork, LocationNetwork, BaselineNetwork
from torch.nn import LSTMCell


class AdaptiveAttention(nn.Module):
    """
    A Recurrent Model of Visual Attention (RAM) [1].

    RAM is a recurrent neural network that processes
    inputs sequentially, attending to different locations
    within the image one at a time, and incrementally
    combining information from these fixations to build
    up a dynamic internal representation of the image.

    References
    ----------
    - Minh et. al., https://arxiv.org/abs/1406.6247
    """

    def __init__(self, h_g, h_l, std, hidden_size, num_classes, patch_amount, patch_size, scale_factor):
        super(AdaptiveAttention, self).__init__()
        self.std = std
        self.sensor = GlimpseNetwork(hidden_size, patch_amount=patch_amount, patch_size=patch_size,
                                     scale_factor=scale_factor)
        self.rnn = LSTMCell(hidden_size, hidden_size)
        self.decider = DecisionNetwork(hidden_size, 2)
        self.locator = LocationNetwork(hidden_size, 2, std)
        self.classifier = ActionNetwork(hidden_size, num_classes)
        self.baseliner = BaselineNetwork(hidden_size, 1)

    def forward(self, x, loc_t_prev, h_t_prev, last=False):
        # sample the image
        g_t = self.sensor(x, loc_t_prev)
        # calculate the next hidden state
        h_t, c_t = self.rnn(g_t, h_t_prev)
        loc_t, log_probs_loc = self.locator(h_t)
        d, log_probs_d = self.decider(h_t)
        log_probas = self.classifier(h_t)
        baseline = self.baseliner(h_t)
        return (h_t, c_t), loc_t, log_probs_loc, log_probas, d, log_probs_d, baseline
