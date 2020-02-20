import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical


class Retina(object):
    """
    A retina that extracts a foveated glimpse `phi`
    around location `l` from an image `x`. It encodes
    the region around `l` at a high-resolution but uses
    a progressively lower resolution for pixels further
    from `l`, resulting in a compressed representation
    of the original image `x`.

    Args
    ----
    - x: a 4D Tensor of shape (B, H, W, C). The minibatch
      of images.
    - l: a 2D Tensor of shape (B, 2). Contains normalized
      coordinates in the range [-1, 1].
    - patch_size: size of the first square patch.
    - patch_amount: number of patches to extract in the glimpse.
    - scale_factor: scaling factor that controls the size of
      successive patches.

    Returns
    -------
    - phi: a 5D tensor of shape (B, k, g, g, C). The
      foveated glimpse of the image.
    """

    def __init__(self, patch_size, patch_amount, scale_factor):
        self.patch_size = patch_size
        self.patch_amount = patch_amount
        self.scale_factor = scale_factor

    def foveate(self, x, loc):
        """
        Extract `patch_amount` square patches of size `patch_size`, centered
        at location `loc`. The initial patch is a square of
        size `patch_size`, and each subsequent patch is a square
        whose side is `scale_factor` times the size of the previous
        patch.

        The `patch_amount` patches are finally resized to (patch_size, patch_size) and
        concatenated into a tensor of shape (B, patch_amount, patch_size, patch_size, C).
        """
        patches = []
        size = self.patch_size
        # extract k patches of increasing size
        for i in range(self.patch_amount):
            patches.append(self.extract_patch(x, loc, size))
            size = int(self.scale_factor * size)

        # resize the patches to squares of size patch_size, first patch is skipped
        for i in range(1, len(patches)):
            k = patches[i].shape[-1] // self.patch_size
            patches[i] = F.avg_pool2d(patches[i], k)

        # concatenate into a single tensor and flatten
        phi = torch.cat(patches, 1)
        phi = phi.view(phi.shape[0], -1)

        return phi

    def extract_patch(self, x, l, size):
        """
        Extract a single patch for each image in the
        minibatch `x`.

        Args
        ----
        - x: a 4D Tensor of shape (B, H, W, C). The minibatch
          of images.
        - l: a 2D Tensor of shape (B, 2).
        - size: a scalar defining the size of the extracted patch.

        Returns
        -------
        - patch: a 4D Tensor of shape (B, size, size, C)
        """
        B, C, H, W = x.shape

        # denormalize coords of patch center
        coords = self.denormalize(H, l)

        # compute top left corner of patch
        patch_x = coords[:, 0] - (size // 2)
        patch_y = coords[:, 1] - (size // 2)

        # loop through mini-batch and extract
        patch = []
        for i in range(B):
            im = x[i].unsqueeze(dim=0)
            T = im.shape[-1]

            # compute slice indices
            from_x, to_x = patch_x[i], patch_x[i] + size
            from_y, to_y = patch_y[i], patch_y[i] + size

            # cast to ints
            try:
                from_x, to_x = from_x.data[0], to_x.data[0]
                from_y, to_y = from_y.data[0], to_y.data[0]
            except IndexError:
                from_x, to_x = from_x.data.item(), to_x.data.item()
                from_y, to_y = from_y.data.item(), to_y.data.item()

            # pad tensor in case exceeds
            if self.exceeds(from_x, to_x, from_y, to_y, T):
                pad_dims = (
                    size // 2 + 1, size // 2 + 1,
                    size // 2 + 1, size // 2 + 1,
                    0, 0,
                    0, 0,
                )
                im = F.pad(im, pad_dims, "constant", 0)

                # add correction factor
                from_x += (size // 2 + 1)
                to_x += (size // 2 + 1)
                from_y += (size // 2 + 1)
                to_y += (size // 2 + 1)

            # and finally extract
            patch.append(im[:, :, from_y:to_y, from_x:to_x])

        # concatenate into a single tensor
        patch = torch.cat(patch)

        return patch

    @staticmethod
    def denormalize(T, coords):
        """
        Convert coordinates in the range [-1, 1] to
        coordinates in the range [0, T] model_where `T` is
        the size of the image.
        """
        return (0.5 * ((coords + 1.0) * T)).long()

    @staticmethod
    def exceeds(from_x, to_x, from_y, to_y, T):
        """
        Check whether the extracted patch will exceed
        the boundaries of the image of size `T`.
        """
        if (
                (from_x < 0) or (from_y < 0) or (to_x > T) or (to_y > T)
        ):
            return True
        return False


class GlimpseNetwork(nn.Module):
    """
    A network that combines the "what" and the "model_where"
    into a glimpse feature vector `g_t`.

    - "what": glimpse extracted from the retina.
    - "model_where": location tuple model_where glimpse was extracted.

    Concretely, feeds the output of the retina `phi` to
    a fc layer and the glimpse location vector `l_t_prev`
    to a fc layer. Finally, these outputs are fed each
    through a fc layer and their sum is rectified.

    In other words:

        `g_t = relu( fc( fc(l) ) + fc( fc(phi) ) )`

    Args
    ----
    - h_g: hidden layer size of the fc layer for `phi`.
    - h_l: hidden layer size of the fc layer for `l`.
    - g: size of the square patches in the glimpses extracted
      by the retina.
    - k: number of patches to extract per glimpse.
    - s: scaling factor that controls the size of successive patches.
    - c: number of num_channels in each image.
    - x: a 4D Tensor of shape (B, H, W, C). The minibatch
      of images.
    - l_t_prev: a 2D tensor of shape (B, 2). Contains the glimpse
      coordinates [x, y] for the previous timestep `t-1`.

    Returns
    -------
    - g_t: a 2D tensor of shape (B, hidden_size). The glimpse
      representation returned by the glimpse network for the
      current timestep `t`.
    """

    def __init__(self, h_g, h_l, learned_start, patch_amount, patch_size):
        super(GlimpseNetwork, self).__init__()
        self.retina = Retina(patch_amount=patch_amount, patch_size=patch_size, scale_factor=scale_factor)
        self.learned_start = learned_start
        self.model_what = WhatModel()
        self.model_where = WhereModel()

    def forward(self, x, k_t_prev):
        # generate k-sample phi from image x
        if k_t_prev is None and self.learned_start:
            phi = self.conv_layer(x.permute(0, 3, 1, 2))
            k_t_prev = self.conv_layer.weight.view(-1, self.channels).repeat(
                [phi.shape[0], 1])
        else:
            if k_t_prev is None:
                k_t_prev = torch.rand([x.shape[0], self.channels]) * 2 - 1
            phi = self.retina.illuminate(x, k_t_prev).view((-1, 1, 28, 28))

        res_phi = self.model_what(phi.view((phi.shape[0], -1)))
        res_k = self.model_where(k_t_prev)
        res = F.relu(res_k + res_phi)
        return res


class WhatModel(nn.Module):

    def __init__(self):
        super(WhatModel, self).__init__()
        self.model= torch.nn.Sequential(
            torch.nn.Linear(784, 128),
            torch.nn.BatchNorm1d(128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 256),
            torch.nn.BatchNorm1d(256)
        )

    def forward(self, x):
        return self.model(x)


class WhereModel(nn.Module):

    def __init__(self):
        super(WhereModel, self).__init__()
        self.model = torch.nn.Sequential(
            torch.nn.Linear(784, 128),
            torch.nn.BatchNorm1d(128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 256),
            torch.nn.BatchNorm1d(256)
        )

    def forward(self, x):
        return self.model(x)


class action_network(nn.Module):
    """
    Uses the internal state `h_t` of the core network to
    produce the final output classification.

    Concretely, feeds the hidden state `h_t` through a fc
    layer followed by a softmax to create a vector of
    output probabilities over the possible num_classes.

    Hence, the environment action `a_t` is drawn from a
    distribution conditioned on an affine transformation
    of the hidden state vector `h_t`, or in other words,
    the action network is simply a linear softmax classifier.

    Args
    ----
    - input_size: input size of the fc layer.
    - output_size: output size of the fc layer.
    - h_t: the hidden state vector of the core network for
      the current time step `t`.

    Returns
    -------
    - a_t: output probability vector over the num_classes.
    """

    def __init__(self, input_size, output_size):
        super(action_network, self).__init__()
        self.fc = nn.Linear(input_size, output_size)

    def forward(self, h_t):
        a_t = F.log_softmax((self.fc(h_t)), dim=1)
        return a_t


class decision_network(nn.Module):
    def __init__(self, input_size, output_size):
        super(decision_network, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, input_size),
            nn.BatchNorm1d(input_size),
            nn.ReLU(),
            nn.Linear(input_size, output_size),
            nn.Softmax(dim=1)
        )

    def forward(self, h_t):
        probs = self.model(h_t)
        try:
            sample = torch.distributions.Categorical(probs=probs).sample()
        except:
            probs = torch.softmax(probs + 1, dim=1)
            sample = torch.distributions.Categorical(probs=probs).sample()
        probs = torch.log(probs)
        return sample, probs


class illumination_network(nn.Module):

    def __init__(self, input_size, output_size, std):
        super(illumination_network, self).__init__()
        self.std = std
        self.model = nn.Sequential(
            nn.Linear(input_size, input_size),
            nn.BatchNorm1d(input_size),
            nn.ReLU(),
            nn.Linear(input_size, output_size),
            nn.Tanh()
        )
        self.std = std

    def forward(self, h_t, valid=False):
        # compute mean
        mu = self.model(h_t)
        noise = torch.randn_like(mu) * self.std
        k_t = torch.tanh(mu + noise)
        return k_t
