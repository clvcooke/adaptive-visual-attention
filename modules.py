import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical, Normal


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

    # noinspection PyTypeChecker
    def __init__(self, hidden_size, patch_amount, patch_size, scale_factor):
        super(GlimpseNetwork, self).__init__()
        self.retina = Retina(patch_amount=patch_amount, patch_size=patch_size, scale_factor=scale_factor)
        self.patch_data_size = patch_size*patch_size*patch_amount
        self.loc_size = 2
        # TODO: pass in h_g and h_l
        print("HACK")
        self.model_what = SimpleMLP(self.patch_data_size, hidden_size, hidden_size=hidden_size//2, hidden_layers=1, final_activation=None)
        self.model_where = SimpleMLP(self.loc_size, hidden_size, hidden_size=hidden_size//2, hidden_layers=1, final_activation=None)

    def forward(self, x, loc_t):
        phi = self.retina.foveate(x, loc_t)
        res_phi = self.model_what(phi)
        res_k = self.model_where(loc_t)
        res = F.relu(res_k + res_phi)
        return res


class SimpleMLP(nn.Module):

    def __init__(self, input_size, output_size, hidden_size=64, hidden_layers=1, final_activation=nn.ReLU):
        super(SimpleMLP, self).__init__()
        layers = []
        curr_size = input_size
        for i in range(hidden_layers):
            layers.append(nn.Linear(curr_size, hidden_size))
            layers.append(nn.ReLU())
            curr_size = hidden_size
        layers.append(nn.Linear(curr_size, output_size))
        if final_activation is not None:
            layers.append(final_activation())
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class ActionNetwork(nn.Module):

    def __init__(self, input_size, output_size):
        super(ActionNetwork, self).__init__()
        self.fc = nn.Linear(input_size, output_size)

    def forward(self, h_t):
        a_t = F.log_softmax((self.fc(h_t)), dim=1)
        return a_t


# noinspection PyTypeChecker
class LocationNetwork(nn.Module):

    def __init__(self, input_size, output_size, std, normalized=True):
        super(LocationNetwork, self).__init__()
        self.normalized = normalized
        self.std = std
        self.model = SimpleMLP(input_size, output_size, hidden_size=input_size, hidden_layers=0,
                               final_activation=nn.Tanh)

    def forward(self, x):
        mu = self.model(x.detach())
        noise = torch.zeros_like(mu)
        noise.data.normal_(std=self.std)
        loc = mu + noise
        log_pi = Normal(mu, self.std).log_prob(loc)
        log_pi = torch.sum(log_pi, dim=1)
        loc = torch.tanh(loc)
        return loc, log_pi


# noinspection PyTypeChecker
class DecisionNetwork(nn.Module):
    def __init__(self, input_size, output_size):
        super(DecisionNetwork, self).__init__()
        self.model = SimpleMLP(input_size, output_size, hidden_layers=1, hidden_size=input_size, final_activation=None)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, h_t):
        probs = self.softmax(self.model(h_t))
        sample = torch.distributions.Categorical(probs=probs).sample()
        log_probs = torch.log(probs)
        return sample, log_probs


class BaselineNetwork(nn.Module):
    def __init__(self, input_size, output_size):
        super(BaselineNetwork, self).__init__()
        self.model = SimpleMLP(input_size, output_size, hidden_layers=0, hidden_size=input_size, final_activation=nn.ReLU)

    def forward(self, x):
        return self.model(x.detach())
