import torch
import wandb

from config import get_config
from data_loader import get_test_loader, get_train_valid_loader
from trainer import Trainer

wandb.init("AVA")


def main(config):
    # ensure reproducibility
    torch.manual_seed(config.random_seed)
    if config.use_gpu:
        torch.cuda.manual_seed(config.random_seed)

    # instantiate data loaders
    if config.is_train:
        data_loader = get_train_valid_loader(task='MNIST',
                                             batch_size=config.batch_size,
                                             random_seed=config.random_seed,
                                             valid_size=config.valid_size)
    else:
        data_loader = get_test_loader(task='MNIST', batch_size=config.batch_size)
    wandb.config.update(config)
    # instantiate trainer
    trainer = Trainer(config, data_loader)
    # either train
    trainer.train()


if __name__ == '__main__':
    config, unparsed = get_config()
    main(config)
