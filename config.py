import argparse

arg_lists = []
parser = argparse.ArgumentParser(description='AVA')


def str2bool(v):
    return v.lower() in ('true', '1')


def add_argument_group(name):
    arg = parser.add_argument_group(name)
    arg_lists.append(arg)
    return arg


# glimpse network params
glimpse_arg = add_argument_group('Glimpse Network Params')
glimpse_arg.add_argument('--patch_size', type=int, default=8,
                         help='size of extracted patch at highest res')
glimpse_arg.add_argument('--glimpse_scale', type=int, default=2,
                         help='scale of successive patches')
glimpse_arg.add_argument('--num_patches', type=int, default=1,
                         help='# of downscaled patches per glimpse')
glimpse_arg.add_argument('--loc_hidden', type=int, default=128,
                         help='hidden size of loc fc')
glimpse_arg.add_argument('--glimpse_hidden', type=int, default=0,
                         help='hidden size of glimpse fc')


# core network params
core_arg = add_argument_group('Core Network Params')
core_arg.add_argument('--num_glimpses', type=int, default=6,
                      help='# of glimpses, i.e. BPTT iterations')
core_arg.add_argument('--hidden_size', type=int, default=256,
                      help='hidden size of rnn')


# reinforce params
reinforce_arg = add_argument_group('Reinforce Params')
reinforce_arg.add_argument('--std', type=float, default=0.17,
                           help='gaussian policy standard deviation')
reinforce_arg.add_argument('--M', type=float, default=10,
                           help='Monte Carlo sampling for valid and test sets')


# data params
data_arg = add_argument_group('Data Params')
data_arg.add_argument('--valid_size', type=float, default=0.2,
                      help='Proportion of training set used for validation')
data_arg.add_argument('--batch_size', type=int, default=8,
                      help='# of images in each batch of data')
data_arg.add_argument('--shuffle', type=str2bool, default=True,
                      help='Whether to shuffle the train and valid indices')
data_arg.add_argument('--show_sample', type=str2bool, default=False,
                      help='Whether to visualize a sample grid of the data')


# training params
train_arg = add_argument_group('Training Params')
train_arg.add_argument('--is_train', type=str2bool, default=True,
                       help='Whether to train or test the model')
train_arg.add_argument('--momentum', type=float, default=0.5,
                       help='Nesterov momentum value')
train_arg.add_argument('--epochs', type=int, default=500,
                       help='# of epochs to train for')
train_arg.add_argument('--init_lr', type=float, default=3e-4,
                       help='Initial learning rate value')
train_arg.add_argument('--lr_patience', type=int, default=10,
                       help='Number of epochs to wait before reducing lr')
train_arg.add_argument('--train_patience', type=int, default=1000,
                       help='Number of epochs to wait before stopping train')
train_arg.add_argument('--loss_balance', type=float, default=1.0,
                       help='loss balance')
train_arg.add_argument('--learned_start', type=bool, default=True,
                       help='If the first pattern should be learned')


# other params
misc_arg = add_argument_group('Misc.')
misc_arg.add_argument('--use_gpu', type=str2bool, default=False,
                      help="Whether to run on the GPU")
misc_arg.add_argument('--best', type=str2bool, default=True,
                      help='Load best model or most recent for testing')
misc_arg.add_argument('--random_seed', type=int, default=1,
                      help='Seed to ensure reproducibility')
misc_arg.add_argument('--data_dir', type=str, default='./data',
                      help='Directory in which data is stored')
misc_arg.add_argument('--ckpt_dir', type=str, default='./ckpt',
                      help='Directory in which to save model checkpoints')
misc_arg.add_argument('--resume', type=str2bool, default=False,
                      help='Whether to resume training from checkpoint')
misc_arg.add_argument('--print_freq', type=int, default=10,
                      help='How frequently to print training details')
misc_arg.add_argument('--plot_freq', type=int, default=1000000,
                      help='How frequently to plot glimpses')


def get_config():
    config, unparsed = parser.parse_known_args()
    return config, unparsed
