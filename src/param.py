import argparse
import random
import numpy as np
import torch


def get_optimizer(optim):
    # Bind the optimizer
    if optim == 'rms':
        print("Optimizer: Using RMSProp")
        optimizer = torch.optim.RMSprop
    elif optim == 'adam':
        print("Optimizer: Using Adam")
        optimizer = torch.optim.Adam
    elif optim == 'adamax':
        print("Optimizer: Using Adamax")
        optimizer = torch.optim.Adamax
    elif optim == 'sgd':
        print("Optimizer: sgd")
        optimizer = torch.optim.SGD
    elif 'bert' in optim:
        optimizer = 'bert'      # The bert optimizer will be bind later.
    else:
        assert False, "Please add your optimizer %s in the list." % optim

    return optimizer


def parse_args():
    parser = argparse.ArgumentParser()

    # feature_extractor
    parser.add_argument('--crop_w', dest='crop_w', type=int, default=112)
    parser.add_argument('--crop_h', dest='crop_h', type=int, default=112)
    parser.add_argument('--resize_w', dest='resize_w', type=int, default=128)
    parser.add_argument('--resize_h', dest='resize_h', type=int, default=168)


    # Data Splits
    parser.add_argument("--train", default='train')
    parser.add_argument("--valid", default=None)
    parser.add_argument("--test", default="test")

    # Training Hyper-parameters
    parser.add_argument('--batchSize', dest='batch_size', type=int, default=32)
    parser.add_argument('--testbatchSize', dest='test_batch_size', type=int, default=1)
    parser.add_argument('--optim', default='adam')
    parser.add_argument('--lr', type=float, default=5e-5)
    parser.add_argument('--epochs', type=int, default=80)
    parser.add_argument('--seed', type=int, default=9595, help='random seed')

    # Model Loading
    parser.add_argument('--load', type=str, default=None,
                        help='Load the model (usually the fine-tuned model).')

    # Note: CRP = R, X, V (three encoders), Transformer
    parser.add_argument("--r_layers", default=5, type=int, help='Number of radar layers')
    parser.add_argument("--x_layers", default=5, type=int, help='Number of CROSS-modality layers.')
    parser.add_argument("--v_layers", default=5, type=int, help='Number of vision layers.')

    # Training configuration
    parser.add_argument("--multiGPU", action='store_const', default=False, const=True)
    parser.add_argument("--numWorkers", dest='num_workers', default=8)
    parser.add_argument("--nums_class", dest='nums_class', default=10)
    parser.add_argument("--output", dest='output', default="../../exps/crp_model/")
    parser.add_argument("--expid", dest='expid_name', default="01_CRP")

    parser.add_argument("--hidden_size", default=768)
    parser.add_argument("--num_attention_heads", default=12)
    parser.add_argument("--hidden_act", default='gelu')
    parser.add_argument("--intermediate_size", default=768)
    parser.add_argument("--hidden_dropout_prob", default=0.1)
    parser.add_argument("--attention_probs_dropout_prob", default=0.1)

    # Parse the arguments.
    args = parser.parse_args()

    # Bind optimizer class.
    args.optimizer = get_optimizer(args.optim)

    # Set seeds
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    return args


args = parse_args()
