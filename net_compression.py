import torch
import torch.nn as nn
import argparse
import yaml
import sys

from data.imagenette import ImagenetteDataset
from data.utils.imagenet_utils import *
from utils.weight_sharing import *
from utils.plot import *
from utils.fitness_controller import FitnessController
from utils.genetic import dump_ga_config, load_ga_config

from compress_optim import *

# net train params - before compression
BATCH_SIZE = 32
DEVICE = CompressConfig.DEVICE
NET_TYPE = 'mobilenet_v2'
NET_REPO = 'pytorch/vision:v0.10.0'
# dataset settings
DATA_PATH = './data/imagenette'
TOP_ACC = 1
# optimization settings
RANGE_OPTIMIZATION = True
RANGE_OPTIMIZATION_TRESHOLD = 0.60
RANGE_OPTIMIZATION_FILE = f'./models/{NET_TYPE}/saves/{NET_TYPE}_layer_perf.csv'

def compress_net(compress_alg:str, search_ranges:tuple, num_iter:int, num_pop:int, show_plt:bool=False, save_plt:bool=False) -> None:
    """Lenet compression function.

    Args:
        compress_alg (str): is the optimizer algorithm for the optimization.
        TODO: search_ranges (tuple): list of touple of cluster ranges for each layer.
        num_iter (int): number of iterations for the algorithm.
        num_pop (int): size of the population for algorithm.
        show_plt (bool, optional): If True, the plot is shown. Defaults to False.
        save_plt (bool, optional): If True, the plot is saved. Defaults to False.

    Raises:
        Exception: If unknown optimization algorithm is choosed.
    """

    # initing the model
    if CompressConfig.VERBOSE:
        print('initing model')

    # ----------------------------------
    dataset = ImagenetteDataset(BATCH_SIZE, DATA_PATH, val_split=0.99)
    model = torch.hub.load(NET_REPO, NET_TYPE, pretrained=True)
    #criterion = nn.CrossEntropyLoss()
    #optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    #train_settings = [criterion, optimizer, dataset, EPOCHS, DEVICE, 1, True]
    #get_trained(model, NET_PATH, train_settings)

    # defining model hooks
    lam_opt = None #lambda mod : torch.optim.Adam(mod.parameters(), lr=LEARNING_RATE)
    lam_train = None #lambda opt, epochs : train_net(model, criterion, opt, dataset, epochs, device=DEVICE)
    lam_test = lambda : get_accuracy(model, dataset.test_dl, DEVICE, topk=TOP_ACC)
    # ----------------------------------

    # initing weightsharing
    if CompressConfig.VERBOSE:
        print('initing weight sharing')
    ws_controller = WeightShare(model, lam_test, lam_opt, lam_train)
    ws_controller.set_reset()
    lam_fitness_vals_fc = lambda i: fitness_vals_fc(i, ws_controller)

    # creating and optimizing search ranges
    #repr_range = [range(args.lower_range, args.upper_range) for _ in range(5)]
    low, up = search_ranges
    search_ranges = [range(low, up) for _ in ws_controller.model_layers]
    lam_test_inp = lambda _ : lam_test()
    if RANGE_OPTIMIZATION:
        if CompressConfig.VERBOSE:
            print('range optimization')
        search_ranges = ws_controller.get_optimized_layer_ranges(search_ranges, lam_test_inp, 
            RANGE_OPTIMIZATION_TRESHOLD, savefile=RANGE_OPTIMIZATION_FILE)

    # get before loss
    before_loss = lam_test()
    if CompressConfig.VERBOSE:
        print(f'before loss: {before_loss}')

    # defining fitness controller
    fit_controll = FitnessController(CompressConfig.OPTIM_TARGET, lam_fitness_vals_fc, fit_from_vals, 
        target_max_offset=CompressConfig.OPTIM_TARGET_UPDATE_OFFSET , lock=CompressConfig.OPTIM_TARGET_LOCK, 
        target_limit=CompressConfig.OPTIM_TARGET_LOW_LIMIT)

    # compression part
    save_data = None
    if compress_alg == 'genetic':
        save_data = compression_genetic_optim(num_iter, num_pop, search_ranges, before_loss, fit_controll)
    elif compress_alg == 'pso':
        save_data = compression_pso_optim(num_iter, num_pop, search_ranges, before_loss, fit_controll)
    elif compress_alg == 'random':
        save_data = compression_random_optim(num_iter * num_pop, search_ranges, before_loss, fit_controll)
    elif compress_alg == 'blackhole':
        save_data = compression_bh_optim(num_iter, num_pop, search_ranges, before_loss, fit_controll)
    else:
        raise Exception('err compress_lenet - unknown compress optimization algorithm')

    # plotting data
    plot_alcr(save_data)
    if show_plt:
        plt.show()
    if save_plt:
        plt.savefig(f'results/plots/{compress_alg}.png', format='pdf')

if __name__ == '__main__':

    parser = argparse.ArgumentParser(prog='lenet_compression.py', description='Optimizes compression of LeNet-5 CNN by different algorithms.'+
    'The outputs are storedn in the results folder.')
    parser.add_argument('-comp', '--compressor', choices=['random', 'pso', 'genetic', 'blackhole'], default='random', help='choose the compression algorithm')
    parser.add_argument('-pop', '--num_population', metavar='N', type=int, default=12, help='set the population count')
    parser.add_argument('-its', '--num_iterations', metavar='N', type=int, default=30, help='set the iteration count')
    parser.add_argument('-up', '--upper_range', metavar='N', type=int, default=51, help='sets the upper range for compression')
    parser.add_argument('-lo', '--lower_range', metavar='N', type=int, default=1, help='sets the lower range for compression')
    parser.add_argument('-hp', '--hide', action='store_false', help='does not show the output plot')
    parser.add_argument('-sv', '--save', action='store_true', help='saves the output plot')
    parser.add_argument('-cfs', '--config_save', type=argparse.FileType('w'), help='dumps current config in given file and ends')
    parser.add_argument('-cfl', '--config_load', type=argparse.FileType('r'), help='loads config from given `.yaml` file')
    args = parser.parse_args()

    # save config
    if args.config_save is not None:
        cfg = dump_comp_config()
        cfg['ga'] = dump_ga_config()
        cfg['net'] = {
            'name': 'Le-Net',
            'type': NET_TYPE,
        }
        cfg['compress space'] = {
            'up': args.upper_range,
            'down': args.lower_range,
            'optimized': RANGE_OPTIMIZATION,
            'opt tresh': RANGE_OPTIMIZATION_TRESHOLD,
        }
        yaml.dump(cfg, args.config_save)
        sys.exit(0)

    # load config
    if args.config_load is not None:
        cfg = yaml.safe_load(args.config_load)
        load_comp_config(cfg)
        load_ga_config(cfg['ga'])
        NET_TYPE = cfg['net']['type']
        RANGE_OPTIMIZATION = cfg['compress space']['optimized']
        RANGE_OPTIMIZATION_TRESHOLD = cfg['compress space']['opt tresh']

    compress_net(args.compressor, (args.lower_range, args.upper_range), args.num_iterations, args.num_population, args.hide, args.save)
