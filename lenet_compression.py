import torch
import torch.nn as nn
import argparse

from data.mnist import MnistDataset
from models.lenet.lenet import LeNet5
from utils.train import *
from utils.weight_sharing import *
from utils.plot import *
from utils.fitness_controller import FitnessController

from compress_optim import *

# net train params - before compression
LEARNING_RATE = 0.0001
BATCH_SIZE = 32
N_CLASSES = 10
EPOCHS = 100
DEVICE = CompressConfig.DEVICE
NET_TYPE = 'tanh'
# net save path
NET_PATH = './models/lenet/saves/lenet_tanh.save'
# optimization settings
RANGE_OPTIMIZATION = True
RANGE_OPTIMIZATION_TRESHOLD = 0.97
RANGE_OPTIMIZATION_FILE = './results/lenet-tanh-layer-perf.csv'

def compress_lenet(compress_alg:str, search_ranges:list, num_iter:int, num_pop:int, show_plt:bool=False, save_plt:bool=False) -> None:

    # initing the lenet model
    dataset = MnistDataset(BATCH_SIZE, './data', val_split=0.5)
    model = LeNet5(N_CLASSES, NET_TYPE)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    train_settings = [criterion, optimizer, dataset, EPOCHS, DEVICE, 1, True]
    get_trained(model, NET_PATH, train_settings)
    before_loss = get_accuracy(model, dataset.test_dl, DEVICE)

    # defining lenet hooks
    lam_opt = lambda mod : torch.optim.Adam(mod.parameters(), lr=LEARNING_RATE)
    lam_train = lambda opt, epochs : train_net(model, criterion, opt, dataset, epochs, device=DEVICE)
    lam_test = lambda : get_accuracy(model, dataset.test_dl, DEVICE)

    # initing weightsharing
    ws_controller = WeightShare(model, lam_opt, lam_train, lam_test)
    ws_controller.set_reset()

    # oprimizing search ranges
    lam_test_inp = lambda _ : get_accuracy(model, dataset.test_dl, DEVICE)
    if RANGE_OPTIMIZATION:
        search_ranges = ws_controller.get_optimized_layer_ranges(search_ranges, lam_test_inp, 
            RANGE_OPTIMIZATION_TRESHOLD, savefile=RANGE_OPTIMIZATION_FILE)

    # defining fitness controller
    fit_controll = FitnessController(CompressConfig.OPTIM_TARGET, fitness_vals_fc, fit_from_vals, 
        target_max_offset=CompressConfig.OPTIM_TARGET_UPDATE_OFFSET , lock=CompressConfig.OPTIM_TARGET_LOCK)

    # compression part
    save_data = None
    if compress_alg == 'genetic':
        save_data = compression_genetic_optim(num_iter, num_pop, search_ranges, before_loss, fit_controll)
    elif compress_alg == 'pso':
        save_data = compression_pso_optim(num_iter, num_pop, search_ranges, before_loss, fit_controll)
    elif compress_alg == 'random':
        save_data = compression_random_optim(num_iter * num_pop, search_ranges, before_loss, fit_controll)
    elif compress_alg == 'bh':
        save_data = compression_bh_optim(num_iter, num_pop, search_ranges, before_loss)
    else:
        raise Exception('err')

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
    args = parser.parse_args()

    repr_range = [range(args.lower_range, args.upper_range) for _ in range(5)]
    compress_lenet(args.compressor, repr_range, args.num_iterations, args.num_population, args.hide, args.save)
