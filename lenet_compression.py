import torch
import torch.nn as nn

from data.mnist import MnistDataset
from models.lenet.lenet import LeNet5
from utils.train import *
from utils.weight_sharing import *
from utils.plot import *

from compress_optim import *

# net train params - before compression
LEARNING_RATE = 0.0001
BATCH_SIZE = 32
N_CLASSES = 10
DEVICE = None
EPOCHS = 100
# net save path
NET_PATH = './models/lenet/saves/lenet.save'
# optimization settings
RANGE_OPTIMIZATION = True
RANGE_OPTIMIZATION_TRESHOLD = 0.97

def compress_lenet(compress_alg, search_ranges, num_iter, num_pop):

    # initing the lenet model
    dataset = MnistDataset(BATCH_SIZE, './data', val_split=0.5)
    model = LeNet5(N_CLASSES)
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

    # oprimizing search ranges
    lam_test_inp = lambda _ : get_accuracy(model, dataset.test_dl, DEVICE)
    if RANGE_OPTIMIZATION:
        search_ranges = ws_controller.get_optimized_layer_ranges(search_ranges, lam_test_inp, 
            RANGE_OPTIMIZATION_TRESHOLD, savefile='./results/lenet-layer-perf.save')

    # compression part
    save_data = None
    if compress_alg == 'genetic':
        save_data = compression_genetic_optim(num_iter, num_pop, search_ranges, before_loss, model, train_settings, ws_controller, NET_PATH)
    elif compress_alg == 'pso':
        save_data = compression_pso_optim(num_iter, num_pop, search_ranges, before_loss, model, train_settings, ws_controller, NET_PATH)
    elif compress_alg == 'random':
        save_data = compression_random_optim(num_iter * num_pop, search_ranges, before_loss, model, train_settings, ws_controller, NET_PATH)
    else:
        raise Exception('err')

if __name__ == '__main__':
    compress_lenet('random' ,[range(1, 21), range(1, 21), range(1, 21), range(1, 21), range(1, 21)], 2, 3)
    print('done')
    compress_lenet('pso' ,[range(1, 21), range(1, 21), range(1, 21), range(1, 21), range(1, 21)], 2, 3)
    print('done')
    compress_lenet('genetic' ,[range(1, 21), range(1, 21), range(1, 21), range(1, 21), range(1, 21)], 2, 3)
    print('done')
