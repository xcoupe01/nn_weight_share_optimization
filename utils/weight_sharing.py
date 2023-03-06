import torch
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
from sklearn.cluster import KMeans
import math
import time
import copy
import csv
import os

from utils.float_prec_reducer import FloatPrecReducer

BITS_IN_BYTE = 8
FLOAT8_SIGNIFICAND_LEN = 3

class Layer:

    def __init__(self, name:str, prec_reducer:FloatPrecReducer = None) -> None:
        """Inits the Layer object with given values

        Args:
            name (str): is the name of the layer.
            prec_reducer (FloatPrecReducer, optional): is the float precision reducer link. Defaults to None.
        """

        self.name = name
        self.weight = None
        self.bias = None
        self.virt_weight_params = 0
        self.virt_bias_params = 0
        self.prec_reducer = prec_reducer
        self.elem_size = None
        self.reset_weight_vals = None
        self.reset_weight_vals = None

    def __repr__(self) -> str:
        return "Layer(" + self.name + ")"

    def set_weight(self, weight: torch.Tensor) -> None:
        """Sets the weights and corresponding data.

        Args:
            weight (torch.Tensor): is the weight to be set.
        """
        self.weight = weight
        self.virt_weight_params = weight.numel()

    def set_reset_weight(self) -> None:
        """Sets the reset point of the network.
        """

        self.reset_weight_vals = {
            'weights': copy.deepcopy(torch.flatten(self.weight.cpu()).detach().numpy()),
            'elem_size': self.elem_size,
            'grad': self.weight.requires_grad,
            'virt_num_param': self.virt_weight_params,
        }

    def reset_weight(self) -> None:
        """Resets the net to the previously given reset point.

        Raises:
            Exception: If the reset point was not set before.
        """

        if self.reset_weight_vals is None:
            raise Exception('Layer reset error - no value to reset to')
        
        device = self.weight.device
        reset_w = torch.tensor(copy.deepcopy(self.reset_weight_vals['weights'])).to(device).reshape(self.weight.shape)

        self.weight.data = reset_w
        self.virt_weight_params = self.reset_weight_vals['virt_num_param']
        self.elem_size = self.reset_weight_vals['elem_size']
        self.weight.requires_grad = self.reset_weight_vals['grad']
        

    def set_bias(self, bias:torch.Tensor) -> None:
        """Sets the bias and corresponding data

        Args:
            bias (torch.Tensor): the bias to be set.
        """
        self.bias = bias
        self.virt_bias_params = bias.numel()

    def plot_weight(self, cluster_centers: list=None, point_labels:list=None) -> None:
        """Plots weight histogram of this layer. 
        It is possible to add cluster centers into the graph

        Args:
            cluster_centers (list, optional): Is the cluster centers to be added. Defaults to None.
            point_labels (list, optional): Is the data labels to be displayed. Defaults to None.
            point_labels (list, optional): Is the data labels to be displayed. Defaults to None.
        """

        plt.clf()
        plt.figure(figsize=(18,6))
        plt.title(self.name)
        plt.xlabel(f'{self.name} layer weights')
        plt.ylabel('count')

        sns.histplot(
            data = pd.DataFrame({
                'weights': torch.flatten(self.weight.cpu()).detach().numpy(),
                'cluster': point_labels,
            }),
            x = 'weights',
            bins = 100,
            hue = 'cluster' if point_labels is not None else None,
            multiple = 'stack'
        )

        if cluster_centers is not None:
            for center in cluster_centers:
                plt.axvline(center, color='r')

    def share_weight(self, n_weights:int, plot:bool = False, assign:bool = False, unlock:bool = True, prec_rtype:str = None, 
        mod_focus:float = None, mod_spread:float = None) -> None:
        """Runs clustering algorithm to determine the centroinds of given number of clustert, then
        computes the correct weight tensor for the network.

        Args:
            n_weights (int): is the number of clusters in the new weight.
            plot (bool, optional): if true, the new weight tensor is plotted. Defaults to False.
            assign (bool, optional): If true the new weight is assgined to the network. Defaults to False.
            unlock (bool, optional): If true, the new weights are set as unlocked in the model. Defaults to True.
            prec_rtype (string, optional): If not None, it defines the float precision reduction type
                (for more information go to 'utils/float_prec_reducer/FloatPrecReducer.py'). Defaults to None.
            mod_focus (float, optional): Modifier of the clustering space - 0 means not modified, the larger the number, 
                the larger the number the more clusters are shifted towards the mean of weights. The bigger the number
                the more is the modification focused on the center point. Defaults to None (== 0.0).
            mod_spread (float, optional): Modifier of the clustering space - 0 means not modigied, the larger the number,
                the more is the the modification spreaded (most spread is around the focus point). Defaults to None (== 0.0).

            The k-means `y` space modification is calculated by following expression:

            y = mods_spread * (x.max - x.min) * tanh(mod_focus * (x - x.mean))

        Raises:
            Exception: If the shaing is runned after locking the last sharing, then an error is raised.
        """

        if not self.weight.requires_grad:
            raise Exception('Layer share_weight error - already locked')
        
        mod_focus = 0 if mod_focus is None else mod_focus
        mod_spread = 0 if mod_spread is None else mod_spread
        
        # getting the current weights and preparing them for clustering
        original_shape = self.weight.shape
        original_type = self.weight.dtype
        device = self.weight.device

        numpy_weights = torch.flatten(self.weight.cpu()).detach().numpy()

        # adjustment of distances by adding new dimension
        numpy_weights_2D = np.vstack((
            numpy_weights, 
            mod_spread * np.ptp(numpy_weights) * np.tanh(mod_focus * (numpy_weights - np.mean(numpy_weights))) # adjustment to mean
        ))

        # plot the weights int the new dimension
        if plot:
            plt.figure(figsize=(18,6))
            plt.scatter(numpy_weights_2D[0], numpy_weights_2D[1], marker="+", label='weights')
            plt.axvline(x=np.mean(numpy_weights), color='r', label='mean')
            plt.axvline(x=0, color='g', label='zero')
            plt.title(f'{self.name} weights space for K-means clustering')
            plt.xlabel('Original weights values')
            plt.ylabel('Modified weights values')
            plt.legend()
            plt.show()

        numpy_weights_2D = np.swapaxes(numpy_weights_2D, 0, 1)

        # clustering
        kmeans = KMeans(n_clusters=n_weights, random_state=42).fit(numpy_weights_2D)

        if plot:
            self.plot_weight(kmeans.cluster_centers_[:, [0]], kmeans.labels_)
        
        # precision reduction if possible
        processed_cluster_centers = kmeans.cluster_centers_
        if self.prec_reducer is not None and prec_rtype is not None:
            processed_cluster_centers = self.prec_reducer.reduce_list(processed_cluster_centers, prec_rtype)

        if assign:
            # assigning and locking the new shared tensor
            new_tensor = [processed_cluster_centers[i][0] for i in kmeans.labels_]
            new_tensor = torch.tensor(new_tensor).to(device)
            new_tensor = new_tensor.reshape(original_shape)

            if new_tensor.dtype != original_type:
                new_tensor.type(original_type)

            self.weight.data = new_tensor
            self.weight.requires_grad = unlock
            self.virt_weight_params = n_weights

            if self.prec_reducer is not None and prec_rtype is not None:
                self.elem_size = self.prec_reducer.get_prec_bytes(prec_rtype)

    def compression_rate(self, mapping_bits:int=None) -> float:
        """Computes compression rate of the layer by following expression:

        par: num of params in the weight tensor
        bits1: amount of bits required to store the original value
        bits1r: amoutn of bits required to store the original reduced value
        bits2: amount of bits required to store the share value (key)
        key: amount of unique shared values 
        
                  par * bits1
        -----------------------------------
        par * bits2 + key * (bits1r + bits2)

        Args:
            mapping_bits (int, optional): Sets the number of bits wanted to be used
                as a key to the codebook (and in the weight matrix). Defaults to None.

        Raises:
            Exception: if the mapping_bits is seted and the number of elements is
            higher than the mapping_bits can map.

        Returns:
            float: the compression rate of the layer.
        """

        # if not shared and locked - no compression
        if self.weight.requires_grad:
            return 1

        par = self.weight.numel()
        bits1 = self.weight.element_size() * BITS_IN_BYTE
        bits1r = self.elem_size * BITS_IN_BYTE if self.elem_size is not None else bits1
        
        # computing number of bits to represent the key to map
        # in case there is only one unique value, the expression gives 0
        # which breaks the rest of the calculation, so this fix is needed
        bits2 = max(math.ceil(math.log(self.virt_weight_params) / math.log(2)), 1)

        if mapping_bits is not None:
            if mapping_bits < bits2:
                raise Exception('Layer compression_rate error - cannot be mapped to given bits')
            bits2 = mapping_bits
        
        # computing the weight-key matrix
        bits_map_key = par * bits2

        # computing the weight mapping table 
        bits_map_vals = self.virt_weight_params * (bits1r + bits2)

        return (par * bits1) / (bits_map_key + bits_map_vals)

class WeightShare:

    def __init__(self, model: torch.nn.Module, opt_create, train, test) -> None:
        """Initializes the weight share object.

        Args:
            model (torch.nn.Module): is the mode thats going to be weight-shared.
            opt_create (function): is a function that returns new optimizer for the given network.
            train (function): is a function that trains the given network.
            test (function): is a function that tests the given network.

        Raises:
            Exception: if unknown net parameter is found.
        """

        self.model = model
        self.model_layers:list[Layer] = []
        self.opt_create = opt_create
        self.train = train
        self.test = test
        self.float_res_reducer = FloatPrecReducer(FLOAT8_SIGNIFICAND_LEN)

        # initing the internal representation of the layers
        for name, param in model.named_parameters():
            
            parsed_name = name.split('.')

            for layer in self.model_layers:
                if layer.name == ".".join(parsed_name[:-1]):
                    break
            else:
                layer = None

            if layer is None:
                self.model_layers.append(Layer(".".join(parsed_name[:-1]), self.float_res_reducer))
                layer = self.model_layers[-1]

            if parsed_name[-1] == 'weight':
                layer.set_weight(param)
            elif parsed_name[-1] == 'bias':
                layer.set_bias(param)
            else:
                raise Exception('WeightShare init error - unknown parameter')

    def print_layers_info(self):
        """Prints a short brief about the model thats being compressed by
        this object.
        """

        print('layer_name', '#weights', '#bias', 'w_locked', 'CR')

        weight_acc = 0
        bias_acc = 0

        for layer in self.model_layers:
            print(layer.name, layer.virt_weight_params, layer.virt_bias_params, 
                not layer.weight.requires_grad, f'{layer.compression_rate():.2f}')
            weight_acc += layer.virt_weight_params
            bias_acc += layer.virt_bias_params
        
        print('Sum num weights, bias: ', weight_acc, bias_acc) 
        print('Compression rate', f'{self.compression_rate():.2f}')

    def compression_rate(self, mapping_bits:int=None) -> float:
        """Computes compression rate of the whole model.
        If non compression is made yet, 1 is returned (no compression).

        Args:
            mapping_bits (int, optional): Defines how many bits will have the key part of the 
            compression table. Defaults to None.

        Returns:
            float: The compression rate of whole model.
        """
        layer_cr = [x.compression_rate(mapping_bits) for x in self.model_layers]
        return sum(layer_cr) / len(layer_cr)

    def share(self, layer_clusters:list, layer_order:list = None, retrain_amount:list = None, prec_reduct:list = None, mods_focus:list = None, 
        mods_spread:list = None, verbose:bool = False) -> dict:
        """Shares the entire model in a given orger to a given number of weight clusters for each layer
        and retrains the model by a given amount.

        Args:
            layer_clusters (list): Number of clusters for each layer. The list must be the same lenght as there are
                layers in the model
            layer_order (list, optional): Order of the compression of the layers. The compression is driven by this list. 
                The format is that the list has the indexes of layers in order to be shared. Defaults to None (all layers from input
                to output layer).
            retrain_amount (list, optional): specifies the retrain amount - the index of the retrain amount corresponds to the
                layer to be retrained. Defaults to None (no retraining).
            prec_rtype (list, optional): If not None, it defines the float precision reduction type for each layer
                (for more information go to 'utils/float_prec_reducer/FloatPrecReducer.py'). Defaults to None.
            mods_focus (list, optional): If not None, it defines the clustering space modification for each layer.
                The modifications is described in the Layer.share function as mod_focus parameter. Defaults to None.
            mods_spread (list, optional): If not None, it defines the clustering space modification for each layer
                The modification is described in the Layer.share function as mod_spread parameter. Defaults to None.
            verbose (bool, optional): To print information about the sharing during the execution. Defaults to False.

        Raises:
            Exception: if the input parameters are bad (lists do not correspond).

        Returns:
            dict: dictionary that contents with information about the sharing.
            The format is like so:
                accuracy: the model accuracy after compression
                compresion: the model compression rate
                time: 
                    - train: the model total retrain time
                    - share: the model total share time
                    - test: the model total test time
        """

        # parameter check
        if  len(layer_clusters) != len(self.model_layers) or \
            layer_order is not None and (max(layer_order) >= len(self.model_layers) or \
            min(layer_order) < 0 or \
            len(layer_order) > len(self.model_layers)) or \
            retrain_amount is not None and len(retrain_amount) != len(self.model_layers) or \
            prec_reduct is not None and len(prec_reduct) != len(self.model_layers) or \
            mods_focus is not None and len(mods_focus) != len(self.model_layers) or \
            mods_spread is not None and len(mods_spread) != len(self.model_layers):
        # parameter check

            print(
                len(layer_clusters) != len(self.model_layers),
                layer_order is not None and (max(layer_order) >= len(self.model_layers) or \
                min(layer_order) < 0 or \
                len(layer_order) > len(self.model_layers)),
                retrain_amount is not None and len(retrain_amount) != len(self.model_layers),
                prec_reduct is not None and len(prec_reduct) != len(self.model_layers),
                mods_focus is not None and len(mods_focus) != len(self.model_layers),
                mods_spread is not None and len(mods_spread) != len(self.model_layers),
                )
            print(layer_clusters)
            print(layer_order)
            print(retrain_amount)
            print(prec_reduct)
            print(mods_focus)
            print(mods_spread)

            raise Exception('WeightShare share error - lists not matching')

        # share init
        if layer_order is None:
            layer_order = range(len(self.model_layers))
        if retrain_amount is None:
            retrain_amount = [None for _ in self.model_layers]
        if prec_reduct is None:
            prec_reduct = [None for _ in self.model_layers]
        if mods_focus is None:
            mods_focus = [None for _ in self.model_layers]
        if mods_spread is None:
            mods_spread = [None for _ in self.model_layers]
        total_share = 0
        total_train = 0
        total_test = 0

        for num, layer in enumerate(layer_order):
            
            share_start = time.time()
            before_params = self.model_layers[layer].virt_weight_params

            # layer share phase
            start_share = time.time()
            self.model_layers[layer].share_weight(layer_clusters[layer], assign=True, unlock=False, prec_rtype=prec_reduct[layer], 
                mod_focus=mods_focus[layer], mod_spread=mods_spread[layer])
            total_share += time.time() - start_share

            # retrain phase
            if retrain_amount[layer] is not None and retrain_amount[layer] > 0:
                start_train = time.time()
                optimizer = self.opt_create(self.model)
                self.train(optimizer, retrain_amount[layer])
                total_train += time.time() - start_train

            start_test = time.time()    
            model_accuracy = self.test()
            total_test += time.time() - start_test

            share_duration = time.time() - share_start
            
            if verbose:
                print(
                    f'Share: {num+1}/{len(layer_order)} --- ',
                    f'Time: {share_duration:.2f}s\t',
                    f'Name: {self.model_layers[layer].name}\t'
                    f'Before params: {before_params}\t',
                    f'After params: {self.model_layers[layer].virt_weight_params}\t',
                    f'Test accuracy: {model_accuracy:.2f}%'
                )

        compression_rate = self.compression_rate()

        return {
            'accuracy': model_accuracy, 
            'compression': compression_rate,
            'times': {
                'train': total_train,
                'share': total_share,
                'test': total_test
            }
        }

    def finetuned_mod(self, layer_clusters:list, mods_focus:list, mods_spread:list, layer_order:list = None, prec_reduct:list = None,
        plot:bool = False) -> list:
        """Tryes to get the best shared net by modulating the space. Computationaly intesive to execute!

        Args:
            layer_clusters (list): number of clusters for each layer.
            mods_focus (list): possible values of focus modulation for each layer.
            mods_spread (list): the spread for each layer.
            layer_order (list, optional): The order of layer processing. Defaults to None (from first to last).
            prec_reduct (list, optional): The share precision reduction. Defaults to None.
            plot (bool, optional): If true, share plots are shown. Defaults to False.

        Returns:
            list: list of the best found focus modulations.
        """

        if layer_order is None:
            layer_order = range(len(self.model_layers))
        if prec_reduct is None:
            prec_reduct = [None for _ in self.model_layers]

        best_focus_vals = [None for _ in self.model_layers]

        # saving original weights and initial share
        for layer in self.model_layers:
            layer.set_reset_weight()
        
        # sharing without modifications
        self.share(layer_clusters, layer_order=layer_order, prec_reduct=prec_reduct)

        for layer in layer_order:

            # layer init
            w_delta = []
            acc = []
            #num_in_cluster_mean = len(org_weight) / SHARE_CLUSTERS[LAYER]

            for focus_val in mods_focus:
                
                # reset layer weights
                self.model_layers[layer].reset_weight()

                # focus scoring
                self.model_layers[layer].share_weight(layer_clusters[layer], assign=True, unlock=False, prec_rtype=prec_reduct[layer], 
                    mod_focus=focus_val, mod_spread=mods_spread[layer])

                post_share_weights = torch.flatten(self.model_layers[layer].weight.cpu()).detach().numpy()
                w_delta.append(np.sum(np.abs(post_share_weights - self.model_layers[layer].reset_weight_vals['weights'])))
                acc.append(self.test())

            # saving the best focus value and setting the net to this val
            best_focus_vals[layer] = mods_focus[np.argmax(acc)]

            self.model_layers[layer].reset_weight()

            self.model_layers[layer].share_weight(layer_clusters[layer], assign=True, unlock=False, prec_rtype=prec_reduct[layer], 
                mod_focus=best_focus_vals[layer], mod_spread=mods_spread[layer])

            # plot
            if plot:
                fig, ax1 = plt.subplots()

                color = 'tab:red'
                ax1.set_xlabel('focus modulation')
                ax1.set_ylabel('weight delta', color=color)
                ax1.plot(mods_focus, w_delta, color=color)
                ax1.tick_params(axis='y', labelcolor=color)

                ax2 = ax1.twinx()

                color = 'tab:blue'
                ax2.set_ylabel('accuracy', color=color)
                ax2.plot(mods_focus, acc, color=color)
                ax2.tick_params(axis='y', labelcolor=color)

                fig.tight_layout()
                plt.show()

        return best_focus_vals
    
    def set_reset(self, layers:list = None) -> None:
        """Sets the reset point of the whole net.

        Args:
            layers (list, optional): List of layers to set the reset point of. 
                Defaults to None (all layer reset points are set).
        """
        
        if layers is None:
            layers = range(len(self.model_layers))

        for layer in layers:
            self.model_layers[layer].set_reset_weight()

    def reset(self, layers:list = None) -> None:
        """Resets the net to previously set reset point.

        Args:
            layers (list, optional): List of layers to be reset. 
                Defaults to None (all layers are reseted).
        """
        
        if layers is None:
            layers = range(len(self.model_layers)) 

        for layer in self.model_layers:
            layer.reset_weight()

    def get_layer_cluster_nums_perf(self, layer_i:int, layer_range:list, perf_fcs:list, pre_perf_fc=None, prec_rtype:str=None) -> list:
        """Gets layer cluster performace for a given set of cluster numbers. After executing
        the model is returned to original state.

        Args:
            layer_i (int): is the layer that is going to be compressed to get the performace.
            layer_range (list): is the list of the number of clusters wanted to be tested.
            perf_fcs (list): are functions that will test the performace.
            pre_perf_fc (function, optional): is a function that is performed before the test.
                It recieves the model layer after sharing. Defaults to None.
            prec_rtype (str, optional): If not None, it defines the float precision reduction type 
                (for more information go to 'utils/float_prec_reducer/FloatPrecReducer.py'). Defaults to None.

        Raises:
            Exception: if the layer is already locked, error is raised.

        Returns:
            list: list whic contains tuples in format: (cluster number, [scores in the order of perf_fcs])
        """

        if not self.model_layers[layer_i].weight.requires_grad:
            raise Exception('Layer get_best_cluster_num error - Already altered weight cannot be precessed')
        
        # saving original data
        self.set_reset()

        output = []

        for k in layer_range:
            
            # clustering and scoring the solution
            self.model_layers[layer_i].share_weight(k, assign=True, unlock=False, prec_rtype=prec_rtype)
            if pre_perf_fc is not None:
                pre_perf_fc(self.model_layers[layer_i])
            k_score = [x(self.model_layers[layer_i]) for x in perf_fcs]
            output.append((k, k_score))

            # returning original data
            self.reset()

        return output

    def get_layers_cluster_nums_perfs(self, layer_ranges:list, perf_fcs, pre_perf_fc=None, prec_rtype:str=None) -> list:
        """Gets all the layers cluster performace for a given set of cluster numbers. After executing
        the model is returned to original state.

        Args:
            layer_ranges (list): is the list of lists of the number of clusters wanted to be tested. 
                Index of the list corresponds to an index of layer in the model.
            perf_fcs (function): are functions that will test the performace.
            pre_perf_fc (_type_, optional): is a function that is performed before the test.
                It recieves the model layer after sharing. Defaults to None.
            prec_rtype (str, optional): If not None, it defines the float precision reduction type 
                (for more information go to 'utils/float_prec_reducer/FloatPrecReducer.py'). Defaults to None.

        Raises:
            Exception: When the layer_ranges do not correspond to the model layers as described an exception is raised.

        Returns:
            list: list in format where index represents layer. Each layer have list of clusters number and corresponding
            list of performaces given by the perf_fcs.
        """
        if len(self.model_layers) != len(layer_ranges):
            raise Exception('WeightShare get_layers_cluster_nums_perfs error - given ranges do not correspond to model layers')

        layer_perfs = []

        for i, layer_range in enumerate(layer_ranges):
            layer_perfs.append(self.get_layer_cluster_nums_perf(i, layer_range, perf_fcs, pre_perf_fc, prec_rtype=prec_rtype))

        return layer_perfs

    def get_optimized_layer_ranges(self, layer_ranges:list, perf_fc, perf_lim:float=None, 
    max_num_range:int=None, savefile:str=None, pre_perf_fc=None, prec_rtype:str=None) -> list:
        """Gets the oprimized clusters nubers for every layer of the model by given metrics.
        The metrics can be that the model with applied clusters number accuracy cant
        get below certain number and/or the number of clusters number can be limited
        excplicitly.

        Args:
            layer_ranges (list): is a list where index corespond to a layer in the model. Each
                index contains a list of possible cluster numbers wanted to be optimized.
            perf_fc (function): is the performace function (probably accuracy getter)
            perf_lim (float, optional): is the lowest accuracy acceptable. Defaults to None.
            max_num_range (int, optional): is the maximum number of clusters numbers for a layer. Defaults to None.
            savefile (str, optional): is a savefile to be loaded from or saved to. Defaults to None.
            pre_perf_fc (function, optional): is a function that is performed before the test.
                It recieves the model layer after sharing. Defaults to None.
            prec_rtype (str, optional): If not None, it defines the float precision reduction type 
                (for more information go to 'utils/float_prec_reducer/FloatPrecReducer.py'). Defaults to None.

        Returns:
            list: a list where an index corresponds to a given layer. Each index contains a list of optimized
            clusters numbers for a given layer.
        """
        
        perfs = []

        # load performances from file
        if savefile is not None and os.path.isfile(savefile):
            with open(savefile, 'r') as f:
                reader = csv.reader(f)
                for row in reader:
                    perfs.append([])
                    for item in row:
                        perfs[-1].append(eval(item))
        
        # compute performances if file not avalible
        else:
            perfs = self.get_layers_cluster_nums_perfs(layer_ranges, [perf_fc], pre_perf_fc, prec_rtype=prec_rtype)
            perfs = [[[y[0], y[1][0]] for y in x] for x in perfs]

        # save performaces if necessary
        os.makedirs(os.path.dirname(savefile), exist_ok=True)
        if savefile is not None and not os.path.isfile(savefile):
            with open(savefile, 'w') as f:
                write = csv.writer(f)
                write.writerows(perfs)

        # get only the ranges that match performace needs
        for i, perf in enumerate(perfs):
            if perf_lim is not None:
                perf = list(filter(lambda x: x[1] >= perf_lim, perf))
            if max_num_range is not None and len(perf) > max_num_range:
                perf.sort(key = lambda x: x[1], reverse=True)
                perf = perf[:max_num_range]
                perf.sort(key = lambda x: x[0])
            perfs[i] = [x[0] for x in perf]

        return perfs
