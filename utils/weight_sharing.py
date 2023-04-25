#!/usr/bin/env python

"""
Author: Vojtěch Čoupek
Description: Implementation of Weight-Sharing techique for pytorch models
"""

import torch
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.mixture import GaussianMixture
from typing import Callable
import math
import time
import copy
import csv
import os
from joblib import parallel_backend
from scipy.stats.stats import pearsonr

from utils.float_prec_reducer import FloatPrecReducer

# some default settings for the sharing
BITS_IN_BYTE = 8
FLOAT8_SIGNIFICAND_LEN = 3
DEFAULT_WS_LAYERS = (torch.nn.Conv1d, torch.nn.Conv2d, torch.nn.Conv3d, torch.nn.Linear)
KMEANS_N_JOBS_TREAD = 4     # tells how many jobs can kmeans use
MIN_SAVED_BITS = 1          # value 0 or 1 - determines computation in cr - see compute_cr function

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
        self.clust_inertia = 0

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
            'weights': torch.flatten(torch.clone(self.weight).cpu()).detach().numpy(),
            'elem_size': self.elem_size,
            'grad': self.weight.requires_grad,
            'virt_num_param': self.virt_weight_params,
            'clust_inertia': self.clust_inertia,
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
        self.clust_inertia = self.reset_weight_vals['clust_inertia']

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
        """

        plot_weights(
            torch.flatten(self.weight.cpu()).detach().numpy(), 
            cluster_centers, 
            point_labels,
            self.name)

    def share_weight(self, n_weights:int, plot:bool = False, assign:bool = False, unlock:bool = True, prec_rtype:str = None, 
        mod_focus:float = None, mod_spread:float = None, n_clust_jobs:int = KMEANS_N_JOBS_TREAD, clust_alg:str = 'kmeans') -> None:
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
            n_clust_jobs (int, optional): Specifies the multiprocessing of clustering. Děfaults to KMEANS_N_JOBS_THREAD specified
                in the source code.
            clust_alg (str, optional): Specifies the clustering algorithm. 'kmeas' for classical K-means, 'minibatch-kmeas' for 
                minibatch kmeans and 'gmm' for gaussian mixture model. Defaults to 'kmeans'.

            The k-means `y` space modification is calculated by following expression:

            y = mods_spread * (x.max - x.min) * tanh(mod_focus * (x - x.mean))

        Raises:
            Exception: If the shaing is runned after locking the last sharing, then an error is raised.
            Exception: If non valid clust_alg parameter is passed
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

        # plot the weights in the new dimension
        if plot:
            plot_kmeans_space(numpy_weights_2D, self.name)
            
        numpy_weights_2D = np.swapaxes(numpy_weights_2D, 0, 1)

        # clustering
        with parallel_backend('threading', n_jobs=n_clust_jobs):
            if clust_alg == 'kmeans':
                kmeans = MiniBatchKMeans(n_clusters=n_weights, random_state=42).fit(numpy_weights_2D)
                labels = kmeans.labels_
                cluster_centers = kmeans.cluster_centers_[:, [0]]
                self.clust_inertia = kmeans.inertia_
            elif clust_alg == 'minibatch-kmeans':
                kmeans = KMeans(n_clusters=n_weights, random_state=42).fit(numpy_weights_2D)
                labels = kmeans.labels_
                cluster_centers = kmeans.cluster_centers_[:, [0]]
                self.clust_inertia = kmeans.inertia_
            elif clust_alg == 'gmm':
                GMM = GaussianMixture(n_components=n_weights, random_state=42).fit(numpy_weights_2D)
                labels = GMM.predict(numpy_weights_2D)
                cluster_centers = GMM.means_[:, [0]]
                self.clust_inertia = 0
            else:
                raise Exception(f'Layer share_weight error - unknown clustering algorithm: {clust_alg}')

        # reduction of additional dimension
        processed_cluster_centers = np.concatenate(cluster_centers)
        
        # precision reduction if possible - reduce to the important dimesion
        if self.prec_reducer is not None and prec_rtype is not None:
            processed_cluster_centers = self.prec_reducer.reduce_list(processed_cluster_centers, prec_rtype)
        
        if plot:
            self.plot_weight(processed_cluster_centers, labels)

        if assign:
            # assigning and locking the new shared tensor
            new_tensor = [processed_cluster_centers[i] for i in labels]
            new_tensor = torch.tensor(new_tensor).to(device)
            new_tensor = new_tensor.reshape(original_shape)

            if new_tensor.dtype != original_type:
                new_tensor = new_tensor.type(original_type)

            self.weight.data = new_tensor
            self.weight.requires_grad = unlock
            self.virt_weight_params = n_weights

            if self.prec_reducer is not None and prec_rtype is not None:
                self.elem_size = self.prec_reducer.get_prec_bytes(prec_rtype)

    def compression_rate(self, mapping_bits:int=None) -> float:
        """Computes compression rate of the layer - see compute_cr() function to see details.

        Args:
            mapping_bits (int, optional): Sets the number of bits wanted to be used
                as a key to the codebook (and in the weight matrix). Defaults to None.

        Returns:
            float: the compression rate of the layer.
        """

        # if not shared and locked - no compression
        if self.weight.requires_grad:
            return 1

        bits_w_old = self.weight.element_size() * BITS_IN_BYTE

        return compute_cr(
            self.weight.numel(),
            self.virt_weight_params,
            bits_w_old,
            self.elem_size * BITS_IN_BYTE if self.elem_size is not None else bits_w_old,
            mapping_bits
            )

class WeightShare:

    def __init__(self, model: torch.nn.Module, 
        test:Callable[[None], float] = None, 
        opt_create:Callable[[torch.nn.Module], torch.optim.Optimizer] = None, 
        train:Callable[[torch.optim.Optimizer, int], None] = None, 
        shared_layer_types:tuple = DEFAULT_WS_LAYERS) -> None:
        """Initializes the weight shate object.

        Args:
            model (torch.nn.Module): is the model thats going to be weight-shared.
            test (Callable[[None], float], optional): is a function that tests the accuracy of a given network. 
            If None, no tests are done and the accuracy reading will default to -1. Defaults to None.
            opt_create (Callable[[torch.nn.Module], torch.optim.Optimizer], optional): is a function that returns new optimizer 
                of the given network which is used before every training. If None no retraining will be conducted. Defaults to None.
            train (Callable[[torch.optim.Optimizer, int], None], optional): is a function that trains the given network. 
                If None no retraining will be conducted. Defaults to None.
            shared_layer_types (tuple, optional): Layer types considered in weight sharing. 
                Defaults to tuple containing 1D, 2D and 3D covoluitons and linear layers.

        Raises:
            Exception: if unknown net parameter is found.
        """

        self.model = model
        self.model_layers:list[Layer] = []
        self.opt_create = opt_create
        self.train = train
        self.test = test
        self.float_res_reducer = FloatPrecReducer(FLOAT8_SIGNIFICAND_LEN)
        self.compress_total = None

        # getting the names of layers to be considered in ws
        layer_names = get_all_spec_layer_names(model=model, pytorch_ws_layers=shared_layer_types)

        # initing the internal representation of the layers
        for name, param in model.named_parameters():
            
            parsed_name = name.split('.')

            if ".".join(parsed_name[:-1]) not in layer_names:
                continue

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
        Takes into account if the model is shared layerwise or modelwise.

        The compression is computed in following steps:
        1) compute all the compression rates of all the layers.
        2) compute a mean of all the compressions.

        It is not weighted mean in the second step becase the size 
        of the layer is already taken into account in the first step.

        Args:
            mapping_bits (int, optional): Defines how many bits will have the key part of the 
            compression table. Defaults to None.

        Returns:
            float: The compression rate of whole model.
        """

        # if total share was done compute for whole model
        if self.compress_total is not None:
            
            return compute_cr(
                sum([l.weight.numel() for l in self.model_layers]),
                self.compress_total['clusters'],
                self.model_layers[0].weight.element_size() * BITS_IN_BYTE,
                self.float_res_reducer.get_prec_bytes(self.compress_total['prec_type']),
                mapping_bits
            )
        
        # compute layer-wise compression
        layer_cr = [x.compression_rate(mapping_bits) for x in self.model_layers]
        return sum(layer_cr) / len(layer_cr)

    def share(self, layer_clusters:list, layer_order:list = None, retrain_amount:list = None, prec_reduct:list = None, mods_focus:list = None, 
        mods_spread:list = None, verbose:bool = False, n_clust_jobs:int = KMEANS_N_JOBS_TREAD, clust_alg:str = 'kmeans') -> dict:
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
            n_clust_jobs (int, optional): Specifies the multiprocessing of clustering. Děfaults to KMEANS_N_JOBS_THREAD specified
                in the source code.
            clust_alg (str, optional): Specifies the clustering algorithm. 'kmeas' for classical K-means, 'minibatch-kmeas' for 
                minibatch kmeans and 'gmm' for gaussian mixture model. Defaults to 'kmeans'.

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

        if not all(v is None for v in retrain_amount) and (self.opt_create is None or self.train is None) and verbose:
            print('WeightShare share Warning - retrain will not be done, no train or opt_create functions given')
            
        for num, layer in enumerate(layer_order):
            
            share_start = time.time()
            before_params = self.model_layers[layer].virt_weight_params

            # layer share phase
            start_share = time.time()
            self.model_layers[layer].share_weight(layer_clusters[layer], assign=True, unlock=False, prec_rtype=prec_reduct[layer], 
                mod_focus=mods_focus[layer], mod_spread=mods_spread[layer], n_clust_jobs=n_clust_jobs, clust_alg=clust_alg)
            total_share += time.time() - start_share

            # retrain phase
            if retrain_amount[layer] is not None and retrain_amount[layer] > 0 \
                and self.train is not None and self.opt_create is not None:
                start_train = time.time()
                optimizer = self.opt_create(self.model)
                self.train(optimizer, retrain_amount[layer])
                total_train += time.time() - start_train

            share_duration = time.time() - share_start
            
            if verbose:
                print(
                    f'Share: {num+1}/{len(layer_order)} --- ',
                    f'Time: {share_duration:.2f}s\t',
                    f'Name: {self.model_layers[layer].name}\t'
                    f'Before params: {before_params}\t',
                    f'After params: {self.model_layers[layer].virt_weight_params}\t',
                )

        # test acc
        if self.test is not None:
            start_test = time.time()    
            model_accuracy = self.test()
            total_test += time.time() - start_test

        compression_rate = self.compression_rate()

        return {
            'accuracy': model_accuracy if self.test is not None else -1, 
            'compression': compression_rate,
            'inertias': [l.clust_inertia for l in self.model_layers],
            'times': {
                'train': total_train,
                'share': total_share,
                'test': total_test
            }
        }

    def share_total(self, model_clusters:int, mod_focus:int = 0, mod_spread:int = 0, prec_rtype:str = None, assign:bool = True, 
        plot:bool = False, unlock:bool = False, n_clust_jobs:int = KMEANS_N_JOBS_TREAD, clust_alg:str = 'kmeans') -> dict:
        """Shares the model not by layers but as a whole - takes all the weights in the model, clusters them and 
        the result is only one key-value table for the whole model.

        Args:
            model_clusters (int): Is the number of clusters to be share to for all the model weights.
            mod_focus (int, optiona): Is the kmeans space focus modulation parameter. Defaults to 0.
            mod_spread (int, optiona): Is the kmeans spread modulation parameter. Defaults to 0.
            prec_rtype (str, optional): Is the precision reduction type for the values. Defaults to None.
            assign (bool, optional): If true, the new weights are assigned to the model. Defaults to True.
            plot (bool, optional): If True, plots of the weights are shown. Defaults to False.
            unlock (bool, optional): If True, the model will be trainable after share. Defaults to False.
            n_clust_jobs (int, optional): Specifies the multiprocessing of clustering. Děfaults to KMEANS_N_JOBS_THREAD specified
                in the source code.
            clust_alg (str, optional): Specifies the clustering algorithm. 'kmeas' for classical K-means, 'minibatch-kmeas' for 
                minibatch kmeans and 'gmm' for gaussian mixture model. Defaults to 'kmeans'.

        Raises:
            Exception: If the model was already shared and locked, it cannot be shared and an Exception is rised.

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

        # init timers
        total_share = 0
        total_test = 0
        start_share = time.time()

        # load all weights        
        numpy_weights = np.array([])
        for layer in self.model_layers:
            if not layer.weight.requires_grad:
                raise Exception('WeightShare share_total error - already locked')
            numpy_weights = np.append(numpy_weights, torch.flatten(layer.weight.cpu()).detach().numpy())

        # adjustment of distances by adding new dimension
        numpy_weights_2D = np.vstack((
            numpy_weights, 
            mod_spread * np.ptp(numpy_weights) * np.tanh(mod_focus * (numpy_weights - np.mean(numpy_weights))) # adjustment to mean
        ))

        # plot the weights in the new dimension
        if plot:
            plot_kmeans_space(numpy_weights_2D, 'All')

        # clustering
        numpy_weights_2D = np.swapaxes(numpy_weights_2D, 0, 1)
        with parallel_backend('threading', n_jobs=n_clust_jobs):
            if clust_alg == 'kmeans':
                kmeans = MiniBatchKMeans(n_clusters=model_clusters, random_state=42).fit(numpy_weights_2D)
                labels = kmeans.labels_
                cluster_centers = kmeans.cluster_centers_[:, [0]]
                inertia = kmeans.inertia_
            elif clust_alg == 'minibatch-kmeans':
                kmeans = KMeans(n_clusters=model_clusters, random_state=42).fit(numpy_weights_2D)
                labels = kmeans.labels_
                cluster_centers = kmeans.cluster_centers_[:, [0]]
                inertia = kmeans.inertia_
            elif clust_alg == 'gmm':
                GMM = GaussianMixture(n_components=model_clusters, random_state=42).fit(numpy_weights_2D)
                labels = GMM.predict(numpy_weights_2D)
                cluster_centers = GMM.means_[:, [0]]
                inertia = 0
            else:
                raise Exception(f'WeightShare share_total error - unknown clustering algorithm: {clust_alg}')

        # reduction of additional dimension
        processed_cluster_centers = np.concatenate(cluster_centers)

        # precision reduction
        if self.float_res_reducer is not None and prec_rtype is not None:
            processed_cluster_centers = self.float_res_reducer.reduce_list(processed_cluster_centers, prec_rtype)

        if plot:
            plot_weights(numpy_weights, processed_cluster_centers, labels, 'All')
        
        # assigning weights if wanted
        if assign:
            new_weights = np.array([processed_cluster_centers[i] for i in labels])
            new_weights = np.split(new_weights, np.cumsum([l.weight.numel() for l in self.model_layers[:-1]]))
            for i, layer in enumerate(self.model_layers):
                new_tensor = torch.tensor(new_weights[i]).to(layer.weight.device)
                new_tensor = new_tensor.reshape(layer.weight.shape)

                if new_tensor.dtype != layer.weight.dtype:
                    new_tensor = new_tensor.type(layer.weight.dtype)

                layer.weight.data = new_tensor
                layer.weight.requires_grad = unlock

            # accuracy test
            if self.test is not None:
                start_test = time.time()    
                model_accuracy = self.test()
                total_test += time.time() - start_test

            # set to compressed
            self.compress_total = {
                'prec_type': prec_rtype if prec_rtype is not None else 'f4',
                'clusters': model_clusters,
            }
            compression_rate = self.compression_rate()

        total_share += time.time() - start_share

        return {
            'accuracy': model_accuracy if self.test and assign is not None else -1, 
            'compression': compression_rate if assign else -1,
            'inertia': inertia,
            'times': {
                'share': total_share,
                'test': total_test
            }
        }

    def finetuned_mod(self, layer_clusters:list, mods_focus:list, mods_spread:list, layer_order:list = None, prec_reduct:list = None,
        plot:bool = False, savefile:str = None, clust_alg:str = 'kmeans', verbose:bool = False, shared_model_savefile:str = None) -> list:
        """Tryes to get the best shared net by modulating the space. Computationaly intesive to execute!
        Needs to have the original net without modifications at the begining.

        Args:
            layer_clusters (list): number of clusters for each layer.
            mods_focus (list): possible values of focus modulation for each layer.
            mods_spread (list): the spread for each layer.
            layer_order (list, optional): The order of layer processing. Defaults to None (from first to last).
            prec_reduct (list, optional): The share precision reduction. Defaults to None.
            plot (bool, optional): If true, share plots are shown. Defaults to False.
            savefile (str, optional): If specified, the data are loaded and save into the file. The file format is
                csv where each row corresponds to a layer and each colum to differend focus value, in the cells there is an array
                in following format - [focus value, w_delta, inertia, accuracy]. Defaults to None.
            clust_alg (str, optional): Specifies the clustering algorithm. 'kmeas' for classical K-means, 'minibatch-kmeas' for 
                minibatch kmeans and 'gmm' for gaussian mixture model. Defaults to 'kmeans'.
            TODO: verbose

        Returns:
            list: list of the best found focus modulations.
        """

        if self.test is None:
            raise Exception('WeightSharing finetuned_mod error - cannot finetune without test function')

        if layer_order is None:
            layer_order = range(len(self.model_layers))
        if prec_reduct is None:
            prec_reduct = [None for _ in self.model_layers]

        best_focus_vals = [None for _ in self.model_layers]

        # savefile content
        file_content = []
        if os.path.isfile(savefile):
            with open(savefile) as f:
                reader = csv.reader(f)
                for row in reader:
                    file_content.append([])
                    for item in row:
                        # save file content to edit the file
                        file_content[-1].append(eval(item))

        # saving original weights and initial share
        self.set_reset()
        
        # sharing without modifications
        if shared_model_savefile:
            self.model.load_state_dict(torch.load(shared_model_savefile))
        else:
            self.share(layer_clusters, layer_order=layer_order, prec_reduct=prec_reduct, clust_alg=clust_alg)

        for layer in layer_order:
            if verbose:
                print(f'Processing layer {layer}')

            # layer init
            w_delta = []
            inertia = []
            acc = []
            already_tested = []
            if len(file_content) > layer:
                # load data from savefile
                already_tested = [x[0] for x in file_content[layer] if x[0] in mods_focus]
                w_delta = [x[1] for x in file_content[layer] if x[0] in mods_focus]
                inertia = [x[2] for x in file_content[layer] if x[0] in mods_focus]
                acc = [x[3] for x in file_content[layer] if x[0] in mods_focus]
            else:
                # add data to savefile
                file_content.append([])

            #num_in_cluster_mean = len(org_weight) / SHARE_CLUSTERS[LAYER]

            for focus_val in [x for x in mods_focus if x not in already_tested]:

                # reset layer weights
                self.model_layers[layer].reset_weight()

                # focus scoring
                self.model_layers[layer].share_weight(layer_clusters[layer], assign=True, unlock=False, prec_rtype=prec_reduct[layer], 
                    mod_focus=focus_val, mod_spread=mods_spread[layer], clust_alg=clust_alg)

                post_share_weights = torch.flatten(self.model_layers[layer].weight.cpu()).detach().numpy()
                w_delta.append(np.sum(np.abs(post_share_weights - self.model_layers[layer].reset_weight_vals['weights'])))
                inertia.append(self.model_layers[layer].clust_inertia)
                acc.append(self.test())

                if savefile is not None:
                    # adding data to savefile
                    file_content[layer].append([focus_val, w_delta[-1], inertia[-1], acc[-1]])
                    file_content[layer].sort(key= lambda x: x[0])
                    with open(savefile, 'w') as f:
                        for row in file_content:
                            write = csv.writer(f)
                            write.writerow(row)

            # saving the best focus value and setting the net to this val
            best_focus_vals[layer] = mods_focus[np.argmax(acc)]
            self.model_layers[layer].reset_weight()
            self.model_layers[layer].share_weight(layer_clusters[layer], assign=True, unlock=False, prec_rtype=prec_reduct[layer], 
                mod_focus=best_focus_vals[layer], mod_spread=mods_spread[layer], clust_alg = clust_alg)

            # plot
            if plot:
                acc = np.array(acc) * 100
                plt.rc('font', size=15)
                fig = plt.figure(figsize=(18, 4.5))
                plt.suptitle(f'Vrstva {layer} - {self.model_layers[layer].name}')

                for i in range(1, 3):
                    
                    ax1 = plt.subplot(1, 2, i)
                    color = 'tab:red'
                    ax1.set_xlabel('focus')
                    ax1.set_ylabel('Přesnost [%]', color=color)
                    ax1.plot(mods_focus, acc, color=color)
                    ax1.tick_params(axis='y', labelcolor=color)

                    ax2 = ax1.twinx()

                    color = 'tab:blue'
                    if i == 1:
                        ax2.set_ylabel('Clustering inertia', color=color)
                        ax2.plot(mods_focus, inertia, color=color)
                        cor = pearsonr(inertia, acc)
                        ax2.set_title(f'Pearsonův korelační koeficient: {cor[0]:1.4f}, p-hodnota: {cor[1]:1.4f}')
                    else:
                        ax2.set_ylabel('Delta vah', color=color)
                        ax2.plot(mods_focus, w_delta, color=color)
                        cor = pearsonr(w_delta, acc)
                        ax2.set_title(f'Pearsonův korelační koeficient: {cor[0]:1.4f}, p-hodnota: {cor[1]:1.4f}')
                    ax2.tick_params(axis='y', labelcolor=color)

                fig.tight_layout()
                os.makedirs(f'./graph', exist_ok=True)
                plt.savefig(f'./graph/ft_{self.model_layers[layer].name}.pdf')
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

    def get_layer_cluster_nums_perf(self, layer_i:int, layer_range:list, perf_fcs:list, pre_perf_fc=None, prec_rtype:str=None, 
            clust_alg:str = 'kmeans') -> list:
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
            clust_alg (str, optional): Specifies the clustering algorithm. 'kmeas' for classical K-means, 'minibatch-kmeas' for 
                minibatch kmeans and 'gmm' for gaussian mixture model. Defaults to 'kmeans'.

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
            self.model_layers[layer_i].share_weight(k, assign=True, unlock=False, prec_rtype=prec_rtype, clust_alg=clust_alg)
            if pre_perf_fc is not None:
                pre_perf_fc(self.model_layers[layer_i])
            k_score = [x(self.model_layers[layer_i]) for x in perf_fcs]
            output.append((k, k_score))

            # returning original data
            self.reset()

        print(f'layer {layer_i} done - {output}')

        return output

    def get_layers_cluster_nums_perfs(self, layer_ranges:list, perf_fcs, pre_perf_fc=None, prec_rtype:str=None, 
            clust_alg:str = 'kmeans', savefile:str = None) -> list:
        """Gets all the layers cluster performace for a given set of cluster numbers. After executing
        the model is returned to original state.

        If savefile is given, first it looks in the savefile and loads the prefs from it. Then it chooses the 
        ones that are in the wanted layer ranges. If any perfs are missing, they are computed and added to the file.

        Args:
            layer_ranges (list): is the list of lists of the number of clusters wanted to be tested. 
                Index of the list corresponds to an index of layer in the model.
            perf_fcs (function): are functions that will test the performace.
            pre_perf_fc (_type_, optional): is a function that is performed before the test.
                It recieves the model layer after sharing. Defaults to None.
            prec_rtype (str, optional): If not None, it defines the float precision reduction type 
                (for more information go to 'utils/float_prec_reducer/FloatPrecReducer.py'). Defaults to None.
            clust_alg (str, optional): Specifies the clustering algorithm. 'kmeas' for classical K-means, 'minibatch-kmeas' for 
                minibatch kmeans and 'gmm' for gaussian mixture model. Defaults to 'kmeans'.
            savefile (str, optional): is a savefile to be loaded from or saved to. Defaults to None.

        Raises:
            Exception: When the layer_ranges do not correspond to the model layers as described an exception is raised.

        Returns:
            list: list in format where index represents layer. Each layer have list of clusters number and corresponding
            list of performaces given by the perf_fcs.
        """
        if len(self.model_layers) != len(layer_ranges):
            raise Exception('WeightShare get_layers_cluster_nums_perfs error - given ranges do not correspond to model layers')

        layer_perfs:list[list] = []
        file_content:list[list] = []

        # load file contents
        if os.path.isfile(savefile):
            with open(savefile) as f:
                reader = csv.reader(f)
                for row in reader:
                    layer_perfs.append([])
                    file_content.append([])
                    for item in row:
                        perf = eval(item)
                        # save file content to edit the file
                        file_content[-1].append(perf)
                        # get only the perfs in rages
                        if perf[0] in layer_ranges[len(layer_perfs) - 1]:
                            layer_perfs[-1].append(perf)

        # save performaces if necessary
        if savefile is not None and not os.path.isfile(savefile):
            os.makedirs(os.path.dirname(savefile), exist_ok=True)

        for i, layer_range in enumerate(layer_ranges):
            if i < len(layer_perfs):
                # trim the computation range by the already computed perfs
                clust_nums = set([x[0] for x in layer_perfs[i]])
                layer_range = [x for x in list(layer_range) if x not in clust_nums]
                
            if len(layer_range) < 1:
                # if nothing to compute skip
                continue

            perf = self.get_layer_cluster_nums_perf(i, layer_range, perf_fcs, pre_perf_fc, prec_rtype=prec_rtype, clust_alg=clust_alg)
            perf = [[x[0], x[1][0]] for x in perf]

            # update the file  
            if i < len(file_content):
                # if added to loaded
                file_content[i] += perf
                file_content[i].sort(key = lambda x: x[0])
            else:
                file_content.append(perf)

            # save the udated file
            with open(savefile, 'w') as f:
                for row in file_content:
                    write = csv.writer(f)
                    write.writerow(row)

            # update the result perfs
            if i < len(layer_perfs):
                # if added to loaded
                layer_perfs[i] += perf
                layer_perfs[i].sort(key = lambda x: x[0])
            else:
                layer_perfs.append(perf)

        return layer_perfs

    def get_optimized_layer_ranges(self, layer_ranges:list, perf_fc, perf_lim:float=None, 
    max_num_range:int=None, savefile:str=None, pre_perf_fc=None, prec_rtype:str=None, clust_alg:str='kmeans') -> list:
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
            clust_alg (str, optional): Specifies the clustering algorithm. 'kmeas' for classical K-means, 'minibatch-kmeas' for 
                minibatch kmeans and 'gmm' for gaussian mixture model. Defaults to 'kmeans'.

        Returns:
            list: a list where an index corresponds to a given layer. Each index contains a list of optimized
            clusters numbers for a given layer.
        """
        
        perfs = []
        
        # compute performances if file not avalible
        
        perfs = self.get_layers_cluster_nums_perfs(layer_ranges, [perf_fc], pre_perf_fc, 
            prec_rtype=prec_rtype, clust_alg=clust_alg, savefile=savefile)

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

def get_all_spec_layer_names(model:torch.nn.Module, prev_name:str = '', pytorch_ws_layers:tuple = DEFAULT_WS_LAYERS) -> list:
    """Recusive funtion to find the names of layers of specified types.

    Args:
        model (torch.nn.Module): Is the model to be searched in.
        prev_name (str, optional): Is used in the recursion - tells the name of the parent to the child. Defaults to ''.
        pytorch_ws_layers (tuple, optional): The chosen layer types to be found. Defaults to DEFAULT_WS_LAYERS.

    Returns:
        list: List of names of the wanted layers.
    """

    result = []

    # recursively explore net for layer names and types
    if len(list(model.children())) > 0:
        # explore childrens
        for name, child in model.named_children():
            new_name = get_all_spec_layer_names(child, prev_name + '.' + name, pytorch_ws_layers=pytorch_ws_layers)
            if new_name is not None:
                result += new_name
    else:
        if isinstance(model, pytorch_ws_layers):
            # appends the names and removes the first char (dot)
            result.append(prev_name[1:])
    return result

def plot_weights(weights:np.array, cluster_centers:list = None, point_labels:list = None, name:str = '') -> None:
    """Plots histogram of given weights.

    Args:
        weights (np.array): Is the array of the model weights.
        cluster_centers (list, optional): Is list of cluster centers (given by kmeans). If none, cluster centers are not displayed.
            Defaults to None.
        point_labels (list, optional): Cluster labels for each point (given by kmeans). If none, weight labels are not displayed. 
            Defaults to None.
        name (str, optional): _description_. Defaults to ''.
    """

    plt.clf()
    plt.figure(figsize=(10,10))
    plt.rc('font', size=20)
    plt.title(f'Váhy vrstvy {name}')
    plt.xlabel(f'Váhy vrstvy {name}')
    plt.ylabel('Počet')

    sns.histplot(
        data = pd.DataFrame({
            'weights': weights,
            'Shluk': point_labels,
        }),
        x = 'weights',
        bins = 100,
        hue = 'Shluk' if point_labels is not None else None,
        multiple = 'stack'
    )

    # add cluster center lines, if possible
    if cluster_centers is not None:
        for center in cluster_centers:
            plt.axvline(center, color='r')
    
    plt.savefig('layer_weights_f0.pdf')

def plot_kmeans_space(numpy_weights_2D, name:str = '') -> None:
    """Plots the kmeans weight space before clustering.

    Args:
        numpy_weights_2D (np.array): Is the numpy array of model weights in 2D (second dimension used to modulate distances)
        name (str, optional): Is the layer name or any nyme printed in plot title. Defaults to ''.
    """

    plt.figure(figsize=(10,10))
    plt.rc('font', size=20)
    plt.scatter(numpy_weights_2D[0], numpy_weights_2D[1], marker="+", label='Váha')
    plt.axvline(x=np.mean(numpy_weights_2D[0]), color='r', label='Průměr')
    plt.axvline(x=0, color='g', label='Nula')
    plt.title(f'{name} shlukovací prostor pro K-Means')
    plt.xlabel('Původní váhy')
    plt.ylabel('Modifikovaný prostor - Tanh')
    plt.legend()
    plt.savefig('layer_kmeans_f0.pdf')
    plt.show()


def compute_cr(num_w_old:int, num_w_new:int, bits_w_old:int, bits_w_new:int, mapping_bits:int = None) -> float:
    """Computes compression rate of the layer by following expression:

        num_w_old: num of params in the weight tensor
        bits_w_old: amount of bits required to store the original value
        bits_w_new: amount of bits required to store the new weight value (key to table)
        bits_k: amount of bits required to store the share value (value of the key)
        key: amount of unique shared values 
        
                      num_w_old * bits_w_org
        ----------------------------------------------------
        num_w_old * bits_w_new + key * (bits_w_new + bits_k)

        Args:
            num_w_old (int): number of old parameters.
            num_w_new (int): number of new parameters (clusters).
            bits_w_old (int): amount of bits to store the original value.
            bits_w_new (int): amount of bits required to store the new value in ws table (precision reduction).
            mapping_bits (int, optional): Sets the number of bits wanted to be used
                as a key to the codebook (and in the weight matrix). Defaults to None.

        Raises:
            Exception: if the mapping_bits is seted and the number of elements is
            higher than the mapping_bits can map.

        Returns:
            float: the compression rate for the given parameters.
        """

    # computing number of bits to represent the key to map
    # in case there is only one unique value, the expression gives 0
    # which breaks the rest of the calculation, so this fix is needed
    # can be altered by the MIN_SAVED_BITS parameter
    bits_k = max(math.ceil(math.log(num_w_new) / math.log(2)), MIN_SAVED_BITS)

    # setting map bits
    if mapping_bits is not None:
        if mapping_bits < bits_k:
            raise Exception('compute_cr error - cannot be mapped to given bits')
        bits_k = mapping_bits
    
    # computing the weight-key matrix
    bits_map_key = num_w_old * bits_k

    # computing the weight mapping table 
    bits_map_vals = num_w_new * (bits_w_new + bits_k)

    return (num_w_old * bits_w_old) / (bits_map_key + bits_map_vals)
