# MobileNet_v2 notebooks

This section describes the notebooks which processes Le-Net-5 experiments.
They can be foun in `notebooks/imagenet_nets_compression` folder. 
Theoreticaly can work with any CNN clasifying imagenet from Pytorch hub.
The folder contains folowing items:

- `imagenet_genetic_compression.ipynb` - 
- `mob_net_ws_test.ipynb` - demnostration of Weight-Sharing on MobileNet.
- `opt_range_combiner.ipynb` - script to concatenate range optimizer outputs if computed in parralel.
- `result_study` folder:
    - `acc_test.ipynb` - accuracy testing of compressed MobileNet. 
    - `finetuning.ipynb` - demonstration of finetuning (can load finetuning savefiles).
    - `range_opt_viz.ipynb` - vizualization of range optimization.
    - `searched_solutions.ipynb` - vizualization of optimization run.