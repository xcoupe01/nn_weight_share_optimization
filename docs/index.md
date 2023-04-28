# NN Weight Share Optimization

Documentation of diploma thesis codes for optimizing weight sharing in neural network, mainly using genetic algorithm.

Algorithms implemented yet:

- genetic algorithm search
- particle swarm optimization
- black hole algorithm
- random search

Implemented:

- layer-wise search space optimization
- float precision reduction
- dynamic fitness target
- finetuning via k-means space modulation
- model-wise compression

Nets tried:

- LeNet-5       - MNIST dataset
- Mobilenet_v2  - Lenette dataset

# Setup environment

To run this project, its necessary to have python interpeter (preferably in version 3.10). To ensure all necessary packages following command in the root of the project.

`python -m pip install -r requirements.txt`

Then everything is ready to run. Alternatively Anaconda can be used to run the project in virtual environment.

# Run

The file `net_compression.py` contains runnable CLI implementation of the Weight-Sharing optimization of MobileNet_v2.
Similarly `lenet_compression.py` works with Le-Net-5 and is pretty much the same in terms of setup.
To set up the compression optimization look at [compressor config](./compressors/compress_config.md).

To run the CLI program, its needed to have setuped python or python virtual enviroment, which is described in the previous section.
Use `python net_compression.py -h` to display the help.

The program has following arguments:

- `-comp {random,pso,genetic,blackhole}`, `--compressor {random,pso,genetic,blackhole}` -choose the compression algorithm
- `pop N`, `--num_population N` - set the population count
- `-its N`, `--num_iterations N` - set the iteration count
- `-up N`, `--upper_range N` - sets the upper range for compression
- `-lo N`, `--lower_range N` - sets the lower range for compression
- `-hp`, `--hide` - does not show the output plot
- `-sv`, `--save` - saves the output plot
- `-cfs CONFIG_SAVE`, `--config_save CONFIG_SAVE` - dumps current config in given file and ends (expect `.yaml`)
- `-cfl CONFIG_LOAD`, `--config_load CONFIG_LOAD` - loads config from given `.yaml` file
- `-sf SAVE_FOLDER`, `--save_folder SAVE_FOLDER` - Folder with the saves to be created, loaded, ect.

Similarly if paralel computing acceleration is needed, ther are accessible scripts to run just the range optimization
and finetuning part in files `net_range_opt.py` and `net_finetune.py`. Try `python [filename] -h` to see the optioal arguments.