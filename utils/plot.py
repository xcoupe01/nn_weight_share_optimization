import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

from utils.fitness_controller import FitnessController

SNS_PALETTE = 'tab10'
SNS_FIGSIZE = (18, 10)
pd.options.mode.chained_assignment = None

def plot_alncl(dataframe:pd.DataFrame) -> None:
    #accuracy loss, num clusters, layer plot

    sns.set(rc={'figure.figsize': SNS_FIGSIZE})
    sns.scatterplot(data=dataframe, x='num clusters', y='accuracy loss', hue='layer', palette=SNS_PALETTE)

def plot_alcr(dataframe:pd.DataFrame, acceptable_division:float=None, target:list[float] = None, pareto:bool = False) -> None:
    # accuracy loss, compression rate plot

    sns.set(rc={'figure.figsize': SNS_FIGSIZE})
    sns.set(font_scale=2)

    if pareto:
        pareto_front = pareto_from_df(dataframe)
        pareto_front['accuracy_loss'] = pareto_front['accuracy_loss'] * 100
        sns.lineplot(data=pareto_front, x='compression', y='accuracy_loss', linestyle='dashed', color='g')

    dataframe = dataframe[dataframe['accuracy_loss'] < 0.05]
    before_loss = dataframe.loc[0]['accuracy'] + dataframe.loc[0]['accuracy_loss'] 
    dataframe['acceptable'] = np.where(dataframe['accuracy_loss'] < acceptable_division, 'passed', 'failed')
    dataframe['accuracy loss [%]'] = dataframe['accuracy_loss'] * 100

    dataframe = dataframe[['accuracy loss [%]', 'compression', 'acceptable']]
    dataframe['type'] = 'Value'

    if target is not None:
        dataframe.loc[len(dataframe.index)] = [(before_loss - target[0]) * 100, target[1], 'passed', 'Target']

    plot = None
    if acceptable_division is not None:
        plot = sns.scatterplot(data=dataframe, x='compression', y='accuracy loss [%]', hue='acceptable', style='type', palette=SNS_PALETTE)
        plot.axhline(acceptable_division * 100, color='red', alpha=0.2)
    else:
        plot = sns.scatterplot(data=dataframe, x='compression', y='accuracy loss [%]', hue='type', palette=SNS_PALETTE)
    
    #plot.set_xlim(5, 20)
    #plot.set_ylim(-1, 4)

def plot_optimalization_progress(files:dict, fit_cont:FitnessController):

    data = {}

    iter_col_name = {
        'PSO': 'time',
        'BH': 'time',
        'GA': 'generation',
        'RND': None,
    }

    for key in files.keys():
        if key not in data.keys():
            data[key] = []

        for file in files[key]:
            df = pd.read_csv(file)
            fit_cont.fit_from_df(df)
            if iter_col_name[key] is not None:
                df = df.groupby(iter_col_name[key]).max().reset_index()
                data[key].append(df[['fitness', 'compression', 'accuracy', iter_col_name[key]]])
            else:
                df = df.groupby(df.index // 20).max().reset_index()
                df['index'] = df.index
                data[key].append(df[['fitness', 'compression', 'accuracy', 'index']])
        data[key] = pd.concat(data[key], axis=0, ignore_index=True)
    
    plt.figure(figsize=(15, 10))
    plt.rc('font', size=10)

    max_val = 0
    for key in data.keys():
        candidate = data[key]['fitness'].max()
        max_val =  candidate if candidate > max_val else max_val

    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = prop_cycle.by_key()['color']

    for index, key in enumerate(data.keys()):
        optim_group = None

        if iter_col_name[key] is not None:
            optim_group = data[key].groupby(iter_col_name[key])
        else:
            optim_group = data[key].groupby('index')

        optim_max = optim_group.max().reset_index()
        optim_mean = optim_group.mean().reset_index() # could be median
        optim_min = optim_group.min().reset_index()

        plt.subplot(1, len(data), index+1)
        plt.plot(range(len(optim_mean['fitness'])), optim_mean['fitness'], color=colors[index])
        plt.fill_between(range(len(optim_mean['fitness'])), optim_max['fitness'], optim_min['fitness'], color=colors[index], alpha=0.4)
        plt.ylim([0, max_val])
        plt.title(key)
        plt.xlabel('iterace')
        plt.ylabel('fitness')
    
    plt.tight_layout()

def pareto_from_df(df:pd.DataFrame) -> pd.DataFrame:
    """Generates pareto front from a given dataframe.

    Args:
        df (pd.DataFrame): is the dataframe to generate the pareto front from.

    Returns:
        pd.DataFrame: the pareto front datapoints.
    """
    pareto_front:pd.DataFrame = pd.DataFrame(columns=list(df.columns))

    for index, row in df.sort_values(by='accuracy', ascending=False).iterrows():
        if len(pareto_front) == 0:
            pareto_front = pareto_front.append(row, ignore_index=True)
            pareto_front.reset_index()
            continue

        if row['compression'] > pareto_front.iloc[-1]['compression']:
            pareto_front = pareto_front.append(row, ignore_index=True)
            pareto_front.reset_index()

    return pareto_front

if __name__ == '__main__':
    #plot_AL_SV(pd.read_csv('./results/test_share.csv'))
    #plt.show()

    plot_alcr(pd.read_csv('./results/test_GA_save.csv'))
    plt.show()