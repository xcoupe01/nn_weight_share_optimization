import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

SNS_PALETTE = 'tab10'
SNS_FIGSIZE = (18, 10)
pd.options.mode.chained_assignment = None

def plot_alncl(dataframe:pd.DataFrame) -> None:
    #accuracy loss, num clusters, layer plot

    sns.set(rc={'figure.figsize': SNS_FIGSIZE})
    sns.scatterplot(data=dataframe, x='num clusters', y='accuracy loss', hue='layer', palette=SNS_PALETTE)

def plot_alcr(dataframe:pd.DataFrame) -> None:
    # accuracy loss, compression rate plot

    ACCEPT_ACCURACY_PER = 1

    dataframe = dataframe[dataframe['accuracy_loss'] < 0.05]
    dataframe['acceptable'] = np.where(dataframe['accuracy_loss'] < ACCEPT_ACCURACY_PER * 0.01, 'passed', 'failed')
    dataframe['accuracy loss [%]'] = dataframe['accuracy_loss'] * 100
    sns.set(rc={'figure.figsize': SNS_FIGSIZE})
    plot = sns.scatterplot(data=dataframe, x='compression', y='accuracy loss [%]', hue='acceptable', palette=SNS_PALETTE)
    plot.axhline(ACCEPT_ACCURACY_PER, color='red', alpha=0.2)


if __name__ == '__main__':
    #plot_AL_SV(pd.read_csv('./results/test_share.csv'))
    #plt.show()

    plot_alcr(pd.read_csv('./results/test_GA_save.csv'))
    plt.show()