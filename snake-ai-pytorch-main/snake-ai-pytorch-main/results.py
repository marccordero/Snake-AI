import matplotlib.pyplot as plt
import pandas as pd
from IPython import display
from constants import EPISODES, MAX_MEMORY, BATCH_SIZE, LR
import os

plt.ion()

def plot(scores, mean_scores, modelo_tipo, save=False):
    display.clear_output(wait=True)
    display.display(plt.gcf())
    plt.clf()
    plt.title('Entrenamiento...')
    plt.xlabel('Numero de juegos')
    plt.ylabel('Puntuacion')
    plt.plot(scores)
    plt.plot(mean_scores)
    plt.ylim(ymin=0)
    plt.text(len(scores)-1, scores[-1], str(scores[-1]))
    plt.text(len(mean_scores)-1, mean_scores[-1], str(mean_scores[-1]))
    plt.show(block=False)
    plt.pause(.1)
    if save:
        plt.savefig(f'grafica_{modelo_tipo}_{EPISODES}_{BATCH_SIZE}_{LR}.png')

def save_results(scores, mean_scores, record, modelo_tipo):
    df = pd.DataFrame({'scores': scores, 'mean_scores': mean_scores, 'record': record})
    df.to_csv(f'resultados_{modelo_tipo}_{EPISODES}_{BATCH_SIZE}_{LR}.csv', index=False)
