"""
Code by @GYQ-AI
Code references: https://github.com/sksq96/pytorch-summary https://github.com/graykode/modelsummary
Inspired by https://github.com/sksq96/pytorch-summary
Motivation: Since the 'torchsummary' tool merely outputs the structural summary for a model, I built a statistical tool
called 'torchdiagram' to tally the module distribution for a model and visualize it intuitively in the pie style.
"""
from .torchsummary import *
import torch
import matplotlib.pyplot as plt
import numpy as np



def pie_diagram(layer_class, layer_count, title='pie chart for module distribution'):
    layer_idx = list(range(len(layer_class)))
    layer_legend = []
    s = np.sum(layer_count)
    for i in layer_idx:
        layer_legend.append(str(i) + '. ' + layer_class[i] + ': ' + str(
            round(layer_count[i] / s * 100, 2)) + '% ' + '(' + str(layer_count[i]) + ')')
    fig = plt.figure(figsize=(7.5, 5))
    ax1 = fig.add_subplot(121)
    pie = ax1.pie(layer_count, labels=layer_idx, autopct=None, startangle=0)
    ax1.axis('equal')
    ax2 = fig.add_subplot(122)
    ax2.axis("off")
    ax2.legend(pie[0], layer_legend, loc='center')
    ax1.set_title(title)
    plt.show()
    plt.savefig('piechart.jpg')


def summary_draw(summary_info):
    layer_class = []
    layer_count = []
    for layer in summary_info:
        layer_name = str(layer).split('-')[0]
        if layer_name not in layer_class:
            layer_class.append(layer_name)
            layer_count.append(0)
        layer_idx = layer_class.index(layer_name)
        layer_count[layer_idx] = layer_count[layer_idx] + 1
    last_pos = len(layer_class) - 1
    layer_class.pop(last_pos)
    layer_count.pop(last_pos)
    pie_diagram(layer_class, layer_count)
    print("layer_class: ", layer_class)
    print("layer_count: ", layer_count)


def visualize(model, input_size, batch_size=-1, device=torch.device('cuda:0'), dtypes=None):
    summary_info, result, params_info = summary_string(
        model, input_size, batch_size, device, dtypes)
    print(result)
    summary_draw(summary_info)

    return params_info

