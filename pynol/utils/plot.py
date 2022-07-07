from typing import Optional
import numpy as np
import matplotlib
import matplotlib.pyplot as plt


def plot(loss: np.ndarray,
         labels: list,
         cum: bool = True,
         title: Optional[str] = None,
         file_path: Optional[str] = None,
         x_label: Optional[str] = 'Iteration',
         y_label: Optional[str] = 'Cumulative Loss'):
    """Visualize the results of multiple learners.:

    Args:
        loss (numpy.ndarray): Losses of multiple learners.
        labels (list): labels of learners.
        cum (bool): Show the cumulative loss or instantaneous loss.
        title (str, optional): Title of the figure.
        file_path (str, optional): File path to save the results.
        x_lable (str, optional): Label of :math:`x` axis.
        y_lable (str, optional): Label of :math:`y` axis.
    """
    plt.figure()
    matplotlib.rcParams['font.family'] = "sans-serif"
    matplotlib.rcParams['font.sans-serif'] = "Arial"
    assert loss.ndim == 3 or loss.ndim == 2
    assert loss.shape[0] == len(labels)
    if loss.ndim == 3:
        xaxis = np.arange(0, loss.shape[2])
        if cum is True:
            loss = np.cumsum(loss, axis=2)
        loss_mean, loss_std = np.mean(loss, axis=1), np.std(loss, axis=1)
    else:
        xaxis = np.arange(0, loss.shape[1])
        if cum is True:
            loss = np.cumsum(loss, axis=1)
        loss_mean, loss_std = loss, np.zeros_like(loss)

    plt.grid(linestyle=':', linewidth=0.5)
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    for i in range(len(loss_mean)):
        plt.plot(xaxis, loss_mean[i], label=labels[i])
        plt.fill_between(
            xaxis,
            loss_mean[i] - loss_std[i],
            loss_mean[i] + loss_std[i],
            alpha=0.15)
    plt.legend(loc='upper left')
    if file_path is not None:
        plt.savefig(file_path)
    plt.show()
