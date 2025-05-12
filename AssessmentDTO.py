import matplotlib.pyplot as plt

from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.lines import Line2D

class AssessmentDTO:

    epoch_list: list
    train_error_list: list
    test_error_list: list
    fig: Figure
    ax: Axes
    line1: Line2D
    line2: Line2D

    def __init__(self):
        self.epoch_list, self.train_error_list, self.test_error_list = [], [], []
        self.fig, self.ax, self.line1, self.line2 = self.get_plot_setup()

    def get_plot_setup(self):
        plt.ion()
        fig, ax = plt.subplots()
        line1, = ax.plot([], [], label='Test Error')
        line2, = ax.plot([], [], label='Eval Error')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Error')
        ax.legend()
        plt.show()

        return fig, ax, line1, line2