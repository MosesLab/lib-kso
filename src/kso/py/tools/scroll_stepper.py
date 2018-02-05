from __future__ import print_function

import numpy as np
import matplotlib.pyplot as plt


class IndexTracker(object):
    def __init__(self, ax, X, ind):
        self.ax = ax
        ax.set_title('use scroll wheel to navigate images')


        self.X = X
        self.slices, self.rows, self.cols = X.shape
        self.ind = ind

        self.im = ax.imshow(self.X[self.ind, :, :], cmap='gray', vmin=0.0, vmax = 1000.0)
        self.update()

    def onscroll(self, event):
        if event.button == 'up':
            self.ind = (self.ind + 1) % self.slices
        else:
            self.ind = (self.ind - 1) % self.slices
        self.update()

    def update(self):
        self.ax.clear()
        self.ax.imshow(self.X[self.ind, :, :], cmap='gray', vmin=0.0, vmax = 1000.0)
        self.im.axes.figure.canvas.draw()

