from __future__ import print_function

import numpy as np
import matplotlib.pyplot as plt


class IndexTracker(object):
    def __init__(self, ax, X, ind, v_min=None, v_max=None):
        self.ax = ax
        ax.set_title('use scroll wheel to navigate images')


        self.X = X
        self.slices, self.rows, self.cols = X.shape
        self.ind = ind

        self.vmin = v_min
        self.vmax = v_max

        self.im = ax.imshow(self.X[self.ind, :, :], vmin=self.vmin, vmax=self.vmax)
        self.update()

    def onscroll(self, event):
        if event.button == 'up':
            self.ind = (self.ind + 1) % self.slices
        else:
            self.ind = (self.ind - 1) % self.slices
        self.update()

    def update(self):
        self.ax.clear()
        self.ax.imshow(self.X[self.ind, :, :], vmin=self.vmin, vmax=self.vmax)
        self.im.axes.figure.canvas.draw()

