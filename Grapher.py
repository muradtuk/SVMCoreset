import matplotlib
matplotlib.use('TKAgg')
import matplotlib.pyplot as plt
import os
import numpy as np
import datetime, seaborn
from scipy import interpolate
import re
import platform
import subprocess
import Util
import scipy.io as sio


class Grapher(object):
    def __init__(self, directoryName):
        self.saveFigWidth = 20
        self.saveFigHeight = 13
        self.fontsize = 50
        self.legendfontsize = self.fontsize
        self.labelpad = 10
        self.linewidth = 9
        self.colors = ['blue', '#20B2AA', 'red', 'cyan', 'green', 'magenta']
        self.color_matching = Util.color_matching
        self.OPEN_FIG = True

        plt.rcParams.update({'font.size': self.fontsize})
        plt.rcParams['xtick.major.pad'] = '{}'.format(self.labelpad * 3)
        plt.rcParams['ytick.major.pad'] = '{}'.format(self.labelpad)
        plt.rcParams['xtick.labelsize'] = self.legendfontsize
        plt.rcParams['ytick.labelsize'] = self.legendfontsize

        seaborn.set_style("whitegrid")
        self.directoryName = directoryName

    def SaveFigure(self, fileName):
        figure = plt.gcf()
        figure.set_size_inches(self.saveFigWidth, self.saveFigHeight)
        plt.savefig(fileName, bbox_inches='tight')

    def errorFill(self, x, y, yerr, color=None, alpha_fill=0.08, ax=None, linewidth=7, timeGraph=False):
        ax = ax if ax is not None else plt.gca()
        if color is None:
            color = ax._get_lines.color_cycle.next()
        if np.isscalar(yerr) or len(yerr) == len(y):
            ymin = y - yerr
            ymax = y + yerr
        elif len(yerr) == 2:
            ymin, ymax = yerr

        if timeGraph:
            y = list(np.maximum(np.ones(np.size(y)), y))
            ax.semilogy(x, y, '-', linewidth=self.linewidth, color=color, basey=2)
        else:
            ax.plot(x, y, color=color, marker='o', markersize='20', linewidth=linewidth)

        yerr = yerr
        ax.fill_between(x, y - yerr, y + yerr, color=color, alpha=alpha_fill, capstyle='round', antialiased=True,
                        linewidth=0)

    def Graph(self, samples, matrix, legend, bareTitle, title, xlabel, ylabel, titleAdditionalTxt='', errBar=None):
        multipleY = matrix.tolist()
        samples = samples.tolist()

        timeGraph = 'time' in titleAdditionalTxt

        if not timeGraph:
            errBar = errBar.tolist()
        yMin = min([min(l) for l in multipleY])
        xMinMax = min([max(l) for l in samples[:-1]])
        xMaxMin = max([min(l) for l in samples[:-1]])

        xMin = min([min(l) for l in samples[:-1]])

        xMax = max([max(l) for l in samples[:-1]])
        ind = np.argmax([max(l) for l in samples[:-1]])

        ACCURATE_SAMPLE_SIZE = True
        if ACCURATE_SAMPLE_SIZE and titleAdditionalTxt is '':
            numPointsAboveMinMax = sum([x > xMinMax for x in samples[ind]])
            for i in range(len(samples[ind])):
                if samples[ind][i] > xMinMax and numPointsAboveMinMax > 1:
                    multipleY[ind] = multipleY[ind][:-1]
                    if not timeGraph:
                        errBar[ind] = errBar[ind][:-1]
                    numPointsAboveMinMax = numPointsAboveMinMax - 1

            xInterp = samples[ind][0:len(multipleY[ind])]
            yInterp = multipleY[ind]
            if not timeGraph:
                yErrInterp = errBar[ind]
            if len(yInterp) >= 2:
                f = interpolate.interp1d(xInterp, yInterp)
                samples[ind].append(xMinMax)
                multipleY[ind].append(f(xMinMax))
                if not timeGraph:
                    g = interpolate.interp1d(xInterp, yErrInterp)
                    errBar[ind].append(g(xMinMax))

        else:
            yMin = -1e-2

        yMin = min([min(l) for l in multipleY])
        yMax = max([max(l) for l in multipleY])
        yMin = -np.abs(yMax)/250
        if xMinMax > xMaxMin:
            XMIN = xMaxMin
            XMAX = xMinMax
        else:
            XMAX = xMaxMin
            XMIN = xMinMax

        plt.xlabel(xlabel, fontsize=self.fontsize)
        plt.ylabel(ylabel, fontsize=self.fontsize)

        if 'Our Coreset' in legend:
            i = legend.index('Our Coreset')
            j = self.colors.index('#20B2AA')
            self.colors[i], self.colors[j] = self.colors[j], self.colors[i]

        for i, y in enumerate(multipleY):
            xSorted, ySorted, yErrSorted = (list(t) for t in zip(*sorted(zip(samples[i], y, errBar[i]))))

            color = self.color_matching[legend[i]]
            if i == 3:
                plt.plot([XMIN, XMAX], [ySorted[0], ySorted[0]], color=color, linewidth=self.linewidth, linestyle='dashed')
            else:
                self.errorFill(np.array(xSorted), np.array(ySorted), np.array(yErrSorted), color, 0.1, plt,
                               self.linewidth)
            THRESH = 10.0
            yMin = min(yMin, min([y[0] - y[1]/THRESH for y in [x for x in zip(ySorted, yErrSorted)]]))
            yMax = max(yMax, max([y[0] + y[1]/THRESH for y in [x for x in zip(ySorted, yErrSorted)]]))

        yMin = yMin if (yMin < 0 and yMin < -0.05) else 0

        plt.ylim(yMin, yMax)

        plt.legend(legend, loc='best', fontsize=self.fontsize)
        fileName = title.replace('.mat', '') + titleAdditionalTxt
        plt.title(bareTitle.replace('RawError', '').replace('-error', '').replace('_', ' '), fontsize=self.fontsize)
        timeStamp = '{:%Y-%m-%d-%H-%M-%S}'.format(datetime.datetime.now())
        fileName = "{}/{}-{}.png".format(self.directoryName, fileName, timeStamp)

        if titleAdditionalTxt is not '':
            titleAdditionalTxt = '-' + titleAdditionalTxt
        fileName = "{}{}.pdf".format(self.directoryName, bareTitle +
                                    titleAdditionalTxt)
        self.SaveFigure(fileName)
        plt.clf()