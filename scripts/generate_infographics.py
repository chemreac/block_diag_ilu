#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import scipy.linalg as linalg
import matplotlib.pyplot as plt

def plot(mat, savefig=None):
    fig = plt.figure()
    img = plt.imshow(np.log10(np.abs(mat)), interpolation='None', cmap='viridis')
    plt.gca().set_xticks([])
    plt.gca().set_yticks([])
    cb = fig.colorbar(img, label=r'$\log_{10}\left(|y|\right)$')
    cb.ax.yaxis.set_label_position('left')
    plt.tight_layout()
    if savefig is not None:
        plt.savefig(savefig, dpi=60)
    else:
        plt.show()

def main(w=1e-3, n=6, N=5, seed=7, diag=1):
    np.random.seed(seed)
    dim = n*N
    dimdim = (dim, dim)
    a = np.zeros(dimdim)
    for i in range(N):
        a[i*n:(i+1)*n, i*n:(i+1)*n] = np.random.random(n*n).reshape((n, n))
        for j in range(n):
            a[i*n+j, i*n+j] += diag
        if i > 0:
            for j in range(n):
                a[i*n + j, (i-1)*n + j] = np.random.random()*w
                a[(i-1)*n + j, i*n + j] = np.random.random()*w
        if i > 1:
            for j in range(n):
                a[i*n + j, (i-2)*n + j] = np.random.random()*w/2
                a[(i-2)*n + j, i*n + j] = np.random.random()*w/2
    plot(a, 'matrix.png')
    p, l, u = linalg.lu(a)
    lu = l + u - np.eye(dim)
    plot(lu, 'lu.png')

if __name__ == '__main__':
    main()
