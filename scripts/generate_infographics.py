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
        plt.savefig(savefig, dpi=100)
    else:
        plt.show()

def main(w=1e-3, n=6, N=5, seed=7, diag=1, ndiag=2, periodic=False, interpolating=False, savefig='matrix'):
    if interpolating and periodic:
        raise ValueError("Different boundary conditions")
    np.random.seed(seed)
    dim = n*N
    dimdim = (dim, dim)
    a = np.zeros(dimdim)
    for i in range(N):
        a[i*n:(i+1)*n, i*n:(i+1)*n] = np.random.random(n*n).reshape((n, n))  # block
        for j in range(n):
            a[i*n+j, i*n+j] += diag
        for di in range(ndiag):
            for j in range(n):
                if i > di:
                    a[i*n + j, (i-di-1)*n + j] = np.random.random()*w/(di+1)
                    a[(i-di-1)*n + j, i*n + j] = np.random.random()*w/(di+1)
                if periodic and i <= di:
                    a[(N-di-1+i)*n + j, i*n + j] = np.random.random()*w/(di+1)
                    a[i*n + j, (N-di-1+i)*n + j] = np.random.random()*w/(di+1)
                if interpolating and i <= di:
                    a[(N-di-1+i)*n + j, (N-2*ndiag-1+i)*n + j] = np.random.random()*w/(di+1)
                    a[i*n + j, (ndiag+di+1)*n + j] = np.random.random()*w/(di+1)

    plot(a, savefig + '.png')
    p, l, u = linalg.lu(a)
    lu = l + u - np.eye(dim)
    plot(lu, savefig + '_lu.png')

if __name__ == '__main__':
    import argh
    argh.dispatch_command(main)
