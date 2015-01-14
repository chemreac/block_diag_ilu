#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
This script aims to demonstrate the
benefit (with respect to precision)
of the ILU implementation compared
to e.g. ignoring sub and super diagonals
completely.

Note that the python wrapper is quite inefficient and
hence not suitable for benchmarking.
"""

from __future__ import (absolute_import, division,
                        print_function, unicode_literals)
from future.builtins import *

import argh
import numpy as np
from scipy.linalg import lu_factor, lu_solve
from scipy.stats import norm

from fakelu import fast_FakeLU

def rms(x, axis=None):
    return np.sqrt(np.mean(x**2, axis=axis))

def rnd(dim):
    return 2*(np.random.random(dim) - 0.5)

def get_test_system(N, n, ndiag, main_diag_factor, off_diag_factor, seed):
    np.random.seed(seed)
    A = np.zeros((N*n, N*n))
    blocks, sub, sup, x_blk = [], [], [], []
    b = rnd(N*n)

    for bi in range(N):
        cur_block = rnd((n, n))
        for i in range(n):
            cur_block[i, i] *= main_diag_factor
        blocks.append(cur_block)
        slc = slice(n*bi, n*(bi+1))
        A[slc, slc] = cur_block
        x_blk.append(lu_solve(lu_factor(cur_block), b[slc]))

    for di in range(ndiag):
        sub_ = rnd((N-di-1)*n)*off_diag_factor**(di+1)
        sup_ = rnd((N-di-1)*n)*off_diag_factor**(di+1)
        for i in range(n*(N-1)):
            A[(di+1)*n + i, i] = sub_[i]
            A[i, (di+1)*n + i] = sup_[i]
        sub.append(sub_)
        sup.append(sup_)

    fLU = fast_FakeLU(A, n, ndiag)
    x_ref = lu_solve(lu_factor(A), b)
    x_ilu = fLU.solve(b)

    return A, b, x_ref, x_ilu, x_blk


def main(N=5, n=5, ndiag=1, main_diag_factor=1.0, off_diag_factor=1.0,
         base_seed=0, seed_range=1, fact_pow2_min=0, fact_pow2_max=1, plot=False,
         npows=0, savefig='None'):
    """
    Ax = b
    """

    npows = npows or fact_pow2_max - fact_pow2_min
    factors = np.linspace(fact_pow2_min, fact_pow2_max, npows)

    ilu_rmsd, blk_rmsd = [], []
    superiority = []

    for seed in range(seed_range):

        ilu_rmsd_local, blk_rmsd_local = [], []

        for diag_fact_pow in factors:
            A, b, x_ref, x_ilu, x_blk = get_test_system(
                N, n, ndiag,
                main_diag_factor*2**diag_fact_pow,
                off_diag_factor/2**diag_fact_pow,
                seed+base_seed)
            ilu_err = x_ilu - x_ref
            blk_err = np.array(x_blk).flatten() - x_ref

            ilu_rmsd_local.append(rms(ilu_err))
            blk_rmsd_local.append(rms(blk_err))

        if plot and seed_range == 1:
            import matplotlib.pyplot as plt

            if npows == 1:
                for idx in (1, 2):
                    plt.subplot(3, 1, idx)
                    plt.plot(ilu_err, label='ILU error')

                for idx in (1, 3):
                    plt.subplot(3, 1, idx)
                    plt.plot(blk_err, label='block error')

                for idx in (1, 2, 3):
                    plt.subplot(3, 1, idx)
                    plt.legend()

                plt.show()
            else:
                plt.semilogy(ilu_rmsd, label="ILU RMSD")
                plt.semilogy(blk_rmsd, label="Block RMSD")
                plt.legend()
                plt.show()
        ilu_rmsd.append(np.array(ilu_rmsd_local))
        blk_rmsd.append(np.array(blk_rmsd_local))
        superiority.append(np.array(blk_rmsd_local) / np.array(ilu_rmsd_local))
        if np.any(superiority[-1] < 1e-3):
            print('1000 x inferior:', seed)

    if plot and seed_range > 1:
        import matplotlib.pyplot as plt

        plt.figure(figsize=(6, 14))
        plot_kwargs = dict(alpha=0.1, linewidth=0.2)

        ax = plt.subplot(3, 1, 1)
        ax.set_xscale('log', basex=10)
        ax.set_yscale('log', basey=10)
        for series in ilu_rmsd:
            plt.plot(2**factors, series, 'b', **plot_kwargs)
        plt.title("ILU")
        plt.xlabel("weight")
        plt.ylabel("RMSD")

        ax = plt.subplot(3, 1, 2)
        ax.set_xscale('log', basex=10)
        ax.set_yscale('log', basey=10)
        for series in blk_rmsd:
            plt.plot(2**factors, series, 'g', **plot_kwargs)
        plt.title("Block RMSD")
        plt.xlabel("weight")
        plt.ylabel("RMSD")

        ax = plt.subplot(3, 1, 3)
        ax.set_xscale('log', basex=10)
        ax.set_yscale('log', basey=10)
        for series in superiority:
            plt.plot(2**factors, series, 'k', **plot_kwargs)
        plt.title("BLOCK RMSD / ILU RMSD")
        plt.xlabel("weight")
        plt.ylabel("RMSD fraction")

        plt.tight_layout()

        if savefig == 'None':
            plt.show()
        else:
            plt.savefig(savefig, dpi=300)


if __name__ == '__main__':
    argh.dispatch_command(main)
