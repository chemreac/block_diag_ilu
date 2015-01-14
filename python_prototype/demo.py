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
from math import exp

from itertools import product

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
        sub_ = rnd((N-di-1)*n)*off_diag_factor
        sup_ = rnd((N-di-1)*n)*off_diag_factor
        for i in range(n*(N-di-1)):
            A[(di+1)*n + i, i] = sub_[i]
            A[i, (di+1)*n + i] = sup_[i]
        sub.append(sub_)
        sup.append(sup_)

    fLU = fast_FakeLU(A, n, ndiag)
    x_ref = lu_solve(lu_factor(A), b)
    x_ilu = fLU.solve(b)

    return A, b, x_ref, x_ilu, x_blk


def main(N=32, n=32, ndiag=1, main_diag_factor=1.0, off_diag_factor=1.0,
         base_seed=0, seed_range=1, fact_pow2_min=4, fact_pow2_max=18,
         plot=False, npows=0, scan_ndiag=False, savefig='None'):
    """
    Ax = b
    """

    npows = npows or fact_pow2_max - fact_pow2_min
    factors = np.linspace(fact_pow2_min, fact_pow2_max, npows)

    ilu_rmsd, blk_rmsd = [], []
    superiority = []

    if scan_ndiag:
        if seed_range != 1:
            raise ValueError("Cannot plot mulitple seeds and scan ndiag")
        ndiag_range = range(1, ndiag+1)
    else:
        ndiag_range = [ndiag]
    combos = product(ndiag_range, range(seed_range))

    #for seed in range(seed_range):
    #seed = base_seed
    #for ndiag in ndiag_range:

    nseries = 0
    for ndiag, seed in combos:
        nseries += 1
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

        if plot and seed_range == 1 and not scan_ndiag:
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

    if plot and (seed_range > 1 or scan_ndiag):
        import matplotlib.pyplot as plt

        plt.figure(figsize=(6, 14))

        if scan_ndiag:
            plot_kwargs = {}
        else:
            decay = exp(-((seed_range-1)/50.0))
            plot_kwargs = dict(alpha=1.0 - 0.9*(1-decay), linewidth=0.2 + 0.8*decay)

        ax = plt.subplot(3, 1, 1)
        ax.set_xscale('log', basex=10)
        ax.set_yscale('log', basey=10)
        clr = lambda idx, rgb: [1.0 - (nseries-idx)/float(nseries) if clridx==rgb else 0.0 for clridx in range(3)]
        for si, series in enumerate(ilu_rmsd):
            if scan_ndiag:
                c = clr(si, 2)  # blue
                lbl = str(ndiag_range[si])
            else:
                c = 'b'
                lbl = None
            plt.plot(2**factors, series, color=c, label=lbl, **plot_kwargs)
        plt.title("ILU")
        plt.xlabel("weight")
        plt.ylabel("RMSD")
        if scan_ndiag:
            plt.legend(loc='best')

        ax = plt.subplot(3, 1, 2)
        ax.set_xscale('log', basex=10)
        ax.set_yscale('log', basey=10)
        for si, series in enumerate(blk_rmsd):
            if scan_ndiag:
                c = clr(si, 1)  # green
                lbl = str(ndiag_range[si])
            else:
                c = 'g'
                lbl = None
            plt.plot(2**factors, series, color=c, label=lbl, **plot_kwargs)
        plt.title("Block RMSD")
        plt.xlabel("weight")
        plt.ylabel("RMSD")
        if scan_ndiag:
            plt.legend(loc='best')

        ax = plt.subplot(3, 1, 3)
        ax.set_xscale('log', basex=10)
        ax.set_yscale('log', basey=10)
        for si, series in enumerate(superiority):
            if scan_ndiag:
                c = clr(si, 0)  # red
                lbl = str(ndiag_range[si])
            else:
                c = 'k'
                lbl = None
            plt.plot(2**factors, series, color=c, label=lbl, **plot_kwargs)
        plt.title("BLOCK RMSD / ILU RMSD")
        plt.xlabel("weight")
        plt.ylabel("RMSD fraction")
        if scan_ndiag:
            plt.legend(loc='best')

        plt.tight_layout()

        if savefig == 'None':
            plt.show()
        else:
            plt.savefig(savefig, dpi=300)


if __name__ == '__main__':
    argh.dispatch_command(main)
