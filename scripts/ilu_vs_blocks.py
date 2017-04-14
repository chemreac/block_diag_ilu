#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This script aims to demonstrate the benefit (with respect to
precision) of the ILU implementation compared to e.g. ignoring sub and
super diagonals completely.
"""

from __future__ import (absolute_import, division, print_function)
from math import exp

from itertools import product, chain

import numpy as np
from scipy.linalg import lu_factor, lu_solve

from block_diag_ilu import ILU, Compressed_from_dense


def rms(x, axis=None):
    return np.sqrt(np.mean(x**2, axis=axis))


def rnd(dim):  # random number between -1 and 1
    return 2*(np.random.random(dim) - 0.5)


def get_test_system(N, n, ndiag, main_diag_factor, off_diag_factor, seed, periodic=False):
    np.random.seed(seed)
    A = np.zeros((N*n, N*n))
    b = rnd(N*n)
    x_blk = np.empty(N*n)

    for bi in range(N):
        cur_block = rnd((n, n))
        for i in range(n):
            cur_block[i, i] *= main_diag_factor
        slc = slice(n*bi, n*(bi+1))
        A[slc, slc] = cur_block
        x_blk[slc] = lu_solve(lu_factor(cur_block), b[slc])

    for di in range(ndiag):
        sub_ = rnd(N*n)*off_diag_factor
        sup_ = rnd(N*n)*off_diag_factor
        upto = n*(N-di-1)
        for i in range(upto):
            A[(di+1)*n + i, i] = sub_[i]
            A[i, (di+1)*n + i] = sup_[i]
        if periodic:
            for i in range((di+1)*n):
                A[(N-di-1)*n + i, i] = sub_[upto + i]
                A[i, (N-di-1)*n + i] = sup_[upto + i]

    x_ilu = ILU(Compressed_from_dense(A, N, n, ndiag, nsat=ndiag if periodic else 0)).solve(b)
    x_ref = lu_solve(lu_factor(A), b)
    return A, b, x_ref, x_ilu, x_blk


def main(N=32, n=32, ndiag=1, main_diag_factor=1.0, off_diag_factor=1.0,
         base_seed=0, seed_range=1, fact_pow2_min=4, fact_pow2_max=18,
         plot=False, npows=0, scan_ndiag=False, savefig='None', periodic=False, verbose=False):
    npows = npows or fact_pow2_max - fact_pow2_min
    factors = np.linspace(fact_pow2_min, fact_pow2_max, npows)

    superiority = []
    ilu_rmsd, blk_rmsd = [], []

    if scan_ndiag:
        if seed_range != 1:
            raise ValueError("Cannot plot mulitple seeds and scan ndiag")
        ndiag_range = range(1, ndiag+1)
    else:
        ndiag_range = [ndiag]
    combos = product(ndiag_range, range(seed_range))

    nseries = 0
    for ndiag, seed in combos:
        nseries += 1
        ilu_rmsd_local, blk_rmsd_local = [], []

        for diag_fact_pow in factors:
            A, b, x_ref, x_ilu, x_blk = get_test_system(
                N, n, ndiag,
                main_diag_factor*2**diag_fact_pow,
                off_diag_factor/2**diag_fact_pow,
                seed+base_seed, periodic=periodic)
            ilu_err = x_ilu - x_ref
            blk_err = x_blk - x_ref

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
                    plt.axis('equal')

                plt.show()
            else:
                minmax = [cb(*chain(ilu_rmsd_local,
                                    blk_rmsd_local)) for cb in (min, max)]
                plt.loglog(blk_rmsd_local, ilu_rmsd_local, 'd')
                plt.loglog(minmax, minmax)
                plt.xlabel('BLOCK RMSD')
                plt.ylabel('ILU RMSD')
                plt.legend()
                plt.show()

        ilu_rmsd.append(np.array(ilu_rmsd_local))
        blk_rmsd.append(np.array(blk_rmsd_local))
        superiority.append(np.array(blk_rmsd_local) / np.array(ilu_rmsd_local))
        if np.any(superiority[-1] < 1e-3):
            print('1000 x inferior:', seed)
    if verbose:
        print('ilu_rmsd_local=', ilu_rmsd)
        print('blk_rmsd_local=', blk_rmsd)

    if plot and (seed_range > 1 or scan_ndiag):
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(3, 1, figsize=(6, 14))

        if scan_ndiag:
            plot_kwargs = {}
        else:
            decay = exp(-((seed_range-1)/50.0))
            plot_kwargs = dict(alpha=1.0 - 0.9*(1-decay), linewidth=0.2 + 0.8*decay)

        axes[0].set_xscale('log', basex=10)
        axes[0].set_yscale('log', basey=10)
        axes[0].axis('equal')
        clr = lambda idx, rgb: [1.0 - (nseries-idx)/float(nseries) if clridx==rgb else 0.0 for clridx in range(3)]
        for si, series in enumerate(ilu_rmsd):
            if scan_ndiag:
                c = clr(si, 2)  # blue
                lbl = str(ndiag_range[si])
            else:
                c = 'b'
                lbl = None
            axes[0].plot(2**factors, series, color=c, label=lbl, **plot_kwargs)
        axes[0].set_title("ILU")
        axes[0].set_xlabel("weight")
        axes[0].set_ylabel("RMSD")
        if scan_ndiag:
            axes[0].legend(loc='best')

        axes[1].set_xscale('log', basex=10)
        axes[1].set_yscale('log', basey=10)
        axes[1].axis('equal')
        for si, series in enumerate(blk_rmsd):
            if scan_ndiag:
                c = clr(si, 1)  # green
                lbl = str(ndiag_range[si])
            else:
                c = 'g'
                lbl = None
            axes[1].plot(2**factors, series, color=c, label=lbl, **plot_kwargs)
        axes[1].set_title("Block RMSD")
        axes[1].set_xlabel("weight")
        axes[1].set_ylabel("RMSD")
        if scan_ndiag:
            axes[1].legend(loc='best')

        axes[2].set_xscale('log', basex=10)
        axes[2].set_yscale('log', basey=10)
        axes[2].axis('equal')
        for si, series in enumerate(superiority):
            if scan_ndiag:
                c = clr(si, 0)  # red
                lbl = str(ndiag_range[si])
            else:
                c = 'k'
                lbl = None
            axes[2].plot(2**factors, series, color=c, label=lbl, **plot_kwargs)
        axes[2].set_title("BLOCK RMSD / ILU RMSD")
        axes[2].set_xlabel("weight")
        axes[2].set_ylabel("RMSD fraction")
        if scan_ndiag:
            axes[2].legend(loc='best')

        fig.tight_layout()

        if savefig == 'None':
            fig.show()
        else:
            fig.savefig(savefig, dpi=300)


if __name__ == '__main__':
    import argh
    argh.dispatch_command(main)
