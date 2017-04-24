from __future__ import (absolute_import, division,
                        print_function, unicode_literals)
import os
import numpy as np
import scipy.linalg
from warnings import warn

# Here comes the fast implementation:
try:
    from _block_diag_ilu import PyILU
except ImportError:
    if os.environ.get("USE_FAST_FAKELU", "0") == "1":
        # You better not use fast_FakeLU()...
        raise
    class PyILU:
        pass

class ILU:

    def __init__(self, A, sub, sup, blockw, ndiag=0):
        fA = np.empty((blockw, A.shape[0]), order='F')
        ssub = np.empty_like(sub)
        ssup = np.empty_like(sup)
        nblocks = A.shape[0]//blockw
        for bi in range(nblocks):
            slc = slice(bi*blockw, (bi+1)*blockw)
            fA[0:blockw, slc] = A[slc, slc]
        idx = 0
        for di in range(ndiag):
            for bi in range(nblocks-di-1):
                for ci in range(blockw):
                    ssub[idx] = A[blockw*(bi+di+1)+ci, blockw*bi + ci]
                    ssup[idx] = A[blockw*bi + ci, blockw*(bi+di+1) + ci]
                    idx += 1
        self._pyilu = PyILU(fA, ssub, ssup, blockw, ndiag)

    def solve(self, b):
        return self._pyilu.solve(b)

    @property
    def sub(self):
        sub = []
        for di in range(self._pyilu.ndiag):
            ssub = []
            for bi in range(self._pyilu.nblocks - di - 1):
                for ci in range(self._pyilu.blockw):
                    ssub.append(self._pyilu.sub_get(di, bi, ci))
            sub.append(ssub)
        return sub

    @property
    def sup(self):
        sup = []
        for di in range(self._pyilu.ndiag):
            ssup = []
            for bi in range(self._pyilu.nblocks - di - 1):
                for ci in range(self._pyilu.blockw):
                    ssup.append(self._pyilu.sup_get(di, bi, ci))
            sup.append(ssup)
        return sup

    @property
    def rowbycol(self):
        nblocks = self._pyilu.nblocks
        blockw = self._pyilu.blockw
        rbc = []
        for bi in range(nblocks):
            l = []
            for ci in range(blockw):
                l.append(self._pyilu.rowbycol_get(bi*blockw+ci))
            rbc.append(l)
        return rbc

    @property
    def colbyrow(self):
        nblocks = self._pyilu.nblocks
        blockw = self._pyilu.blockw
        rbc = []
        for bi in range(nblocks):
            l = []
            for ri in range(blockw):
                l.append(self._pyilu.colbyrow_get(bi*blockw+ri))
            rbc.append(l)
        return rbc

    @property
    def LU_merged(self):
        nblocks = self._pyilu.nblocks
        blockw = self._pyilu.blockw
        ndiag = self._pyilu.ndiag
        dim = nblocks*blockw
        LU = np.zeros((dim, dim))
        LUblocks = self._pyilu.get_LU()
        for bi in range(nblocks):
            slc = slice(bi*blockw, (bi+1)*blockw)
            LU[slc, slc] = LUblocks[:, slc]
        for di in range(ndiag):
            idx = 0
            for bi in range(nblocks-di-1):
                for ci in range(blockw):
                    lri_u = self._pyilu.rowbycol_get(idx)
                    lri_l = self._pyilu.rowbycol_get(idx+blockw*di)
                    LU[bi*blockw + lri_l + blockw*(di+1), idx] = self._pyilu.sub_get(
                        di, bi, ci)
                    LU[bi*blockw + lri_u, idx + blockw*(di+1)] = self._pyilu.sup_get(
                        di, bi, ci)
                    idx += 1
        return LU

    @property
    def piv(self):
        blockw = self._pyilu.blockw
        p = []
        for bi in range(self._pyilu.nblocks):
            pp = []
            for ci in range(blockw):
                pp.append(self._pyilu.piv_get(bi*blockw+ci))
            p.append(pp)
        return p


def fast_FakeLU(A, n, ndiag=0):

    assert A.shape[0] == A.shape[1]
    assert A.shape[0] % n == 0
    nblocks = A.shape[0]//n
    sub, sup = [], []
    for di in range(ndiag):
        ssub, ssup = [], []
        for gi in range((nblocks-di-1)*n):
            ssub.append(A[gi + (di+1)*n, gi])
            ssup.append(A[gi, gi + (di+1)*n])
        sub.extend(ssub)
        sup.extend(ssup)
    # memory view taking address of first element workaround:
    # if len(sub) == 0:
    #     sub.append(0)
    #     sup.append(0)
    return ILU(np.asfortranarray(A),
               np.array(sub, dtype=np.float64),
               np.array(sup, dtype=np.float64),
               n, ndiag)

# Below is the prototype from which block_diag_ilu.hpp was
# designed: (tests were made for FakeLU and should pass
# for fast_FakeLU above)


def rowpiv2rowbycol(piv):
    rowbycol = np.arange(len(piv))
    for i in range(len(piv)):
        j = piv[i]
        if i != j:
            tmp = rowbycol[j]
            rowbycol[j] = i
            rowbycol[i] = tmp
    return rowbycol


class FakeLU:
    def __init__(self, A, n, ndiag=0):
        self.lu, self.piv, self.rowbycol = [], [], []
        self.colbyrow = []
        self.n = n
        self.ndiag = ndiag
        assert A.shape[0] == A.shape[1]
        assert A.shape[0] % n == 0
        self.N = A.shape[0]//n

        # Block diagonal
        for bi in range(self.N):
            slc = slice(bi*self.n, (bi+1)*self.n)
            lu, piv = scipy.linalg.lu_factor(A[slc, slc])
            self.lu.append(lu)
            self.piv.append(piv)
            self.rowbycol.append(rowpiv2rowbycol(piv))
            self.colbyrow.append([list(self.rowbycol[-1]).index(x) for x in range(self.n)])

        # Sub diagonal
        self.sub, self.sup = [], []
        for di in range(1, self.ndiag+1):
            ssub = []
            ssup = []
            for bi in range(self.N-di):
                for ci in range(self.n):
                    d = self.lu[bi][ci, ci]
                    ssub.append(A[(bi+di)*n + ci, bi*n + ci]/d) # sub[column_idx]
                    ssup.append(A[bi*n + ci, (bi+di)*n + ci]) # sup[column_idx]

            self.sub.append(ssub)
            self.sup.append(ssup)

    @property
    def L_dot_U(self):
        # ILU => L*U ~= A
        # this should give a better approximation of A
        # Only useful for debugging / accuracy tests...
        A = np.zeros((self.N*self.n, self.N*self.n))
        for bi in range(self.N):
            # Diagonal blocks...
            L = np.zeros((self.n, self.n))
            U = L.copy()
            for ri in range(self.n):
                for ci in range(self.n):
                    if ci == ri:
                        U[ri, ci] = self.lu[bi][ri, ci]
                        L[ri, ci] = 1.0
                    elif ci > ri:
                        U[ri, ci] = self.lu[bi][ri, ci]
                    else:
                        L[ri, ci] = self.lu[bi][ri, ci]
            slc = slice(bi*self.n, (bi+1)*self.n)
            A[slc, slc] = np.dot(L, U)
        for di in range(1, self.ndiag+1):  # diag
            for bi in range(self.N-di):  # block
                for ci in range(self.n):
                    # upper
                    A[bi*self.n + self.rowbycol[bi][ci], (bi+di)*self.n+ci] = self.sup[di-1][bi*self.n + ci]
                    # lower
                    A[(bi+di)*self.n+self.rowbycol[bi+di][ci], bi*self.n+ci] = self.sub[di-1][bi*self.n + ci]*self.lu[bi][ci, ci]
        return A

    # def permute_vec(self, x):
    #     n = np.empty_like(x)
    #     for bi in range(self.N):
    #         for li in range(self.n):
    #             n[bi*self.n+li] = x[bi*self.n+self.rowbycol[bi][li]]
    #     return n

    # def antipermute_vec(self, x):
    #     n = x[:]
    #     for bi in range(self.N):
    #         for li in range(self.n):
    #             n[bi*self.n+li] = x[bi*self.n+self.colbyrow[bi][li]]
    #     return n

    def solve(self, b):
        """
        LUx = b:
           Ly = b
           Ux = y
        """
        #b = self.permute_vec(b)
        y = []
        for bri in range(self.N):  # block row index
            for li in range(self.n):  # local row index
                s = 0.0
                for lci in range(li): # local column index
                    s += self.lu[bri][li, lci]*y[bri*self.n+lci]
                for di in range(1, self.ndiag+1):
                    if bri >= di:
                        # di:th sub diagonal (counted as distance from main diag)
                        ci = self.colbyrow[bri][li]
                        s += self.sub[di-1][(bri-di)*self.n+ci]*y[
                            (bri-di)*self.n + ci]
                y.append(b[bri*self.n+self.rowbycol[bri][li]]-s) # Doolittle: L[i, i] == 1
        x = [0]*len(y)
        for bri in range(self.N-1, -1, -1):
            for li in range(self.n - 1, -1, -1):
                s = 0.0
                for ci in range(li+1, self.n):
                    s += self.lu[bri][li, ci]*x[bri*self.n + ci]
                for di in range(1, self.ndiag+1):
                    if bri < self.N-di:
                        ci = self.colbyrow[bri][li]
                        s += self.sup[di-1][bri*self.n+ci]*x[(bri+di)*self.n + ci]
                x[bri*self.n+li] = (y[bri*self.n + li] - s)/self.lu[bri][li, li]
        return x #self.antipermute_vec(x)

    @property
    def LU_merged(self):
        A = np.zeros((self.N*self.n, self.N*self.n))
        for bi in range(self.N):
            slc = slice(bi*self.n, (bi+1)*self.n)
            A[slc, slc] = self.lu[bi]
            for ci in range(self.n):
                for di in range(1, self.ndiag+1):
                    # bi means block row index:
                    if bi >= di:
                        A[bi*self.n+self.rowbycol[bi][ci], (bi-di)*self.n+ci] = self.sub[di-1][(bi-di)*self.n + ci]
                    if bi < self.N-di:
                        A[bi*self.n+self.rowbycol[bi][ci], (bi+di)*self.n+ci] = self.sup[di-1][bi*self.n + ci]
        return A
