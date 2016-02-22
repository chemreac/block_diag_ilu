block_diag_ilu
==============

.. image:: http://hera.physchem.kth.se:9090/api/badges/chemreac/block_diag_ilu/status.svg
   :target: http://hera.physchem.kth.se:9090/chemreac/block_diag_ilu
   :alt: Build status


``block_diag_ilu`` is an open source `C++ single header-file implementeation
<https://github.com/chemreac/block_diag_ilu/tree/master/include>`_ of an
incomplete LU decomposition routine suitable for diagonally dominant block diagonal
matrices with sub- and super diagonals of small magnitude. It is useful for
preconditioning linear systems when e.g. integrating discretized PDEs of mixed
chemical kinetics / diffusion problems where the diffusion process may be accurately
considered a mild perturbation.

Conditional compilation
-----------------------
The following macros affect the compilation:

+--------------------------+-----------------------------------------------+---------------+
|Variable name             |Action                                         |Default        |
+==========================+===============================================+===============+
|NDEBUG                    |use ``std::unique_ptr`` instead of             |undefined      |
|                          |``std::vector`` as underlying data structure.  |               |
+--------------------------+-----------------------------------------------+---------------+
|WITH_BLOCK_DIAG_ILU_DGETRF|Use unblocked (parallell) internal             |undefined      |
|                          |implementation of LACKPACKS's ``dgetrf`` (uses |               |
|                          |OpenMP)                                        |               |
+--------------------------+-----------------------------------------------+---------------+


License
-------
The source code is Open Source and is released under the very permissive
"simplified (2-clause) BSD license". See ``LICENSE.txt`` for further details.
Contributors are welcome to suggest improvements at https://github.com/chemreac/block_diag_ilu

Author
------
Björn Dahlgren, contact:
 - gmail adress: bjodah
 - kth.se adress: bda
