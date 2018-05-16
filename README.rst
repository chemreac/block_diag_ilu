block_diag_ilu
==============

.. image:: http://hera.physchem.kth.se:9090/api/badges/chemreac/block_diag_ilu/status.svg
   :target: http://hera.physchem.kth.se:9090/chemreac/block_diag_ilu
   :alt: Build status


``block_diag_ilu`` is an open source `C++ single header-file implementation
<https://github.com/chemreac/block_diag_ilu/tree/master/block_diag_ilu/include>`_ of an
incomplete LU decomposition routine suitable for diagonally dominant (square) block diagonal
matrices with sub- and super diagonals of small magnitude. It is useful for
preconditioning linear systems. The use-case in mind is for integrating discretized PDEs of mixed
chemical kinetics / diffusion problems where the diffusion process may be accurately
considered a mild perturbation.

A picture is worth a thousand words, so if your matrix looks anything like this:

.. image:: https://raw.githubusercontent.com/bjodah/block_diag_ilu/master/scripts/matrix.png
   :scale: 50%
   :alt: Diagonally dominant block diagonal matrix with sub- and super-diagonals
   
then its LU decomposition then looks like this:

.. image:: https://raw.githubusercontent.com/bjodah/block_diag_ilu/master/scripts/matrix_lu.png
   :scale: 50%
   :alt: LU decomposition of same matrix

then ``block_diag_ilu`` should be able to save quite a bit of time when
solving linear systems approximately, *e.g.* for preconditioning.

Conditional compilation
-----------------------
The following macros affect the compilation:

+---------------------------+-----------------------------------------------+---------------+
|Macro name                 |Action (when defined)                          |Default        |
+===========================+===============================================+===============+
|NDEBUG                     |use ``std::unique_ptr`` instead of             |undefined      |
|                           |``std::vector`` as underlying data structure.  |               |
+---------------------------+-----------------------------------------------+---------------+
|BLOCK_DIAG_ILUT_WITH_OPENMP|factorize blocks in parallel. Set the          |undefined      |
|                           |environment variable                           |               |
|                           |`BLOCK_DIAG_ILU_NUM_THREADS` to control number |               |
|                           |of threads.                                    |               |
+---------------------------+-----------------------------------------------+---------------+

``setup.py`` will set these when envinronment variables with those are set to "1".

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
