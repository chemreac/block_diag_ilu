v0.3.7
======
- Avoid using exceptions in program flow

v0.3.6
======
- Only require C++11
- Deprecate BLOCK_DIAG_ILU_WITH_GETRF macro

v0.3.5
======
- Require C++14 for now, rationale: Python's distutils' C++ support is truly terrible
  (CXXFLAGS not supported for starters).

v0.3.4
======
- More robust build (conda-recipe)
- Updated AnyODE

v0.3.3
======
- Fix ld in Cython wrapper

v0.3.2
======
- BlockDiagMatrix needed to inspect m_colmaj

v0.3.1
======
- Bumpy AnyODE
- Make helper functions inline (avoid multiple definition)

v0.3.0
======
- Use int throughout instead of unsigned and size_t.
- Require C++14
- Support satellite diagonals (from e.g. MOL with PBC)

v0.2.5
======
- Enhanced conda recipe

v0.2.4
======
- More robust setup.py

v0.2.2
======
- More robust setup.py

v0.2.1
======
- More robust setup.py

v0.2.0
======
- solve may now return non-zero exit (in C++, raises error in Python)
- check for NaN in solve
- check for zero diagonal elements in U

v0.1.1
======
- LU bindings in Python

v0.1
====
- Initial release with a minimal Python interface 
