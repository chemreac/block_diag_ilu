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
