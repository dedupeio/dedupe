======================
Mac OS X Install Notes
======================


Apple’s implementation of `BLAS <http://en.wikipedia.org/wiki/BLAS>`__
does not support using `BLAS calls on both sides of a fork
<http://mail.scipy.org/pipermail/numpy-discussion/2012-August/063589.html>`__.

The upshot of this is that you can't do parallel processing with numpy
(which uses BLAS).

One way to get around this is to compile NumPy against a different
implementation of BLAS such as
`OpenBLAS <https://github.com/xianyi/OpenBLAS>`__. Here’s how you might
go about that:

Install OpenBlas with Homebrew Science
~~~~~~~~~~~~~~~~~~~~~
You can install OpenBlas from with Homebrew Science.

.. code:: bash

    $ brew install homebrew/science/openblas


Clone and build NumPy
~~~~~~~~~~~~~~~~~~~~~

.. code:: bash

    $ git clone git://github.com/numpy/numpy.git numpy
    $ cd numpy
    $ pip uninstall numpy (if it is already installed)
    $ cp site.cfg.example site.cfg

Edit site.cfg and uncomment/update the code to match below:

::

    [DEFAULT]
    library_dirs = /usr/local/opt/openblas/lib
    include_dirs = /usr/local/opt/openblas/include

    [atlas]
    atlas_libs = openblas
    libraries = openblas

    [openblas]
    libraries = openblas
    library_dirs = /usr/local/opt/openblas/lib
    include_dirs = /usr/local/opt/openblas/include

You may need to change the ``library_dirs`` and ``include_dirs`` paths
to match where you installed OpenBlas (see
http://stackoverflow.com/a/14391693/1907889 for details).

Then install with:

::

    python setup.py build && python setup.py install

Then reinstall Dedupe:

::

    pip uninstall Dedupe
    python setup.py install


