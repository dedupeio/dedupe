======================
Mac OS X Install Notes
======================


Apple’s implementation of `BLAS <http://en.wikipedia.org/wiki/BLAS>`__
does not support using `BLAS calls on both sides of a
fork <http://mail.scipy.org/pipermail/numpy-discussion/2012-August/063589.html>`__.
The upshot of this is that, when using NumPy functions that rely on BLAS
calls within a forked process (such as ones created when you push a job
into a multiprocessing pool) the fork might never actually fully exit.
Which means you end up with orphaned processes until the process that
was originally forked exits. Since under the hood Dedupe relies upon
NumPy calls within a multiprocessing pool, this can be an issue if you
are planning on running something like a daemon process that then forks
off processes that run Dedupe.

One way to get around this is to compile NumPy against a different
implementation of BLAS such as
`OpenBLAS <https://github.com/xianyi/OpenBLAS>`__. Here’s how you might
go about that:

Clone and build OpenBLAS source with ``USE_OPENMP=0`` flag
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: bash

    $ git clone https://github.com/xianyi/OpenBLAS.git
    $ cd OpenBLAS
    $ make USE_OPENMP=0 (or `make BINARY=64 USE_OPENMP=0` for 64 bit)
    $ mkdir /usr/local/opt/openblas # Change this to suit your needs
    $ make PREFIX=/usr/local/opt/openblas install # Make sure this matches the path above

Clone and build NumPy
~~~~~~~~~~~~~~~~~~~~~

Make sure it knows where you just built OpenBLAS. This involves editing
the site.cfg file within the NumPy source (see
http://stackoverflow.com/a/14391693/1907889 for details). The paths that
you’ll enter in there are relative to the ones used in step one above.

.. code:: bash

    $ git clone git://github.com/numpy/numpy.git numpy
    $ cd numpy
    $ pip uninstall numpy (if it is already installed)
    $ cp site.cfg.example site.cfg

Edit site.cfg and uncomment/update the code to match below:

::

    [openblas]
    libraries = openblas
    library_dirs = /usr/local/opt/openblas/lib
    include_dirs = /usr/local/opt/openblas/include

Then install with:

::

    python setup.py build && python setup.py install

Then reinstall Dedupe:

::

    pip uninstall Dedupe
    python setup.py install

The `Homebrew Science <https://github.com/Homebrew/homebrew-science>`__
formulae also offer an OpenBLAS formula but as of this writing it `was
still
referencing <https://github.com/Homebrew/homebrew-science/blob/master/openblas.rb>`__
the current release of OpenBLAS (0.2.8) which does not include a fix for
`a bug <https://github.com/xianyi/OpenBLAS/issues/294>`__ which is the
whole reason this is necessary in the first place. Once that fix is
rolled into a release and the Homebrew formula is updated, this will be
a better approach to getting this setup.
