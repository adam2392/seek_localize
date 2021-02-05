:orphan:

.. _installation:

Installation
============

Dependencies
------------

* ``mne`` (>=0.22)
* ``mne-bids`` (>=0.6)
* ``numpy`` (>=1.14)
* ``nibabel`` (>=)
* ``joblib`` (>=1.0.0)
* ``pandas`` (>=1.0.0)
* ``scipy`` (>=1.5.0)
* ``matplotlib`` (optional, for using the interactive data inspector)

We require that you use Python 3.6 or higher.
You may choose to install ``seek_localize`` `via pip <#Installation via pip>`_.

Installation via Pipenv
-----------------------

To install seek-localize including all dependencies required to use all features,
simply run the following at the root of the repository:

.. code-block:: bash

    python -m venv .venv
    pipenv install

If you want to install a snapshot of the current development version, run:

.. code-block:: bash

   pip install --user -U https://api.github.com/repos/adam2392/seek-localize/zipball/master

To check if everything worked fine, the following command should not give any
error messages:

.. code-block:: bash

   python -c 'import seek_localize'
