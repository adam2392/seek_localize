:orphan:

.. _Contribute:

Contributing to SEEK-Localize
=============================

(adopted from scikit-learn)

``The latest contributing guide is available in the repository at:``
`https://github.com/ncsl/seek/CONTRIBUTING.md`_

There are many ways to contribute to SEEK, with the most common ones
being contribution of code or documentation to the project. Improving the
documentation is no less important than improving the pipeline itself. If you
find a typo in the documentation, or have made improvements, do not hesitate to
submit a GitHub pull request.

But there are many other ways to help. In particular answering queries on the
`issue tracker <https://github.com/ncsl/seek/issues>`_, and
investigating bugs are very valuable contributions that decrease the burden on 
the project maintainers.

Another way to contribute is to report issues you're facing, and give a "thumbs
up" on issues that others reported and that are relevant to you. It also helps
us if you spread the word: reference the project from your blog and articles,
link to it from your website, or simply star it in GitHub to say "I use it".

Another way to contribute is specifically to make additional pipelines that improve 
the accuracy of contact localization for iEEG data using the T1 and CT images.

Code of Conduct
---------------

We abide by the principles of openness, respect, and consideration of others
of the Python Software Foundation: https://www.python.org/psf/codeofconduct/.

Code Guidelines
----------------

*Before starting new code*, we highly recommend opening an issue on `GitHub <https://github.com/ncsl/seek>`_ to discuss potential changes.

* Please use standard `black <https://black.readthedocs.io/en/stable/>`_ Python style guidelines. To test that your code complies with those, you can run:

  .. code-block:: bash

     $ black --check seek_localize/
     $ make check

  In addition to `black`, `make check` command runs `pydocstyle`, `codespell`, `check-manifest` and `mypy`.

* Use `NumPy style <https://numpydoc.readthedocs.io/en/latest/format.html>`_ for docstrings. Follow existing examples for simplest guidance.

* New functionality must be **validated** against sample datasets.

* Changes must be accompanied by **updated documentation** and examples.

* After making changes, **ensure all tests pass**. This can be done by running:

  .. code-block:: bash

     $ pytest --doctest-modules
     $ make test

Install development version of seek-localize
--------------------------------------------
First, you should [fork](https://help.github.com/en/github/getting-started-with-github/fork-a-repo) the `seek_localize` repository.
Then, clone the fork and install it in.

.. code-block:: bash

    $ git clone https://github.com/adam2392/seek_localize
    $ python3 -m venv .venv
    $ pipenv install --dev

Installing via ``pipenv`` with the ``--dev`` flag will install all the development dependencies as well.

Building the documentation
--------------------------
The documentation can be built using sphinx. To build the documentation locally, one can run:

.. code-block:: bash

    $ cd doc/
    $ make html

or

.. code-block:: bash

    $ make html-noplot

if you don't want to run the examples to build the documentation.
This will result in a faster build but produce no plots in the examples.

BIDS-Validation
---------------
To robustly apply seek workflows and reconstruction visualiztion, we rely on the BIDS specification
for storing data. One can use the `bids-validator <https://github.com/bids-standard/bids-validator>`_ to verify that a dataset is BIDS-compliant.

Test your installation by running:

.. code-block:: bash

    $ bids-validator --version

To then validate your dataset before using ``seek_localize``

.. code-block:: bash

    $ bids-validator <bids_root_path>