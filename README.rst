=======================================================
SEEK-Localize (Stereotactic ElectroEncephalography Kit)
=======================================================

.. image:: https://circleci.com/gh/adam2392/seek_localize.svg?style=svg
   :target: https://circleci.com/gh/adam2392/seek_localize
   :alt: CircleCI

.. image:: https://github.com/adam2392/seek_localize/workflows/.github/workflows/main.yml/badge.svg
    :target: https://github.com/adam2392/seek_localize/actions/
    :alt: GitHub Actions

.. image:: https://img.shields.io/badge/code%20style-black-000000.svg
   :target: https://github.com/ambv/black
   :alt: Code style: black

This repo helps localize iEEG electrodes on CT images.

For ECoG data, we do not explicitly have a process outlined, but these are significantly easier since grids can
be easily interpolated. See `Fieldtrip Toolbox`_.

Documentation
-------------
To see the entire documentation, see http://neuroseek.azurewebsites.net/docs/seek/

Setup and Installation
----------------------

See `INSTALLATION GUIDE <https://github.com/adam2392/seek_localize/blob/master/doc/installation.rst>`_ for full instructions.
A quick setup can occur with ``pipenv``.

.. code-block:: bash

    # clone repository locally
    $ git clone https://github.com/adam2392/seek_localize
    $ python3.8 -m venv .venv
    $ pipenv install


Data Organization
-----------------

We use BIDS. See https://github.com/bids-standard/bids-starter-kit/wiki/The-BIDS-folder-hierarchy
for more information. We recommend the following BIDS structure with the minimally required set of files.

.. code-block::

   {bids_root}/
        /sub-001/
            /anat/
                - sub-001_*_T1w.nii
            /ct/
                - sub-001_*_CT.nii
            /ieeg/
                - sub-001_*_channels.tsv
                - sub-001_*_electrodes.tsv
                - *

Development
===========

Seek was created and is maintained by `Adam Li <https://adam2392.github.io>`_. It is also maintained and contributed by
`Christopher Coogan <https://github.com/TheBrainChain>`_ and other researchers in the NCSL and Crone lab. Contributions are more than welcome so feel free to contact me, open an issue or submit a pull request! See the
`contribution guide <https://github.com/adam2392/seek_localize/blob/master/doc/contributing.rst>`_.

To report a bug, please visit the `GitHub repository <https://github.com/adam2392/seek_localize/issues/>`_.

Note that this program is provided with NO WARRANTY OF ANY KIND. If you can, always double check the results with a human researcher, or clinician.

How to cite?
============

If you want to cite ``seek_localize``, please cite the following paper(s).



Acknowledgement
===============

Appelhoff, S., Sanderson, M., Brooks, T., Vliet, M., Quentin, R., Holdgraf, C.,
Chaumon, M., Mikulan, E., Tavabi, K., Höchenberger, R., Welke, D., Brunner, C.,
Rockhill, A., Larson, E., Gramfort, A., & Jas, M. (2019): **MNE-BIDS: Organizing
electrophysiological data into the BIDS format and facilitating their analysis.**
*Journal of Open Source Software,* 4:1896. DOI: [10.21105/joss.01896](https://doi.org/10.21105/joss.01896)

- `iEEG-BIDS <https://doi.org/10.1038/s41597-019-0105-7>`_

FAQ
===

