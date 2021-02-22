=======================================================
seek_localize (Stereotactic ElectroEncephalography Kit)
=======================================================

.. image:: https://circleci.com/gh/adam2392/seek_localize.svg?style=svg
   :target: https://circleci.com/gh/adam2392/seek_localize
   :alt: CircleCI

.. image:: https://github.com/adam2392/seek_localize/workflows/.github/workflows/main.yml/badge.svg
    :target: https://github.com/adam2392/seek_localize/actions/
    :alt: GitHub Actions

.. image:: https://github.com/adam2392/seek_localize/workflows/test_suite/badge.svg
    :target: https://github.com/adam2392/seek_localize/actions/
    :alt: Test Suite

.. image:: https://img.shields.io/badge/code%20style-black-000000.svg
   :target: https://github.com/ambv/black
   :alt: Code style: black

.. image:: https://codecov.io/gh/adam2392/seek_localize/branch/master/graph/badge.svg
   :target: https://codecov.io/gh/adam2392/seek_localize

.. image:: https://readthedocs.org/projects/seek_localize/badge/?version=latest
    :target: https://seek_localize.readthedocs.io/en/latest/?badge=latest
    :alt: Documentation Status

.. image:: https://zenodo.org/badge/DOI/10.5281/zenodo.4508674.svg
   :target: https://doi.org/10.5281/zenodo.4508674

seek_localize helps localize iEEG electrodes on MRI/CT images and
deals with data processing for iEEG-BIDS data.

Documentation
-------------
The documentation can be found under the following links:

- for the `stable release <https://seek_localize.readthedocs.io/en/stable/index.html>`_
- for the `latest (development) version <https://seek_localize.readthedocs.io/en/latest/index.html>`_

To see the ``seek`` documentation, see http://neuroseek.azurewebsites.net/docs/seek/


Setup and Installation
----------------------

See `INSTALLATION GUIDE <https://github.com/adam2392/seek_localize/blob/master/doc/installation.rst>`_ for full instructions.
A quick setup can occur with github and ``pipenv``. This has been tested on
Python versions 3.7, 3.8 and 3.9.

.. code-block:: bash

    # clone repository locally
    $ git clone https://github.com/adam2392/seek_localize
    $ python3.8 -m venv .venv
    $ pipenv install

Through pip

.. code-block:: bash

    $ pip install seek_localize

Quick Usage
-----------
Here is a quick look at a basic use-case, where we want to label the anatomical regions each
electrode is in, based on FreeSurfer.

.. code-block:: python

    from seek_localize import label_elecs_anat
    from mne_bids import BIDSPath

    # define file path to the T1w image that electrode coordinates are localized in
    img_fname = ...

    # define a path to the electrodes.tsv file in iEEG-BIDS
    bids_path = BIDSPath(..., suffix='electrodes', extension='.tsv')

    # define filepath to the FreeSurferColor Look up Table
    # (it is stored locally)
    fs_lut_fpath = seek_localize.fs_lut_fpath

    label_elecs_anat(bids_path, img_fname, fs_lut_fpath)


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
seek_localize was created and is maintained by `Adam Li <https://adam2392.github.io>`_. It is also maintained and contributed by
`Christopher Coogan <https://github.com/TheBrainChain>`_ and other researchers in the NCSL and Crone lab.
Contributions are more than welcome so feel free to contact me, open an issue or submit a pull request! See the
`contribution guide <https://github.com/adam2392/seek_localize/blob/master/docs/contributing.rst>`_.

To report a bug, please visit the `GitHub repository <https://github.com/adam2392/seek_localize/issues/>`_.

Note that this program is provided with NO WARRANTY OF ANY KIND. If you can, always double check the results with a
human researcher, or clinician.

How to cite?
============

If you want to cite ``seek_localize``, please cite the following paper(s).

Adam Li. (2021, February 5). seek_localize (Version 0.0.1). Zenodo. http://doi.org/10.5281/zenodo.4508674

Acknowledgement
===============

Appelhoff, S., Sanderson, M., Brooks, T., Vliet, M., Quentin, R., Holdgraf, C.,
Chaumon, M., Mikulan, E., Tavabi, K., Höchenberger, R., Welke, D., Brunner, C.,
Rockhill, A., Larson, E., Gramfort, A., & Jas, M. (2019): **MNE-BIDS: Organizing
electrophysiological data into the BIDS format and facilitating their analysis.**
*Journal of Open Source Software,* 4:1896. DOI: [10.21105/joss.01896](https://doi.org/10.21105/joss.01896)


FAQ
===
1. For ECoG data, we do not explicitly have a process outlined, but these are significantly easier since grids can
be easily interpolated. See `Fieldtrip Toolbox`_.

.. _FieldTrip Toolbox: http://www.fieldtriptoolbox.org/tutorial/human_ecog/