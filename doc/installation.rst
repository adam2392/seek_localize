:orphan:

.. _installation:

INSTALLATION GUIDE
==================
``seek`` uses open-source third-party software to run the various workflows (e.g. `Freesurfer`_).
``seek`` itself is a wrapper using snakemake_. The best way to install the 3rd party software for ``seek`` usage
is via a Docker installation.

To fully install SEEK and run workflows, one will need to:

#. install SEEK repository
#. pull Docker containers

We outline some of these steps below. After you have set up everything (don't forget to
format your data repository according to our necessary format), then you can easily run
the snakemake workflows. For more information on running workflows after
installation, see :doc:`usage instructions <use>`.

Prerequisites
-------------

seek Installation
-----------------
There are a few ways to install seek itself.

.. code-block:: bash

    # clone repository locally
    $ git clone https://github.com/adam2392/seek_localize
    # update pip and pipenv
    $ pip install --upgrade pip
    $ pip install --upgrade pipenv
    $ python3.8 -m venv .venv
    $ pipenv install

Now to run localization algorithms for your CT image. Your BIDS dataset should be setup as such:

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

Here, the ``*electrodes.tsv`` file should already be there, with minimally 2 contacts localized manually per
sEEG electrode.

.. _Blender: https://www.blender.org/download/Blender2.81/blender-2.81-linux-glibc217-x86_64.tar.bz2/
.. _Freesurfer: https://surfer.nmr.mgh.harvard.edu/fswiki/DownloadAndInstall
.. _FSL Flirt: https://fsl.fmrib.ox.ac.uk/fsl/fslwiki/FslInstallation/
.. _SPM: https://www.fil.ion.ucl.ac.uk/spm/software/spm12/
.. _FieldTripToolbox: http://www.fieldtriptoolbox.org/download/
.. _Docker: https://docs.docker.com/get-docker/