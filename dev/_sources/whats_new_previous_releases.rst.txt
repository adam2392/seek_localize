:orphan:

.. currentmodule:: seek_localize

What was new in previous releases?
==================================

.. currentmodule:: seek_localize
.. _changes_0_1:

Version 0.1 (2020-10-22)
------------------------

Notable changes
~~~~~~~~~~~~~~~
This first version of ``seek-localize`` serves to provide a tested and
documented API for interfacing with ``*electrodes.tsv``, ``*coordsystem.json``,
``*T1w.nii`` and FreeSurfer derivative files to provide anatomical information
for iEEG electrodes.

Author List
~~~~~~~~~~~

- `Adam Li`_
- `Christopher Coogan`_
- `Chester Huynh`_

Detailed list of changes
~~~~~~~~~~~~~~~~~~~~~~~~


Enhancements
^^^^^^^^^^^^
- 

Bug fixes
^^^^^^^^^

- Anatomical labeling occurs now via voxel space

API changes
^^^^^^^^^^^

- Refactored semi-automated algorithm for localizing contacts on CT img, in :code:`seek/localizae_contacts/electrode_clustering` by `Chester Huynh`_ (:gh:`16`)


.. include:: authors.rst
