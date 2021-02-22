:orphan:

.. currentmodule:: seek_localize

What was new in previous releases?
==================================

.. currentmodule:: seek_localize
.. _changes_0_2:

Version 0.2 (2021-02-22)
------------------------

Notable changes
~~~~~~~~~~~~~~~
This version of ``seek-localize`` provides improved BIDS io and
a unified API to perform conversion of units and coordinate systems.

Authors
~~~~~~~

* `Adam Li`_
* `Chester Huynh`_

Changelog
~~~~~~~~~

- Added string representation for ``Sensors`` class, by `Adam Li`_ (:gh:`3`)

API
~~~

- Implementation of :func:`seek_localize.write_dig_bids` for outputting ``*_electrodes.tsv`` and ``*_coordsystem.json`` files, by `Adam Li`_ (:gh:`5`)
- Improve API to have conversion of units (voxels and xyz :func:`seek_localize.convert_coord_units`) and conversion of coordinate spaces (:func:`seek_localize.convert_coord_space`) that are BIDS-complaint, by `Adam Li`_ (:gh:`3`)

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
