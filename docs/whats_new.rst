:orphan:

.. _whats_new:

.. currentmodule:: seek_localize

What's new?
===========

Here we list a changelog of SEEK-Localize.

.. currentmodule:: seek_localize

Current
-------

.. currentmodule:: seek_localize
.. _changes_0_2:

Version 0.2 (unreleased)
------------------------

Authors
~~~~~~~

* `Adam Li`_
* `Chester Huynh`_

Changelog
~~~~~~~~~

- Added string representation for ``Sensors`` class, by `Adam Li`_ (:gh:`3`)

Bug
~~~

-

API
~~~

- Implementation of :func:`seek_localize.write_dig_bids` for outputting ``*_electrodes.tsv`` and ``*_coordsystem.json`` files, by `Adam Li`_ (:gh:`4`)
- Improve API to have conversion of units (voxels and xyz :func:`seek_localize.convert_coord_units`) and conversion of coordinate spaces (:func:`seek_localize.convert_coord_space`) that are BIDS-complaint, by `Adam Li`_ (:gh:`3`)

:doc:`Find out what was new in previous releases <whats_new_previous_releases>`

.. include:: authors.rst