============================================
Tutorial for the Gamma-Ray Astronomy Session
============================================

Date & time
-----------
Wednesday, 30th of September 2026

Session 1: 11:30 - 13:00
Session 2: 14:30 - 17:00

Tutor
-----

.. list-table::
   :header-rows: 1

   * - .. image:: https://github.com/TjarkMiener.png?size=100
        :target: https://github.com/TjarkMiener
        :alt: Tjark Miener
     
   * - `Tjark Miener <https://github.com/TjarkMiener>`_


CTLearn: Deep Learning for IACT Event Reconstruction
====================================================

.. image:: https://zenodo.org/badge/DOI/10.5281/zenodo.3342952.svg
   :target: https://doi.org/10.5281/zenodo.3342952
   :alt: DOI

.. image:: https://img.shields.io/pypi/v/ctlearn
    :target: https://pypi.org/project/ctlearn/
    :alt: Latest Release

.. image:: https://github.com/ctlearn-project/ctlearn/actions/workflows/python-package-conda.yml/badge.svg
    :target: https://github.com/ctlearn-project/ctlearn/actions/workflows/python-package-conda.yml
    :alt: Continuos Integration
    
.. image:: tutorials/ctlearn/images/CTLearnTextCTinBox_WhiteBkgd.png
   :target: tutorials/ctlearn/images/CTLearnTextCTinBox_WhiteBkgd.png
   :alt: CTLearn Logo


CTLearn is a package under active development to run deep learning models to analyze data from all major current and future arrays of imaging atmospheric Cherenkov telescopes (IACTs). CTLearn can load R1/DL0/DL1 data from `CTAO <https://www.cta-observatory.org/>`_ (Cherenkov Telescope Array Observatory), `FACT <https://www.isdc.unige.ch/fact/>`_\ , `H.E.S.S. <https://www.mpi-hd.mpg.de/hfm/HESS/>`_\ , `LST-1 <https://www.lst1.iac.es/>`_\ , `MAGIC <https://magic.mpp.mpg.de/>`_\ , and `VERITAS <https://veritas.sao.arizona.edu/>`_ telescopes reduced by `ctapipe <https://github.com/cta-observatory/ctapipe>`_ and processed by `DL1DataHandler <https://github.com/cta-observatory/dl1-data-handler>`_.

* Code, feature requests, bug reports, pull requests: https://github.com/ctlearn-project/ctlearn
* Documentation: https://ctlearn.readthedocs.io
* License: BSD-3

Installation for users
----------------------

First, create and activate a fresh conda environment:

.. code-block:: bash

   mamba create -n ctlearn -c conda-forge python==3.12 llvmlite
   conda activate ctlearn

The lastest version fo this package can be installed as a pip package:

.. code-block:: bash

   pip install ctlearn

See the documentation for further information like `installation instructions for developers <https://ctlearn.readthedocs.io/en/latest/installation.html#installing-with-pip-setuptools-from-source-for-development>`_, `package usage <https://ctlearn.readthedocs.io/en/stable/usage.html>`_, and `dependencies <https://ctlearn.readthedocs.io/en/stable/installation.html#dependencies>`_ among other topics.

`ViTables <https://github.com/uvemas/ViTables>`_ is a component of the PyTables family. It is a GUI for browsing and editing files in both PyTables and HDF5 formats. It is an useful tool throughout the tutorial and it is recommended to install ViTables in a separated conda/mamba environment:

.. code-block:: bash

   mamba create -n vitables -c conda-forge python==3.12 vitables
   conda run -n vitables vitables hdf5_filepaths
