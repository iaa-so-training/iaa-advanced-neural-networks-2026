============================================
Tutorial for the Gamma-Ray Astronomy Session
============================================

Date & time
-----------
Wednesday, 30th of September 2026

Session 1: 11:30 - 13:00 (Tutorial 1 and Tutorial 2 Part A)
Session 2: 14:30 - 17:00 (Tutorial 2 Part B and Tutorial 3)

Tutor
-----

.. list-table::
   :header-rows: 1

   * - .. image:: https://github.com/tjarkmiener.png?size=100
        :target: https://github.com/tjarkmiener
        :alt: Tjark Miener
     
   * - `Tjark Miener <https://github.com/tjarkmiener>`_

with the help of:

.. list-table::
   :header-rows: 1

   * - .. image:: https://github.com/cpozogonzalez.png?size=100
        :target: https://github.com/cpozogonzalez
        :alt: Cristian Pozo González

     - .. image:: https://github.com/rlopezcoto.png?size=100
        :target: https://github.com/rlopezcoto
        :alt: Rubén López-Coto

   * - `Cristian Pozo González <https://github.com/cpozogonzalez>`_
     - `Rubén López-Coto <https://github.com/rlopezcoto>`_

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
    
.. image:: images/CTLearnTextCTinBox_WhiteBkgd.png
   :target: images/CTLearnTextCTinBox_WhiteBkgd.png
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

The lastest version (v0.10.4) fo this package can be installed as a pip package:

.. code-block:: bash

   pip install ctlearn==0.10.4

See the documentation for further information like `installation instructions for developers <https://ctlearn.readthedocs.io/en/latest/installation.html#installing-with-pip-setuptools-from-source-for-development>`_, `package usage <https://ctlearn.readthedocs.io/en/stable/usage.html>`_, and `dependencies <https://ctlearn.readthedocs.io/en/stable/installation.html#dependencies>`_ among other topics.

`ViTables <https://github.com/uvemas/ViTables>`_ is a component of the PyTables family. It is a GUI for browsing and editing files in both PyTables and HDF5 formats. It is an useful tool throughout the tutorial and it is recommended to install ViTables in a separated conda/mamba environment:

.. code-block:: bash

   mamba create -n vitables -c conda-forge python==3.12 vitables
   conda run -n vitables vitables hdf5_filepaths


CTAO open-source testdata
-------------------------

The tutorials use three different types of data. Some notebooks download very small test datasets directly. These datasets contain only a few events and are mainly intended for CI and automated testing, although we also use them in some examples. Other notebooks generate mock data to mimic observations from the LST-1 telescope, since the original LST-1 observational data are private to the collaboration.

For the more advanced tutorials, we use the publicly available CTAO open-source simulation data released on `Zenodo <https://zenodo.org/records/7298569>`_. Since the original data were produced with an older version of ctapipe, we took the official open-source dataset and reprocessed it with a recent `ctapipe` version. The resulting data are provided for the tutorials through the  `IAA Cloud <https://cloud.iaa.es/index.php/s/77d6MK4rGfanSKx>`_.

These data follow the CTAO data-model structure and can therefore be directly used with the standard ctapipe tools. They provide a realistic dataset for exploring the complete analysis workflow, while keeping in mind that the tutorial datasets and statistics are not intended for production-level or competitive physics results.
