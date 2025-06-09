Installation
============

Requirements
------------

QuLearn requires Python 3.9 or higher. The package has been tested on Python versions 3.9, 3.10, and 3.11.

Dependencies
------------

The main dependencies include:

* PyTorch (2.0)
* TensorFlow (2.12.0)
* PennyLane (0.28.0)
* NumPy
* SciPy
* Matplotlib
* scikit-learn

Installation Methods
---------------------

Using pip
~~~~~~~~~

You can install QuLearn using pip:

.. code-block:: bash

    pip install qulearn

Using Poetry
~~~~~~~~~~~~

If you prefer using Poetry:

.. code-block:: bash

    poetry add qulearn

From Source
~~~~~~~~~~~

To install from source:

.. code-block:: bash

    git clone https://github.com/MazenAli/qulearn.git
    cd qulearn
    poetry install

Development Installation
~~~~~~~~~~~~~~~~~~~~~~~~~

For development installation with all dependencies:

.. code-block:: bash

    poetry install --with dev,test,docs

Verifying Installation
----------------------

You can verify the installation by running:

.. code-block:: python

    import qulearn
    print(qulearn.__version__) 