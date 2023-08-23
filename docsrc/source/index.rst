.. cinnamon-tf

Cinnamon Tensorflow Package
================================================

The tensorflow package offers ``Component`` and related ``Configuration`` that rely on the Tensorflow library.

Thus, ``cinnamon-tf`` mainly provides ``Model``, ``Callback`` and ``Helper`` implementations.

===============================================
Components and Configurations
===============================================

The tensorflow package defines the following ``Component`` and ``Configuration``

- Neural networks: a high-level implementation to build your tensorflow-based neural models.
- Callbacks (e.g., early stopping)
- Framework helpers for tensorflow/cuda deterministic behaviour

===============================================
Install
===============================================

pip
   .. code-block:: bash

      pip install cinnamon-tf

git
   .. code-block:: bash

      git clone https://github.com/federicoruggeri/cinnamon_tf


.. toctree::
   :maxdepth: 4
   :hidden:
   :caption: Contents:
   :titlesonly:

    Model <model.rst>
    Callback <callback.rst>
    Helper <helper.rst>
    Catalog <catalog.rst>
    cinnamon-tf <cinnamon_tf.rst>