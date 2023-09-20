v-net.pytorch
-------------

.. image:: https://img.shields.io/badge/Python-3.10-3776AB.svg?style=flat&logo=python&logoColor=white
   :alt: Python 3.10
   :target: https://www.python.org

.. image:: https://img.shields.io/badge/PyTorch-2.0.0-EE4C2C.svg?style=flat&logo=pytorch
    :alt: PyTorch
    :target: https://pytorch.org

.. image:: https://img.shields.io/badge/license-MIT-green.svg
    :alt: License: MIT
    :target: https://choosealicense.com/licenses/mit/

.. image:: interrogate_badge.svg
   :alt: Interrogate Docstring Coverage
   :target: https://interrogate.readthedocs.io/en/latest/

.. image:: https://img.shields.io/badge/code%20style-black-000000.svg
   :alt: Code Style: Black
   :target: https://black.readthedocs.io/en/stable/


A PyTorch implementation of the medical segmentation model given in F. Milletari, N. Navab and S.A. Ahmadi's `"V-Net: Fully Convolutional Neural Networks for Volumetric Medical Image Segmentation" <https://arxiv.org/pdf/1606.04797.pdf>`_.


Usage
=====

To include this model in your code, simply copy and paste the contents of ``vnet.py`` to the your desired directory.

This implementation makes available some hyper-parameters that generalise the model. These are:

* ``in_channels (int)``: the number of channels of the input batch tensor. The expected input shape is (B, N, X, Y, Z) where B is the number of batches, N is the number of channels and X, Y, Z are three spatial dimensions.
* ``out_channels (int)``: the number of channels of the output segmentation mask. The expected output shape is (B, N, X, Y, Z) where B is the number of batches, N is the number of classes in the mask and X, Y, Z are the three spatial dimensions.
* ``depth (int)``: the number of up/downsamples.
* ``wf (int)``: base 2 of the number of output channels in the first convolutional block, the number of output channels in each down/upsampling unit will be doubled/halved thereafter.
* ``batch_norm (bool)``: whether to perform batch normalisation after each convolution―activation unit.
* ``activation (str)``: the name of the activation function, must be one of ``'elu'``, ``'relu'`` or ``'prelu'``. 
* ``loss (str)``: the name of the intended loss function, must be one of ``'dice'``, ``'nll'``, ``'none'`` or ``None``. If the Dice loss will be used (``'dice'``), then the logits will be passed through Softmax, if the negative log-likelihood will be used (``'nll'``), LogSoftmax will be applied, if the raw logits are required then the user should pass ``'none'`` or ``None``).
* ``kaiming_normal (bool)``: whether to initialise the weights of the convolutional layers with ``nn.init.kaiming_normal_``.


The hyper-parameters expected to recreate the model given in the paper will hence be:

.. code-block:: python

    model = VNet(
        in_channels=2,
        out_channels=2,
        depth=5,
        wf=4,
        batch_norm=False,
        activation='prelu',
        loss='dice',
        kaiming_normal=False,
    )






Requirements
===========

This model has been tested with Python 3.10.12 and PyTorch 2.0.0.


Building
========

Building this repository has dependencies other than those required by ``vnet.py``. To install them, run:

.. code-block:: console

    pip install -r requirements.txt


A ``Makefile`` has been provided to generate the ``torchinfo`` summary and the compute graph. Before performing a commit, run the following:

.. code-block:: console

    make clean && make


Pre-commit hooks have been provided to perform code quality checks on ``vnet.py``. Before performing a commit, install them by running:

.. code-block:: console

    pre-commit install


License
=======

This project is subject to the MIT license. For more details, view ``COPYING.rst``.

