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

**NOTE**: the paper's results have not yet been recreated, hence do not expect it to work as intended.

Usage
=====

To include this model in your code, simply copy and paste the contents of ``vnet.py`` to your desired directory.

This implementation makes available some hyper-parameters that generalise the model. These are:

* ``in_channels (int)``: the number of channels of the input batch tensor. The expected input shape is (B, N, X, Y, Z) where B is the number of batches, N is the number of channels and X, Y, Z are three spatial dimensions.
* ``out_channels (int)``: the number of channels of the output segmentation mask. The expected output shape is (B, N, X, Y, Z) where B is the number of batches, N is the number of classes in the mask and X, Y, Z are the three spatial dimensions.
* ``depth (int)``: the number of up/downsamples.
* ``wf (int)``: base 2 of the number of channels which to multiply with the input number of channels to give the number of output channels after the first convolutional block, the number of output channels in each down/upsampling unit will be doubled/halved thereafter.
* ``batch_norm (bool)``: whether to perform batch normalisation after each convolution―activation unit.
* ``activation (str)``: the name of the activation function, must be one of ``'elu'``, ``'relu'`` or ``'prelu'``. 
* ``loss (str)``: the name of the intended loss function, must be one of ``'dice'``, ``'nll'``, ``'none'`` or ``None``. If the Dice loss will be used (``'dice'``), then the logits will be passed through Softmax, if the negative log-likelihood will be used (``'nll'``), LogSoftmax will be applied, if the raw logits are required then the user should pass ``'none'`` or ``None``).
* ``kaiming_normal (bool)``: whether to initialise the weights of the convolutional layers with ``nn.init.kaiming_normal_``.

The hyper-parameters expected to recreate the model given in the original paper will hence be:

.. code-block:: python

    original_model = VNet(
        in_channels=1,
        out_channels=2,
        depth=5,
        wf=4,
        batch_norm=False,
        activation='prelu',
        loss='dice',
        kaiming_normal=False,
    )


Model Summary
=============

The following table is the ``torchinfo`` report of ``original_model``.

.. code-block:: bash

      =================================================================================================================================================
      Layer (type:depth-idx)                        Input Shape               Output Shape              Param #                   Param %
      =================================================================================================================================================
      VNet                                          [4, 1, 128, 128, 64]      [4, 2, 128, 128, 64]      --                             --
      ├─ModuleList: 1-1                             --                        --                        --                             --
      │    └─VNetDownBlock: 2-1                     [4, 1, 128, 128, 64]      [4, 16, 128, 128, 64]     --                             --
      │    │    └─VNetNConvBlock: 3-1               [4, 1, 128, 128, 64]      [4, 16, 128, 128, 64]     --                             --
      │    │    │    └─Sequential: 4-1              [4, 1, 128, 128, 64]      [4, 16, 128, 128, 64]     --                             --
      │    │    │    │    └─Conv3d: 5-1             [4, 1, 128, 128, 64]      [4, 16, 128, 128, 64]     2,016                       0.00%
      │    │    │    │    └─PReLU: 5-2              [4, 16, 128, 128, 64]     [4, 16, 128, 128, 64]     16                          0.00%
      │    │    └─Sequential: 3-2                   [4, 16, 128, 128, 64]     [4, 32, 64, 64, 32]       --                             --
      │    │    │    └─Conv3d: 4-2                  [4, 16, 128, 128, 64]     [4, 32, 64, 64, 32]       4,128                       0.01%
      │    │    │    └─PReLU: 4-3                   [4, 32, 64, 64, 32]       [4, 32, 64, 64, 32]       32                          0.00%
      │    └─VNetDownBlock: 2-2                     [4, 32, 64, 64, 32]       [4, 32, 64, 64, 32]       --                             --
      │    │    └─VNetNConvBlock: 3-3               [4, 32, 64, 64, 32]       [4, 32, 64, 64, 32]       --                             --
      │    │    │    └─Sequential: 4-4              [4, 32, 64, 64, 32]       [4, 32, 64, 64, 32]       --                             --
      │    │    │    │    └─Conv3d: 5-3             [4, 32, 64, 64, 32]       [4, 32, 64, 64, 32]       128,032                     0.29%
      │    │    │    │    └─PReLU: 5-4              [4, 32, 64, 64, 32]       [4, 32, 64, 64, 32]       32                          0.00%
      │    │    │    │    └─Conv3d: 5-5             [4, 32, 64, 64, 32]       [4, 32, 64, 64, 32]       128,032                     0.29%
      │    │    │    │    └─PReLU: 5-6              [4, 32, 64, 64, 32]       [4, 32, 64, 64, 32]       32                          0.00%
      │    │    └─Sequential: 3-4                   [4, 32, 64, 64, 32]       [4, 64, 32, 32, 16]       --                             --
      │    │    │    └─Conv3d: 4-5                  [4, 32, 64, 64, 32]       [4, 64, 32, 32, 16]       16,448                      0.04%
      │    │    │    └─PReLU: 4-6                   [4, 64, 32, 32, 16]       [4, 64, 32, 32, 16]       64                          0.00%
      │    └─VNetDownBlock: 2-3                     [4, 64, 32, 32, 16]       [4, 64, 32, 32, 16]       --                             --
      │    │    └─VNetNConvBlock: 3-5               [4, 64, 32, 32, 16]       [4, 64, 32, 32, 16]       --                             --
      │    │    │    └─Sequential: 4-7              [4, 64, 32, 32, 16]       [4, 64, 32, 32, 16]       --                             --
      │    │    │    │    └─Conv3d: 5-7             [4, 64, 32, 32, 16]       [4, 64, 32, 32, 16]       512,064                     1.16%
      │    │    │    │    └─PReLU: 5-8              [4, 64, 32, 32, 16]       [4, 64, 32, 32, 16]       64                          0.00%
      │    │    │    │    └─Conv3d: 5-9             [4, 64, 32, 32, 16]       [4, 64, 32, 32, 16]       512,064                     1.16%
      │    │    │    │    └─PReLU: 5-10             [4, 64, 32, 32, 16]       [4, 64, 32, 32, 16]       64                          0.00%
      │    │    │    │    └─Conv3d: 5-11            [4, 64, 32, 32, 16]       [4, 64, 32, 32, 16]       512,064                     1.16%
      │    │    │    │    └─PReLU: 5-12             [4, 64, 32, 32, 16]       [4, 64, 32, 32, 16]       64                          0.00%
      │    │    └─Sequential: 3-6                   [4, 64, 32, 32, 16]       [4, 128, 16, 16, 8]       --                             --
      │    │    │    └─Conv3d: 4-8                  [4, 64, 32, 32, 16]       [4, 128, 16, 16, 8]       65,664                      0.15%
      │    │    │    └─PReLU: 4-9                   [4, 128, 16, 16, 8]       [4, 128, 16, 16, 8]       128                         0.00%
      │    └─VNetDownBlock: 2-4                     [4, 128, 16, 16, 8]       [4, 128, 16, 16, 8]       --                             --
      │    │    └─VNetNConvBlock: 3-7               [4, 128, 16, 16, 8]       [4, 128, 16, 16, 8]       --                             --
      │    │    │    └─Sequential: 4-10             [4, 128, 16, 16, 8]       [4, 128, 16, 16, 8]       --                             --
      │    │    │    │    └─Conv3d: 5-13            [4, 128, 16, 16, 8]       [4, 128, 16, 16, 8]       2,048,128                   4.65%
      │    │    │    │    └─PReLU: 5-14             [4, 128, 16, 16, 8]       [4, 128, 16, 16, 8]       128                         0.00%
      │    │    │    │    └─Conv3d: 5-15            [4, 128, 16, 16, 8]       [4, 128, 16, 16, 8]       2,048,128                   4.65%
      │    │    │    │    └─PReLU: 5-16             [4, 128, 16, 16, 8]       [4, 128, 16, 16, 8]       128                         0.00%
      │    │    │    │    └─Conv3d: 5-17            [4, 128, 16, 16, 8]       [4, 128, 16, 16, 8]       2,048,128                   4.65%
      │    │    │    │    └─PReLU: 5-18             [4, 128, 16, 16, 8]       [4, 128, 16, 16, 8]       128                         0.00%
      │    │    └─Sequential: 3-8                   [4, 128, 16, 16, 8]       [4, 256, 8, 8, 4]         --                             --
      │    │    │    └─Conv3d: 4-11                 [4, 128, 16, 16, 8]       [4, 256, 8, 8, 4]         262,400                     0.60%
      │    │    │    └─PReLU: 4-12                  [4, 256, 8, 8, 4]         [4, 256, 8, 8, 4]         256                         0.00%
      ├─ModuleList: 1-2                             --                        --                        --                             --
      │    └─VNetUpBlock: 2-5                       [4, 256, 8, 8, 4]         [4, 128, 16, 16, 8]       --                             --
      │    │    └─VNetNConvBlock: 3-9               [4, 256, 8, 8, 4]         [4, 256, 8, 8, 4]         --                             --
      │    │    │    └─Sequential: 4-13             [4, 256, 8, 8, 4]         [4, 256, 8, 8, 4]         --                             --
      │    │    │    │    └─Conv3d: 5-19            [4, 256, 8, 8, 4]         [4, 256, 8, 8, 4]         8,192,256                  18.61%
      │    │    │    │    └─PReLU: 5-20             [4, 256, 8, 8, 4]         [4, 256, 8, 8, 4]         256                         0.00%
      │    │    │    │    └─Conv3d: 5-21            [4, 256, 8, 8, 4]         [4, 256, 8, 8, 4]         8,192,256                  18.61%
      │    │    │    │    └─PReLU: 5-22             [4, 256, 8, 8, 4]         [4, 256, 8, 8, 4]         256                         0.00%
      │    │    │    │    └─Conv3d: 5-23            [4, 256, 8, 8, 4]         [4, 256, 8, 8, 4]         8,192,256                  18.61%
      │    │    │    │    └─PReLU: 5-24             [4, 256, 8, 8, 4]         [4, 256, 8, 8, 4]         256                         0.00%
      │    │    └─Sequential: 3-10                  [4, 256, 8, 8, 4]         [4, 128, 16, 16, 8]       --                             --
      │    │    │    └─ConvTranspose3d: 4-14        [4, 256, 8, 8, 4]         [4, 128, 16, 16, 8]       262,272                     0.60%
      │    │    │    └─PReLU: 4-15                  [4, 128, 16, 16, 8]       [4, 128, 16, 16, 8]       128                         0.00%
      │    └─VNetUpBlock: 2-6                       [4, 128, 16, 16, 8]       [4, 64, 32, 32, 16]       --                             --
      │    │    └─VNetNConvBlock: 3-11              [4, 256, 16, 16, 8]       [4, 128, 16, 16, 8]       --                             --
      │    │    │    └─Sequential: 4-16             [4, 256, 16, 16, 8]       [4, 128, 16, 16, 8]       --                             --
      │    │    │    │    └─Conv3d: 5-25            [4, 256, 16, 16, 8]       [4, 128, 16, 16, 8]       4,096,128                   9.30%
      │    │    │    │    └─PReLU: 5-26             [4, 128, 16, 16, 8]       [4, 128, 16, 16, 8]       128                         0.00%
      │    │    │    │    └─Conv3d: 5-27            [4, 128, 16, 16, 8]       [4, 128, 16, 16, 8]       2,048,128                   4.65%
      │    │    │    │    └─PReLU: 5-28             [4, 128, 16, 16, 8]       [4, 128, 16, 16, 8]       128                         0.00%
      │    │    │    │    └─Conv3d: 5-29            [4, 128, 16, 16, 8]       [4, 128, 16, 16, 8]       2,048,128                   4.65%
      │    │    │    │    └─PReLU: 5-30             [4, 128, 16, 16, 8]       [4, 128, 16, 16, 8]       128                         0.00%
      │    │    └─Sequential: 3-12                  [4, 128, 16, 16, 8]       [4, 64, 32, 32, 16]       --                             --
      │    │    │    └─ConvTranspose3d: 4-17        [4, 128, 16, 16, 8]       [4, 64, 32, 32, 16]       65,600                      0.15%
      │    │    │    └─PReLU: 4-18                  [4, 64, 32, 32, 16]       [4, 64, 32, 32, 16]       64                          0.00%
      │    └─VNetUpBlock: 2-7                       [4, 64, 32, 32, 16]       [4, 32, 64, 64, 32]       --                             --
      │    │    └─VNetNConvBlock: 3-13              [4, 128, 32, 32, 16]      [4, 64, 32, 32, 16]       --                             --
      │    │    │    └─Sequential: 4-19             [4, 128, 32, 32, 16]      [4, 64, 32, 32, 16]       --                             --
      │    │    │    │    └─Conv3d: 5-31            [4, 128, 32, 32, 16]      [4, 64, 32, 32, 16]       1,024,064                   2.33%
      │    │    │    │    └─PReLU: 5-32             [4, 64, 32, 32, 16]       [4, 64, 32, 32, 16]       64                          0.00%
      │    │    │    │    └─Conv3d: 5-33            [4, 64, 32, 32, 16]       [4, 64, 32, 32, 16]       512,064                     1.16%
      │    │    │    │    └─PReLU: 5-34             [4, 64, 32, 32, 16]       [4, 64, 32, 32, 16]       64                          0.00%
      │    │    │    │    └─Conv3d: 5-35            [4, 64, 32, 32, 16]       [4, 64, 32, 32, 16]       512,064                     1.16%
      │    │    │    │    └─PReLU: 5-36             [4, 64, 32, 32, 16]       [4, 64, 32, 32, 16]       64                          0.00%
      │    │    └─Sequential: 3-14                  [4, 64, 32, 32, 16]       [4, 32, 64, 64, 32]       --                             --
      │    │    │    └─ConvTranspose3d: 4-20        [4, 64, 32, 32, 16]       [4, 32, 64, 64, 32]       16,416                      0.04%
      │    │    │    └─PReLU: 4-21                  [4, 32, 64, 64, 32]       [4, 32, 64, 64, 32]       32                          0.00%
      │    └─VNetUpBlock: 2-8                       [4, 32, 64, 64, 32]       [4, 16, 128, 128, 64]     --                             --
      │    │    └─VNetNConvBlock: 3-15              [4, 64, 64, 64, 32]       [4, 32, 64, 64, 32]       --                             --
      │    │    │    └─Sequential: 4-22             [4, 64, 64, 64, 32]       [4, 32, 64, 64, 32]       --                             --
      │    │    │    │    └─Conv3d: 5-37            [4, 64, 64, 64, 32]       [4, 32, 64, 64, 32]       256,032                     0.58%
      │    │    │    │    └─PReLU: 5-38             [4, 32, 64, 64, 32]       [4, 32, 64, 64, 32]       32                          0.00%
      │    │    │    │    └─Conv3d: 5-39            [4, 32, 64, 64, 32]       [4, 32, 64, 64, 32]       128,032                     0.29%
      │    │    │    │    └─PReLU: 5-40             [4, 32, 64, 64, 32]       [4, 32, 64, 64, 32]       32                          0.00%
      │    │    │    │    └─Conv3d: 5-41            [4, 32, 64, 64, 32]       [4, 32, 64, 64, 32]       128,032                     0.29%
      │    │    │    │    └─PReLU: 5-42             [4, 32, 64, 64, 32]       [4, 32, 64, 64, 32]       32                          0.00%
      │    │    └─Sequential: 3-16                  [4, 32, 64, 64, 32]       [4, 16, 128, 128, 64]     --                             --
      │    │    │    └─ConvTranspose3d: 4-23        [4, 32, 64, 64, 32]       [4, 16, 128, 128, 64]     4,112                       0.01%
      │    │    │    └─PReLU: 4-24                  [4, 16, 128, 128, 64]     [4, 16, 128, 128, 64]     16                          0.00%
      ├─VNetOutputBlock: 1-3                        [4, 16, 128, 128, 64]     [4, 2, 128, 128, 64]      --                             --
      │    └─VNetNConvBlock: 2-9                    [4, 32, 128, 128, 64]     [4, 16, 128, 128, 64]     --                             --
      │    │    └─Sequential: 3-17                  [4, 32, 128, 128, 64]     [4, 16, 128, 128, 64]     --                             --
      │    │    │    └─Conv3d: 4-25                 [4, 32, 128, 128, 64]     [4, 16, 128, 128, 64]     64,016                      0.15%
      │    │    │    └─PReLU: 4-26                  [4, 16, 128, 128, 64]     [4, 16, 128, 128, 64]     16                          0.00%
      │    └─Sequential: 2-10                       [4, 16, 128, 128, 64]     [4, 2, 128, 128, 64]      --                             --
      │    │    └─Conv3d: 3-18                      [4, 16, 128, 128, 64]     [4, 2, 128, 128, 64]      34                          0.00%
      │    │    └─PReLU: 3-19                       [4, 2, 128, 128, 64]      [4, 2, 128, 128, 64]      2                           0.00%
      =================================================================================================================================================
      Total params: 44,032,020
      Trainable params: 44,032,020
      Non-trainable params: 0
      Total mult-adds (T): 1.09
      =================================================================================================================================================
      Input size (MB): 16.78
      Forward/backward pass size (MB): 5922.36
      Params size (MB): 176.13
      Estimated Total Size (MB): 6115.26
      =================================================================================================================================================



Requirements
===========

This model has been developed with Python 3.10.12 and PyTorch 2.0.0.


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

