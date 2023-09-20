vnet.pytorch
------------

A PyTorch implementation of the medical segmentation model given in F. Milletari, N. Navab and S.A. Ahmadi's "V-Net: Fully Convolutional Neural Networks for Volumetric Medical Image Segmentation" (https://arxiv.org/pdf/1606.04797.pdf).

This implementation makes available some hyper-parameters that generalises the model. These are:

* ``in_channels (int)``: the number of channels of the input batch tensor. The expected input shape is (B, N, X, Y, Z) where B is the number of batches, N is the number of channels and X, Y, Z are three spatial dimensions.
* ``out_channels (int)``: the number of channels of the output segmentation mask. The expected output shape is (B, N, X, Y, Z) where B is the number of batches, N is the number of classes in the mask and X, Y, Z are the three spatial dimensions.
* ``depth``: the number of up/downsamples.
* ``wf``: base 2 of the number of output channels in the first convolutional block, the number of output channels in each down/upsampling unit will be doubled/halved thereafter.
* ``activation``: the name of the activation function, must be one of ``elu``, ``relu`` or ``prelu``. 
* ``loss``: the name of the intended loss function, must be one of ``dice`` or ``nll``. If the Dice loss will be used, then the logits will be passed through Softmax, otherwise LogSoftmax will be applied in preparation for the Negative Log-likelihood.

The hyper-parameters expected to recreate the model given in the paper will hence be:

    model = VNet(
        in_channels=2,
        out_channels=2,
        depth=5,
        wf=4,
        activation='prelu',
        loss='dice'
    )


Usage
=====

To include this model in your code, simply copy and paste the contents of `vnet.py` to the your desired directory.


Requirements
===========

This model has been tested with Python 3.10.12 and PyTorch 2.0.0.


Building
========

Building this repository has dependencies other than those required by `vnet.py`. To install them, run:

    pip install -r requirements.txt


A ``Makefile`` has been provided to generate the `torchinfo` summary and the compute graph. Before performing a commit, run the following:

    make clean && make


Pre-commit hooks have been provided to perform code quality checks on `vnet.py`. Before performing a commit, install them by running:

    pre-commit install


License
=======

This project is subject to the MIT license. For more details, view `COPYING.rst`.




