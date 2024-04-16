.. meta::
  :description: MIOpen documentation
  :keywords: MIOpen, ROCm, API, documentation

********************************************************************
SmoothL1Loss Layer (experimental)
********************************************************************

Creates a criterion that uses a squared term if the absolute element-wise error falls below beta and an L1 term otherwise.

To enable this, define ``MIOPEN_BETA_API`` before including ``miopen.h``.

miopenGetSmoothL1LossWorkspaceSize

miopenSmoothL1LossForward
----------------------------------

.. doxygenfunction::  miopenGetSmoothL1LossWorkspaceSize
.. doxygenfunction::  miopenSmoothL1LossForward
