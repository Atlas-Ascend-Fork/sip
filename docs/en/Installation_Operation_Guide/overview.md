# Overview

<!-- md-trans-meta sourceCommit=25a0c73916203ce4555696808917dc3c4c840c42 translatedAt=2026-10-10T12:22:49.354Z pushedAt=2026-10-14T09:51:08.187Z -->

The Ascend signal processing acceleration library (AscendSiPBoost) is designed for AI model scenarios (supporting PyTorch calls) and signal processing scenarios (supporting direct C++ calls). It provides high-performance operators in the signal processing domain, including BLAS (Basic Linear Algebra Subprograms), FFT (Fast Fourier Transform), complex number basic computation, and signal-domain fusion operators.

This document is intended to guide the installation and deployment of the signal processing acceleration library, and to provide example code based on typical use cases to help developers quickly become familiar with its usage.

## Architecture

The following figure shows the position of the AscendSiPBoost in the Ascend operator technology stack.
 ![Architecture Diagram](../API_Reference/figures/en-us_image_0000002568414177.png)

- Signal processing acceleration library framework: Responsible for operator management, binary loading of operators on the device side, and tiling on the host side; responsible for providing interfaces for single-operator and multi-operator batch calls, among others.
- FFT operators: Include dedicated NPU kernels and the PLAN framework, providing external interfaces to implement C2C, C2R, and R2C for developers to use.
- BLAS operators: Provide dedicated kernels in accordance with BLAS-related standard definitions, and offer external interfaces from level 1 to level 3 for developers to use.
- Complex number basic computation library: Provides basic operators that support complex number types, allowing users to use them in combinations as needed. Not provided in this release.
- Signal-domain fusion operator library: Includes fusion operators such as PC, MTD, CFAR, and Interpolation, supporting scenarios such as pulse signal analysis, moving target detection, and constant false alarm rate (CFAR). This release provides some interpolation operators.
- Solver operators: Primarily provides complex linear algebra functions based on BLAS, such as matrix decomposition and eigenvalue solving. Not provided in this release.

## Supported Architecture

  <!-- npu="910b" id1 -->
  <term>Atlas A2 products</term>
  <!-- end id1 -->

  <!-- npu="A3" id2 -->
  <term>Atlas A3 products</term>
  <!-- end id2 -->

  <!-- npu="950" id3 -->
  <term>Ascend 950PR/Ascend 950DT products</term>
  <!-- end id3 -->
