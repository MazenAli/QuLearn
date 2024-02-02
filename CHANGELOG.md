# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.6.0] - 2024-02-02

### Added

- Hat basis modules
- MPS modules
- Hat basis QFE to qlayer

## [0.5.1] - 2023-11-21

### Fixed

- Output shape for batched inputs consistent with PyTorch

## [0.5.0] - 2023-11-16

### Added

- Added variable exponent base to data embeddings
- Added multiple observables option to Measurement Layer

## [0.4.0] - 2023-09-18

### Added

- Added quantum kernel models and ridge regression trainers

### Changed

- Changed model builders and configs in examples

## [0.3.0] - 2023-06-24

### Changed

- Refactored and simplified trainer, better logging

## [0.2.0] - 2023-06-22

### Added

- Naive (non-efficient) estimation of Fisher matrices
- Estimation of effective dimension (see arXiv:2112.04807)

## [0.1.0] - 2023-06-02

### Added

- Hybrid model functionality.
- Capacity metrics estimation routines.
- Synthetic data generation methods.
- Trainers.
- Basic unit tests for all the features.
- Initial documentation for the API.
