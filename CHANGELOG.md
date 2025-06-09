# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.8.1] - 2025-06-09

### Added

- Add documentation
- Publish to PyPI

## [0.8.0] - 2024-07-20

### Added

- Add Linear2DBasisQFE

### Chore

- Move type aliases to a separate module
- Update Makefile
- Add github workflows

## [0.7.0] - 2024-02-06

### Added

- Add TwoQubitMPS and EmbedU layers

## [0.6.2] - 2024-02-06

### Fixed

- Hat basis negative close to 0 values, sqrt in HatBasisQFE returned nan

## [0.6.1] - 2024-02-05

### Fixed

- MPS contraction

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
