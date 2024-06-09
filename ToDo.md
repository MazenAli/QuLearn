- add input checking
- qlayer: remove item() calls for compatibility
- qlayer: handle 2D case where x or y are out of bounds
- qlayer: too much boiler plate code, reuse mps module
- pennylane non-default devices with lightning qubits, adjoint differentation and Hamiltonians
are bugged, the observable weights are not differentiated