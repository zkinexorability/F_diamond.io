# DiamondIO BGG Circuit Optimizer

A high-performance implementation of BGG (Boneh-Goh-Nissim) lattice-based cryptography for polynomial evaluation circuits.

## Overview

This repository contains an optimized implementation of lattice-based cryptographic operations focused on efficient polynomial circuit evaluation. It utilizes the Diamond I/O framework to build, evaluate, and optimize cryptographic circuits based on Ring Learning With Errors (RLWE) encryption.

## Features

- 🚀 High-performance polynomial operations using DCRT (Double Chinese Remainder Theorem)
- 🔄 Efficient base decomposition for RLWE ciphertexts
- 🧠 Parallel evaluation using Rayon for multi-threaded processing
- 🔑 BGG public key sampling with configurable parameters
- 📊 Memory usage tracking and optimization
- ⚡ Keccak-256 hash-based sampling

## Dependencies

- `diamond_io`: Core cryptographic operations and circuit definitions
- `keccak_asm`: Optimized Keccak-256 hash implementation
- `rand`: Random number generation
- `rayon`: Parallel computation utilities

## Usage

The core functionality is demonstrated in the `test_build_final_step_circuit` test:

```rust
// Initialize parameters with dimension 8192, 6 towers, 51-bit modulus, 20-bit base
let params = DCRTPolyParams::new(8192, 6, 51, 20);

// Build a simple AND-gate circuit for evaluation
let mut circuit = PolyCircuit::new();
// ...circuit definition...

// Generate encryption keys and ciphertext
// ...key generation and encryption...

// Build the final circuit for evaluation
let final_circuit = build_final_digits_circuit(
    &a_decomposed,
    &b_decomposed,
    circuit.clone()
);

// Evaluate the circuit in parallel
let eval_outputs = final_circuit.eval(&params, &pubkeys[0], &pubkeys[1..]);
```

## Performance Considerations

This implementation focuses on performance optimization:

- Minimizes memory allocations for large polynomial operations
- Uses parallel processing for matrix operations and output conversion
- Efficiently decomposes ciphertexts to reduce computational complexity
- Optimizes variable scope and naming for better readability
- Tracks memory usage at critical points with `log_mem`

## Installation

1. Add the following to your `Cargo.toml`:

```toml
[dependencies]
diamond_io = "0.1.0"
keccak_asm = "0.1.0"
rand = "0.8"
rayon = "1.5"
```

2. Import the necessary modules in your Rust file as shown in the example.

## Documentation

For more information on the underlying cryptographic primitives:

- BGG (Boneh-Goh-Nissim): A cryptographic scheme for homomorphic evaluation
- RLWE (Ring Learning With Errors): The foundation of lattice-based cryptography
- DCRT (Double Chinese Remainder Theorem): An optimization technique for polynomial operations

## Contributing

Contributions to improve performance, expand functionality, or fix bugs are welcome. Please ensure all tests pass before submitting pull requests.

## License

[MIT License](LICENSE)
