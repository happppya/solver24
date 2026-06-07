# 24Solver

This repository implements a high-performance, parallelized algebraic solver for a highly generalized variant of the classic "Make 24" mathematical puzzle. The architecture relies on contiguous memory arenas for zero-cost Abstract Syntax Tree (AST) allocations, aggressive state-hashing to prune isomorphic branches, and Rayon multithreading to brute-force the large solution space. After finding all solutions, the shortest 10 and longest 10 are shown.

## The Original 24 Game
The classic "Make 24" game is a mathematical puzzle where a player is given four numbers (typically integers from 1 to 9). The objective is to find a mathematical expression that evaluates to exactly 24. Players are strictly limited to using all four numbers exactly once and applying the four elementary arithmetic operations: addition ($+$), subtraction ($-$), multiplication ($\times$), and division ($\div$). Parentheses can be used to alter the order of operations.For example, given the inputs 4, 7, 8, 8, a valid solution is $(7 - (8 \div 8)) \times 4 = 24$.

## Usage

Compile and run in release mode for best performance
```
cargo build --release
```

Follow the execution syntax:
```
cargo run --release -- [FLAGS] [NUMBERS...]
```

**Configuration Flags:**
 - --error (FLOAT): Sets an absolute $\pm$ tolerance around the target (24.0).
 - --percent-error (FLOAT): Sets a relative percentage tolerance based on the target. Note: This is mutually exclusive with --error.
 - --limit (INTEGER): Imposes a hard ceiling on the number of generated abstract syntax trees to prevent unbounded memory exhaustion during wide permutations.

Example:
```
cargo run --release --percent-error 0.1 10 14 2 3
```
