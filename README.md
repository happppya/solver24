# 24Solver

This repository implements a high-performance, parallelized algebraic solver for a highly generalized variant of the classic "Make 24" mathematical puzzle. The architecture relies on contiguous memory arenas for zero-cost Abstract Syntax Tree (AST) allocations, aggressive state-hashing to prune isomorphic branches, and Rayon multithreading to brute-force the large solution space. After finding all solutions, the shortest 10 and longest 10 are shown.

## The Original 24 Game
The classic "Make 24" game is a mathematical puzzle where a player is given four numbers (typically integers from 1 to 9). The objective is to find a mathematical expression that evaluates to exactly 24. Players are strictly limited to using all four numbers exactly once and applying the four elementary arithmetic operations: addition ($+$), subtraction ($-$), multiplication ($\times$), and division ($\div$). Parentheses can be used to alter the order of operations.For example, given the inputs 4, 7, 8, 8, a valid solution is $(7 - (8 \div 8)) \times 4 = 24$.

## Usage

Compile and run in release mode for best performance:
```bash
cargo build --release
```

Follow the execution syntax:
```bash
cargo run --release -- [FLAGS] [NUMBERS...]
```

**Configuration Flags:**

* `-fast` or `--fast`: Activates Priority Queue-driven Best-First Search. Short-circuits and instantly halts execution upon finding the first valid abstract syntax tree.
* `--error <FLOAT>`: Sets an absolute ± tolerance around the target.
* `--percent-error <FLOAT>`: Sets a relative percentage tolerance based on the target. *Note: This is mutually exclusive with `--error`.*
* `--limit <INTEGER>`: Imposes a hard ceiling on the number of generated abstract syntax trees to prevent unbounded memory exhaustion during wide permutations.

**Examples:**

Run an exhaustive search with a relative error margin:
```bash
cargo run --release -- --percent-error 0.1 10 14 2 3
```

Run in fast mode to instantly find a single exact solution:
```bash
cargo run --release -- -fast 7 7 7 7
```