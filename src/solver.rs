use arrayvec::ArrayVec;
use rayon::prelude::*;
use std::collections::{HashSet, hash_map::DefaultHasher};
use std::hash::{Hash, Hasher};
use std::sync::Arc;

use crate::expr::{Expr, ExprTree, Op};
use crate::operations::{apply_binary, apply_unary};

/// Stack-allocated hashing.
#[inline(always)]
fn hash_pool(pool: &[Expr]) -> u64 {
    let mut values = [0_i64; 12]; // Stack array avoids heap allocations
    let len = pool.len().min(12);

    for i in 0..len {
        // Quantize the float to 6 decimal places to bypass floating-point bit jitter
        values[i] = (pool[i].value.to_f64() * 1_000_000.0).round() as i64;
    }

    let active_slice = &mut values[..len];
    active_slice.sort_unstable(); // Order independent

    let mut hasher = DefaultHasher::new();
    active_slice.hash(&mut hasher);
    hasher.finish()
}

pub struct Solver<'a> {
    pub target: f64,
    pub target_min: f64,
    pub target_max: f64,
    pub operations: &'a [Op],
    pub max_unary_depth: u8,
}

impl<'a> Solver<'a> {
    pub fn new(target: f64, target_min: f64, target_max: f64, operations: &'a [Op]) -> Self {
        Solver {
            target,
            target_min,
            target_max,
            operations,
            max_unary_depth: 2,
        }
    }

    /// Exhaustive Pure DFS with memoization
    pub fn solve_pure(&self, pool: ArrayVec<Expr, 12>, cache: &mut HashSet<u64>) -> Vec<Expr> {
        // Fast range search base case
        if pool.len() == 1 {
            let val = pool[0].value.to_f64();
            if val >= self.target_min && val <= self.target_max {
                return vec![pool[0].clone()];
            }
            return vec![];
        }

        let state_hash = hash_pool(&pool);
        if !cache.insert(state_hash) {
            return vec![];
        }

        let mut solutions = Vec::new();

        // Binary operations
        for i in 0..pool.len() {
            for j in 0..pool.len() {
                if i == j {
                    continue;
                }

                for &op in self.operations {
                    if op.is_commutative() && i > j {
                        continue;
                    }

                    if let Some(new_val) = apply_binary(op, &pool[i].value, &pool[j].value) {
                        if new_val.is_too_large() {
                            continue;
                        }

                        let mut new_pool = ArrayVec::<Expr, 12>::new();
                        new_pool.push(Expr {
                            value: new_val,
                            tree: Arc::new(ExprTree::Node(
                                op,
                                pool[i].tree.clone(),
                                Some(pool[j].tree.clone()),
                            )),
                            unary_depth: 0,
                        });

                        for k in 0..pool.len() {
                            if k != i && k != j {
                                new_pool.push(pool[k].clone());
                            }
                        }

                        solutions.extend(self.solve_pure(new_pool, cache));
                    }
                }
            }
        }

        // Unary operations
        for i in 0..pool.len() {
            if pool[i].unary_depth >= self.max_unary_depth {
                continue;
            }

            for &op in self.operations {
                if let Some(new_val) = apply_unary(op, &pool[i].value) {
                    if new_val.is_too_large() {
                        continue;
                    }

                    let mut new_pool = pool.clone();
                    new_pool[i] = Expr {
                        value: new_val,
                        tree: Arc::new(ExprTree::Node(op, pool[i].tree.clone(), None)),
                        unary_depth: pool[i].unary_depth + 1,
                    };

                    solutions.extend(self.solve_pure(new_pool, cache));
                }
            }
        }

        solutions
    }

    /// Parallel entry point
    pub fn solve_parallel(&self, pool: ArrayVec<Expr, 12>) -> Vec<Expr> {
        let mut first_gen_pools = Vec::new();

        for i in 0..pool.len() {
            for j in 0..pool.len() {
                if i == j {
                    continue;
                }
                for &op in self.operations {
                    if op.is_commutative() && i > j {
                        continue;
                    }

                    if let Some(new_val) = apply_binary(op, &pool[i].value, &pool[j].value) {
                        if new_val.is_too_large() {
                            continue;
                        }

                        let mut new_pool = ArrayVec::<Expr, 12>::new();
                        new_pool.push(Expr {
                            value: new_val,
                            tree: Arc::new(ExprTree::Node(op, pool[i].tree.clone(), Some(pool[j].tree.clone()))),
                            unary_depth: 0,
                        });

                        for k in 0..pool.len() {
                            if k != i && k != j {
                                new_pool.push(pool[k].clone());
                            }
                        }
                        first_gen_pools.push(new_pool);
                    }
                }
            }
        }

        // Parallel execution collecting all valid branches
        first_gen_pools
            .into_par_iter()
            .flat_map(|p| {
                let mut local_cache = HashSet::new(); // Give each thread its own cache
                self.solve_pure(p, &mut local_cache)
            })
            .collect()
    }
}
