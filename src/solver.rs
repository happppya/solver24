use arrayvec::ArrayVec;
use rayon::prelude::*;
use rustc_hash::{FxHashSet, FxHasher};
use std::hash::{Hash, Hasher};
use std::sync::atomic::{AtomicUsize, Ordering};

use crate::expr::{AstArena, Expr, ExprTree, Op};
use crate::operations::{apply_binary, apply_unary};

/// Granularity for numerical hashing to prevent non-associative precision drift.
const HASH_EPSILON: f64 = 10_000.0;

/// Hashes an individual value for permutation-invariant accumulation.
#[inline(always)]
fn hash_value(val: f64) -> u64 {
    let normalized = (val * HASH_EPSILON).round() as i64;
    let mut hasher = FxHasher::default();
    normalized.hash(&mut hasher);
    hasher.finish()
}

/// Generates a permutation-invariant hash of the active expression pool.
#[inline(always)]
fn hash_pool(pool: &[Expr]) -> u64 {
    pool.iter().fold(0_u64, |acc, expr| {
        // Wrapping add guarantees permutation invariance without O(N log N) sorting
        acc.wrapping_add(hash_value(expr.value.to_f64()))
    })
}

/// Generates a hash for a prospective binary operation state without allocating.
#[inline(always)]
fn hash_next_state_binary(pool: &[Expr], skip_i: usize, skip_j: usize, new_val: f64) -> u64 {
    let mut hash = hash_value(new_val);
    for (idx, expr) in pool.iter().enumerate() {
        if idx != skip_i && idx != skip_j {
            hash = hash.wrapping_add(hash_value(expr.value.to_f64()));
        }
    }
    hash
}

/// Generates a hash for a prospective unary operation state without allocating.
#[inline(always)]
fn hash_next_state_unary(pool: &[Expr], skip_i: usize, new_val: f64) -> u64 {
    let mut hash = hash_value(new_val);
    for (idx, expr) in pool.iter().enumerate() {
        if idx != skip_i {
            hash = hash.wrapping_add(hash_value(expr.value.to_f64()));
        }
    }
    hash
}

/// A highly parallelized, AST-generating search solver.
///
/// # Examples
/// ```
/// use arrayvec::ArrayVec;
/// use crate::expr::{AstArena, Op};
/// // Let `ops` be an array of supported operations and `pool` an initialized ArrayVec of Expr.
/// // let solver = Solver::new(24.0, 24.0, 24.0, &ops, None);
/// // let base_arena = AstArena::default();
/// // let results = solver.solve_parallel(pool, &base_arena);
/// ```
pub struct Solver<'a> {
    pub target: f64,
    pub target_min: f64,
    pub target_max: f64,
    pub operations: &'a [Op],
    comm_ops: Vec<Op>,
    non_comm_ops: Vec<Op>,
    pub max_unary_depth: u8,
    pub limit: Option<usize>,
}

impl<'a> Solver<'a> {
    /// Constructs a new Solver, pre-partitioning operations for optimized pipeline execution.
    pub fn new(
        target: f64,
        target_min: f64,
        target_max: f64,
        operations: &'a [Op],
        limit: Option<usize>,
    ) -> Self {
        let (comm_ops, non_comm_ops): (Vec<Op>, Vec<Op>) = operations
            .iter()
            .copied()
            .partition(|op| op.is_commutative());

        Solver {
            target,
            target_min,
            target_max,
            operations,
            comm_ops,
            non_comm_ops,
            max_unary_depth: 2,
            limit,
        }
    }

    /// Recursively resolves the expression permutations on a single thread.
    pub fn solve_pure(
        &self,
        pool: ArrayVec<Expr, 12>,
        cache: &mut FxHashSet<u64>,
        transient_arena: &mut AstArena,
        results_arena: &mut AstArena,
        counter: &AtomicUsize,
        local_steps: &mut usize,
    ) -> Vec<Expr> {
        // Mask check to batch atomic loads and prevent hardware cache-line contention
        *local_steps += 1;
        if *local_steps & 0x3FF == 0 {
            if let Some(max) = self.limit {
                if counter.load(Ordering::Relaxed) >= max {
                    return vec![];
                }
            }
        }

        if pool.len() == 1 {
            let val = pool[0].value.to_f64();
            if val >= self.target_min && val <= self.target_max {
                if let Some(max) = self.limit {
                    if counter.fetch_add(1, Ordering::Relaxed) >= max {
                        return vec![];
                    }
                }

                let mut sol = pool[0].clone();
                sol.tree_idx = results_arena.copy_from(transient_arena, sol.tree_idx);
                return vec![sol];
            }
            return vec![];
        }

        let state_hash = hash_pool(&pool);
        if !cache.insert(state_hash) {
            return vec![];
        }

        let mut solutions = Vec::new();
        let checkpoint = transient_arena.len();

        // Commutative operations
        for i in 0..pool.len() {
            for j in (i + 1)..pool.len() {
                for &op in &self.comm_ops {
                    if let Some(new_val) = apply_binary(op, &pool[i].value, &pool[j].value) {
                        if new_val.is_too_large() {
                            continue;
                        }

                        let next_hash = hash_next_state_binary(&pool, i, j, new_val.to_f64());
                        if cache.contains(&next_hash) {
                            continue;
                        }

                        let tree_idx = transient_arena.alloc(ExprTree::Node(
                            op,
                            pool[i].tree_idx,
                            Some(pool[j].tree_idx),
                        ));
                        let mut new_pool = pool.clone();
                        new_pool.swap_remove(j);
                        new_pool.swap_remove(i);
                        new_pool.push(Expr {
                            value: new_val,
                            tree_idx,
                            unary_depth: 0,
                        });

                        solutions.extend(self.solve_pure(
                            new_pool,
                            cache,
                            transient_arena,
                            results_arena,
                            counter,
                            local_steps,
                        ));
                        transient_arena.truncate(checkpoint);
                    }
                }
            }
        }

        // Non-commutative operations
        for i in 0..pool.len() {
            for j in 0..pool.len() {
                if i == j {
                    continue;
                }
                for &op in &self.non_comm_ops {
                    if let Some(new_val) = apply_binary(op, &pool[i].value, &pool[j].value) {
                        if new_val.is_too_large() {
                            continue;
                        }

                        let next_hash = hash_next_state_binary(&pool, i, j, new_val.to_f64());
                        if cache.contains(&next_hash) {
                            continue;
                        }

                        let tree_idx = transient_arena.alloc(ExprTree::Node(
                            op,
                            pool[i].tree_idx,
                            Some(pool[j].tree_idx),
                        ));
                        let mut new_pool = pool.clone();
                        let (max_idx, min_idx) = if i > j { (i, j) } else { (j, i) };
                        new_pool.swap_remove(max_idx);
                        new_pool.swap_remove(min_idx);
                        new_pool.push(Expr {
                            value: new_val,
                            tree_idx,
                            unary_depth: 0,
                        });

                        solutions.extend(self.solve_pure(
                            new_pool,
                            cache,
                            transient_arena,
                            results_arena,
                            counter,
                            local_steps,
                        ));
                        transient_arena.truncate(checkpoint);
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

                    let next_hash = hash_next_state_unary(&pool, i, new_val.to_f64());
                    if cache.contains(&next_hash) {
                        continue;
                    }

                    let tree_idx =
                        transient_arena.alloc(ExprTree::Node(op, pool[i].tree_idx, None));
                    let mut new_pool = pool.clone();
                    new_pool[i] = Expr {
                        value: new_val,
                        tree_idx,
                        unary_depth: pool[i].unary_depth + 1,
                    };

                    solutions.extend(self.solve_pure(
                        new_pool,
                        cache,
                        transient_arena,
                        results_arena,
                        counter,
                        local_steps,
                    ));
                    transient_arena.truncate(checkpoint);
                }
            }
        }

        solutions
    }

    /// Solves the target by chunking the initial combinatorial depth onto Rayon threads.
    /// Returns mapped batches to avoid O(N) arena clones per solution string hit.
    pub fn solve_parallel(
        &self,
        pool: ArrayVec<Expr, 12>,
        base_arena: &AstArena,
    ) -> Vec<(Vec<Expr>, AstArena)> {
        let mut first_gen_tasks = Vec::new();
        let counter = AtomicUsize::new(0);

        for i in 0..pool.len() {
            for j in (i + 1)..pool.len() {
                for &op in &self.comm_ops {
                    if let Some(new_val) = apply_binary(op, &pool[i].value, &pool[j].value) {
                        if new_val.is_too_large() {
                            continue;
                        }

                        let mut transient_arena = base_arena.clone();
                        let tree_idx = transient_arena.alloc(ExprTree::Node(
                            op,
                            pool[i].tree_idx,
                            Some(pool[j].tree_idx),
                        ));

                        let mut new_pool = pool.clone();
                        new_pool.swap_remove(j);
                        new_pool.swap_remove(i);
                        new_pool.push(Expr {
                            value: new_val,
                            tree_idx,
                            unary_depth: 0,
                        });

                        first_gen_tasks.push((new_pool, transient_arena));
                    }
                }
            }
        }

        for i in 0..pool.len() {
            for j in 0..pool.len() {
                if i == j {
                    continue;
                }
                for &op in &self.non_comm_ops {
                    if let Some(new_val) = apply_binary(op, &pool[i].value, &pool[j].value) {
                        if new_val.is_too_large() {
                            continue;
                        }

                        let mut transient_arena = base_arena.clone();
                        let tree_idx = transient_arena.alloc(ExprTree::Node(
                            op,
                            pool[i].tree_idx,
                            Some(pool[j].tree_idx),
                        ));

                        let mut new_pool = pool.clone();
                        let (max_idx, min_idx) = if i > j { (i, j) } else { (j, i) };
                        new_pool.swap_remove(max_idx);
                        new_pool.swap_remove(min_idx);
                        new_pool.push(Expr {
                            value: new_val,
                            tree_idx,
                            unary_depth: 0,
                        });

                        first_gen_tasks.push((new_pool, transient_arena));
                    }
                }
            }
        }

        first_gen_tasks
            .into_par_iter()
            .map(|(p, mut t_arena)| {
                let mut local_cache = FxHashSet::default();
                let mut results_arena = AstArena::default();
                let mut local_steps = 0;
                
                let results = self.solve_pure(
                    p,
                    &mut local_cache,
                    &mut t_arena,
                    &mut results_arena,
                    &counter,
                    &mut local_steps,
                );
                (results, results_arena)
            })
            .collect()
    }
}