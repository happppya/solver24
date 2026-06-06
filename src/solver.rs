use arrayvec::ArrayVec;
use rayon::prelude::*;
use rustc_hash::{FxHashSet, FxHasher};
use std::hash::{Hash, Hasher};
use std::sync::atomic::{AtomicUsize, Ordering};

use crate::operations::{apply_binary, apply_unary};
use crate::types::{AstArena, Expr, ExprTree, Op};

/// Granularity for numerical hashing to prevent non-associative precision drift.
const HASH_EPSILON: f64 = 10_000.0;

#[inline(always)]
fn hash_value(val: f64) -> u64 {
    let normalized = (val * HASH_EPSILON).round() as i64;
    let mut hasher = FxHasher::default();
    normalized.hash(&mut hasher);
    hasher.finish()
}

#[inline(always)]
fn hash_pool(pool: &[Expr]) -> u64 {
    pool.iter().fold(0_u64, |acc, expr| {
        acc.wrapping_add(hash_value(expr.value.to_f64()))
    })
}

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

/// A highly parallelized, AST-generating search solver with cyclic branch pruning.
///
/// # Examples
/// ```
/// use arrayvec::ArrayVec;
/// use crate::expr::{AstArena, Op};
/// // let solver = Solver::new(24.0, 24.0, 24.0, &ops, None);
/// // let base_arena = AstArena::default();
/// // let results = solver.solve_parallel(pool, &base_arena);
/// ```
pub struct Solver<> {
    pub target: f64,
    pub target_min: f64,
    pub target_max: f64,
    comm_ops: Vec<Op>,
    non_comm_ops: Vec<Op>,
    unary_ops: Vec<Op>,
    pub limit: Option<usize>,
}

impl<'a> Solver<> {
    /// Constructs a new Solver, pre-partitioning operations by arity and commutativity.
    pub fn new(
        target: f64,
        target_min: f64,
        target_max: f64,
        operations: &'a [Op],
        limit: Option<usize>,
    ) -> Self {
        let unary_ops: Vec<Op> = operations
            .iter()
            .copied()
            .filter(|op| op.is_unary())
            .collect();
        let binary_ops = operations.iter().copied().filter(|op| !op.is_unary());

        let (comm_ops, non_comm_ops): (Vec<Op>, Vec<Op>) =
            binary_ops.partition(|op| op.is_commutative());

        Solver {
            target,
            target_min,
            target_max,
            comm_ops,
            non_comm_ops,
            unary_ops,
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

        // Commutative binary operations
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
                            unary_mask: 0,
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

        // Non-commutative binary operations
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
                            unary_mask: 0,
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

        // Unary operations: Restricted exclusively to the most recently generated node
        // preventing permutation explosion on independent pool variables.
        let new_node_idx = pool.len() - 1;
        for &op in &self.unary_ops {
            let op_mask = op.family_mask();
            // Prune inverse operations and chained asymptotic families
            if (pool[new_node_idx].unary_mask & op_mask) != 0 {
                continue;
            }

            if let Some(new_val) = apply_unary(op, &pool[new_node_idx].value) {
                if new_val.is_too_large() {
                    continue;
                }

                let next_hash = hash_next_state_unary(&pool, new_node_idx, new_val.to_f64());
                if cache.contains(&next_hash) {
                    continue;
                }

                let tree_idx =
                    transient_arena.alloc(ExprTree::Node(op, pool[new_node_idx].tree_idx, None));
                let mut new_pool = pool.clone();
                new_pool[new_node_idx] = Expr {
                    value: new_val,
                    tree_idx,
                    unary_mask: pool[new_node_idx].unary_mask | op_mask,
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

        solutions
    }

    /// Solves the target by chunking the initial combinatorial depth onto Rayon threads.
    pub fn solve_parallel(
        &self,
        pool: ArrayVec<Expr, 12>,
        base_arena: &AstArena,
    ) -> Vec<(Vec<Expr>, AstArena)> {
        let mut first_gen_tasks = Vec::new();
        let counter = AtomicUsize::new(0);

        // Initial pool unary tasks
        for i in 0..pool.len() {
            for &op in &self.unary_ops {
                let op_mask = op.family_mask();
                if (pool[i].unary_mask & op_mask) != 0 {
                    continue;
                }

                if let Some(new_val) = apply_unary(op, &pool[i].value) {
                    if new_val.is_too_large() {
                        continue;
                    }

                    let mut transient_arena = base_arena.clone();
                    let tree_idx =
                        transient_arena.alloc(ExprTree::Node(op, pool[i].tree_idx, None));

                    let mut new_pool = pool.clone();
                    // Shift processed node to the end to align with `solve_pure` expectations
                    let mut modified_expr = new_pool.swap_remove(i);
                    modified_expr.value = new_val;
                    modified_expr.tree_idx = tree_idx;
                    modified_expr.unary_mask |= op_mask;
                    new_pool.push(modified_expr);

                    first_gen_tasks.push((new_pool, transient_arena));
                }
            }
        }

        // Initial pool commutative binary tasks
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
                            unary_mask: 0,
                        });

                        first_gen_tasks.push((new_pool, transient_arena));
                    }
                }
            }
        }

        // Initial pool non-commutative binary tasks
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
                            unary_mask: 0,
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
