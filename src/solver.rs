use arrayvec::ArrayVec;
use rayon::prelude::*;
use rustc_hash::FxHashSet;
use std::cell::RefCell;
use std::cmp::Ordering as CmpOrdering;
use std::collections::BinaryHeap;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

use crate::operations::{apply_binary, apply_unary};
use crate::types::{AstArena, Expr, ExprTree, Op};

/// Granularity for numerical hashing to prevent non-associative precision drift.
const HASH_EPSILON: f64 = 10_000.0;

/// Fast integer mix equivalent to FxHash for a single scalar. 
#[inline(always)]
fn hash_value(val: f64) -> u64 {
    let normalized = (val * HASH_EPSILON).round() as i64;
    (normalized as u64).wrapping_mul(0x517cc1b727220a95)
}

#[inline(always)]
fn hash_pool(pool: &[Expr]) -> u64 {
    pool.iter().fold(0_u64, |acc, expr| {
        acc.wrapping_add(hash_value(expr.value.to_f64()))
    })
}

#[inline(always)]
fn hash_update_binary(base_hash: u64, val_i: f64, val_j: f64, new_val: f64) -> u64 {
    base_hash
        .wrapping_sub(hash_value(val_i))
        .wrapping_sub(hash_value(val_j))
        .wrapping_add(hash_value(new_val))
}

#[inline(always)]
fn hash_update_unary(base_hash: u64, old_val: f64, new_val: f64) -> u64 {
    base_hash
        .wrapping_sub(hash_value(old_val))
        .wrapping_add(hash_value(new_val))
}

/// A prioritized state wrapper for the Best-First Search algorithm.
#[derive(Clone)]
struct SearchState {
    pool: ArrayVec<Expr, 12>,
    arena: AstArena,
    hash: u64,
    heuristic_score: i64,
}

impl PartialEq for SearchState {
    fn eq(&self, other: &Self) -> bool {
        self.heuristic_score == other.heuristic_score
    }
}
impl Eq for SearchState {}

impl PartialOrd for SearchState {
    fn partial_cmp(&self, other: &Self) -> Option<CmpOrdering> {
        Some(self.cmp(other))
    }
}

impl Ord for SearchState {
    fn cmp(&self, other: &Self) -> CmpOrdering {
        // Reverses the comparison to operate BinaryHeap as a min-heap.
        other.heuristic_score.cmp(&self.heuristic_score)
    }
}

/// A highly parallelized, AST-generating search solver with cyclic branch pruning.
///
/// # Examples
/// ```
/// use arrayvec::ArrayVec;
/// use crate::types::{AstArena, Op};
/// // let solver = Solver::new(24.0, 24.0, 24.0, &ops, None);
/// // let base_arena = AstArena::default();
/// // let exhaustive_results = solver.solve_parallel(pool.clone(), &base_arena);
/// // let fast_result = solver.find_first(pool, &base_arena);
/// ```
pub struct Solver {
    pub target: f64,
    pub target_min: f64,
    pub target_max: f64,
    comm_ops: Vec<Op>,
    non_comm_ops: Vec<Op>,
    unary_ops: Vec<Op>,
    pub limit: Option<usize>,
}

impl Solver {
    /// Constructs a new Solver, pre-partitioning operations by arity and commutativity.
    pub fn new(
        target: f64,
        target_min: f64,
        target_max: f64,
        operations: &[Op],
        limit: Option<usize>,
    ) -> Self {
        let unary_ops: Vec<Op> = operations.iter().copied().filter(|op| op.is_unary()).collect();
        let binary_ops = operations.iter().copied().filter(|op| !op.is_unary());
        let (comm_ops, non_comm_ops): (Vec<Op>, Vec<Op>) =
            binary_ops.partition(|op| op.is_commutative());

        Self {
            target,
            target_min,
            target_max,
            comm_ops,
            non_comm_ops,
            unary_ops,
            limit,
        }
    }

    /// Evaluates the distance of a given state pool from the target.
    /// Represents the heuristic cost function $h(x) = \min_{v \in x} |v - \text{target}| + \text{size penalty}$.
    #[inline(always)]
    fn calculate_heuristic(&self, pool: &[Expr]) -> i64 {
        let mut min_dist = f64::INFINITY;
        for expr in pool {
            let dist = (expr.value.to_f64() - self.target).abs();
            if dist < min_dist {
                min_dist = dist;
            }
        }
        
        // Minor penalty to longer pools naturally guides the heap toward completed trees.
        let score = min_dist + (pool.len() as f64 * 5.0);
        (score * 1000.0) as i64
    }

    /// Primary combinatorial state expansion. Abstracted to perfectly DRY the logic
    /// between Rayon parallel generation, sequential recursive descent, and Best-First Search.
    #[inline(always)]
    fn expand_states<F>(
        &self,
        pool: &ArrayVec<Expr, 12>,
        current_hash: u64,
        cache: &RefCell<FxHashSet<u64>>,
        arena: &mut AstArena,
        unary_all_nodes: bool,
        mut on_state: F,
    ) where
        F: FnMut(ArrayVec<Expr, 12>, u64, &mut AstArena),
    {
        let checkpoint = arena.len();

        // 1. Commutative binary operations
        for i in 0..pool.len() {
            for j in (i + 1)..pool.len() {
                for &op in &self.comm_ops {
                    if let Some(new_val) = apply_binary(op, &pool[i].value, &pool[j].value) {
                        if new_val.is_too_large() { continue; }
                        let next_hash = hash_update_binary(
                            current_hash, pool[i].value.to_f64(), pool[j].value.to_f64(), new_val.to_f64()
                        );
                        if cache.borrow().contains(&next_hash) { continue; }

                        let tree_idx = arena.alloc(ExprTree::Node(op, pool[i].tree_idx, Some(pool[j].tree_idx)));
                        let mut new_pool = pool.clone();
                        new_pool.swap_remove(j);
                        new_pool.swap_remove(i);
                        new_pool.push(Expr { value: new_val, tree_idx, unary_mask: 0 });

                        on_state(new_pool, next_hash, arena);
                        arena.truncate(checkpoint);
                    }
                }
            }
        }

        // 2. Non-commutative binary operations
        for i in 0..pool.len() {
            for j in 0..pool.len() {
                if i == j { continue; }
                for &op in &self.non_comm_ops {
                    if let Some(new_val) = apply_binary(op, &pool[i].value, &pool[j].value) {
                        if new_val.is_too_large() { continue; }
                        let next_hash = hash_update_binary(
                            current_hash, pool[i].value.to_f64(), pool[j].value.to_f64(), new_val.to_f64()
                        );
                        if cache.borrow().contains(&next_hash) { continue; }

                        let tree_idx = arena.alloc(ExprTree::Node(op, pool[i].tree_idx, Some(pool[j].tree_idx)));
                        let mut new_pool = pool.clone();
                        let (max_idx, min_idx) = if i > j { (i, j) } else { (j, i) };
                        new_pool.swap_remove(max_idx);
                        new_pool.swap_remove(min_idx);
                        new_pool.push(Expr { value: new_val, tree_idx, unary_mask: 0 });

                        on_state(new_pool, next_hash, arena);
                        arena.truncate(checkpoint);
                    }
                }
            }
        }

        // 3. Unary operations
        let unary_start = if unary_all_nodes { 0 } else { pool.len().saturating_sub(1) };
        for i in unary_start..pool.len() {
            for &op in &self.unary_ops {
                let op_mask = op.family_mask();
                if (pool[i].unary_mask & op_mask) != 0 { continue; }

                if let Some(new_val) = apply_unary(op, &pool[i].value) {
                    if new_val.is_too_large() { continue; }
                    let next_hash = hash_update_unary(current_hash, pool[i].value.to_f64(), new_val.to_f64());
                    if cache.borrow().contains(&next_hash) { continue; }

                    let tree_idx = arena.alloc(ExprTree::Node(op, pool[i].tree_idx, None));
                    let mut new_pool = pool.clone();
                    
                    let mut modified_expr = new_pool.swap_remove(i);
                    modified_expr.value = new_val;
                    modified_expr.tree_idx = tree_idx;
                    modified_expr.unary_mask |= op_mask;
                    new_pool.push(modified_expr);

                    on_state(new_pool, next_hash, arena);
                    arena.truncate(checkpoint);
                }
            }
        }
    }

    /// Internal recursive solver utilized by exhaustive parallel search.
    fn solve_internal(
        &self,
        pool: ArrayVec<Expr, 12>,
        current_hash: u64,
        cache: &RefCell<FxHashSet<u64>>,
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

        if !cache.borrow_mut().insert(current_hash) {
            return vec![];
        }

        let mut solutions = Vec::new();
        self.expand_states(
            &pool, current_hash, cache, transient_arena, false,
            |new_pool, next_hash, next_arena| {
                solutions.extend(self.solve_internal(
                    new_pool, next_hash, cache, next_arena, results_arena,
                    counter, local_steps
                ));
            }
        );

        solutions
    }

    /// Pre-calculates the first generation of permutations to feed Rayon threads.
    fn generate_initial_tasks(&self, pool: ArrayVec<Expr, 12>, base_arena: &AstArena) -> Vec<(ArrayVec<Expr, 12>, AstArena, u64)> {
        let mut tasks = Vec::new();
        let mut t_arena = base_arena.clone();
        let current_hash = hash_pool(&pool);
        let empty_cache = RefCell::new(FxHashSet::default());

        self.expand_states(
            &pool, current_hash, &empty_cache, &mut t_arena, true,
            |new_pool, next_hash, arena| {
                tasks.push((new_pool, arena.clone(), next_hash));
            }
        );
        tasks
    }

    /// Solves the target comprehensively by chunking the initial combinatorial depth onto threads.
    pub fn solve_parallel(
        &self,
        pool: ArrayVec<Expr, 12>,
        base_arena: &AstArena,
    ) -> Vec<(Vec<Expr>, AstArena)> {
        let tasks = self.generate_initial_tasks(pool, base_arena);
        let counter = AtomicUsize::new(0);

        tasks.into_par_iter().map(|(p, mut t_arena, hash)| {
            let local_cache = RefCell::new(FxHashSet::default());
            let mut results_arena = AstArena::default();
            let mut local_steps = 0;

            let results = self.solve_internal(
                p, hash, &local_cache, &mut t_arena, &mut results_arena,
                &counter, &mut local_steps
            );
            (results, results_arena)
        }).collect()
    }

    /// Priority Queue-driven Best-First Search. 
    /// Instantly returns the first resolved abstract syntax tree by aggressively minimizing cost.
    pub fn find_first(
        &self,
        initial_pool: ArrayVec<Expr, 12>,
        base_arena: &AstArena,
    ) -> Option<(Expr, AstArena)> {
        let mut heap = BinaryHeap::new();
        let cache_cell = RefCell::new(FxHashSet::default());

        let initial_hash = hash_pool(&initial_pool);
        heap.push(SearchState {
            heuristic_score: self.calculate_heuristic(&initial_pool),
            pool: initial_pool,
            arena: base_arena.clone(),
            hash: initial_hash,
        });

        let mut steps = 0;

        while let Some(state) = heap.pop() {
            if let Some(max) = self.limit {
                if steps >= max { return None; }
            }
            steps += 1;

            if state.pool.len() == 1 {
                let val = state.pool[0].value.to_f64();
                if val >= self.target_min && val <= self.target_max {
                    return Some((state.pool[0].clone(), state.arena));
                }
                continue;
            }

            // Reject cyclic regressions or redundant branches.
            if !cache_cell.borrow_mut().insert(state.hash) {
                continue;
            }

            let mut current_arena = state.arena;

            self.expand_states(
                &state.pool,
                state.hash,
                &cache_cell,
                &mut current_arena,
                false,
                |new_pool, next_hash, next_arena| {
                    let score = self.calculate_heuristic(&new_pool);
                    heap.push(SearchState {
                        pool: new_pool,
                        arena: next_arena.clone(),
                        hash: next_hash,
                        heuristic_score: score,
                    });
                }
            );
        }

        None
    }
}