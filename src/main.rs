mod operations;
mod solver;
mod types;

use arrayvec::ArrayVec;
use std::collections::HashMap;
use std::env;
use std::time::Instant;

use crate::solver::Solver;
use crate::types::{AstArena, Expr, ExprTree, FastRatio, Value};

/// Represents a validated, rendered mathematical solution.
pub struct RenderedSolution {
    pub formula: String,
    pub value: f64,
    pub error: f64,
}

/// Formats and prints the top and bottom solutions for a given dataset.
///
/// Handles both exact and approximate solutions dynamically.
fn print_solutions(label: &str, sols: &[RenderedSolution], is_exact: bool) {
    println!("\n=== {} Solutions (Total: {}) ===", label, sols.len());
    if sols.is_empty() {
        println!("None found.");
        return;
    }

    println!("--- Top 10 Shortest ---");
    for (i, sol) in sols.iter().take(10).enumerate() {
        if is_exact {
            println!("{}. {} = 24", i + 1, sol.formula);
        } else {
            println!(
                "{}. {} = {:.4} ({:+.4})",
                i + 1,
                sol.formula,
                sol.value,
                sol.error
            );
        }
    }

    if sols.len() > 10 {
        println!("--- Top 10 Longest ---");
        let start_idx = sols.len().saturating_sub(10).max(10);
        for (i, sol) in sols[start_idx..].iter().enumerate() {
            if is_exact {
                println!("{}. {} = 24", start_idx + i + 1, sol.formula);
            } else {
                println!(
                    "{}. {} = {:.4} ({:+.4})",
                    start_idx + i + 1,
                    sol.formula,
                    sol.value,
                    sol.error
                );
            }
        }
    }
}

fn main() {
    let mut args = env::args().skip(1);
    let mut input_strings = Vec::new();
    let mut abs_error: Option<f64> = None;
    let mut pct_error: Option<f64> = None;
    let mut limit: Option<usize> = None;

    // TODO: Replace manual loop with `clap` parser
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--error" => {
                let val_str = args.next().unwrap_or_else(|| {
                    eprintln!("Fatal: Missing value for --error");
                    std::process::exit(1);
                });
                abs_error = Some(val_str.parse().unwrap_or_else(|_| {
                    eprintln!("Fatal: Invalid float for --error");
                    std::process::exit(1);
                }));
            }
            "--percent-error" => {
                let val_str = args.next().unwrap_or_else(|| {
                    eprintln!("Fatal: Missing value for --percent-error");
                    std::process::exit(1);
                });
                pct_error = Some(val_str.parse().unwrap_or_else(|_| {
                    eprintln!("Fatal: Invalid float for --percent-error");
                    std::process::exit(1);
                }));
            }
            "--limit" => {
                let val_str = args.next().unwrap_or_else(|| {
                    eprintln!("Fatal: Missing value for --limit");
                    std::process::exit(1);
                });
                limit = Some(val_str.parse().unwrap_or_else(|_| {
                    eprintln!("Fatal: Invalid integer for --limit");
                    std::process::exit(1);
                }));
            }
            _ => input_strings.push(arg),
        }
    }

    if abs_error.is_some() && pct_error.is_some() {
        eprintln!("Fatal: --error and --percent-error are mutually exclusive.");
        std::process::exit(1);
    }

    if input_strings.is_empty() {
        println!("WARN: No inputs provided. Falling back to default debug state (7, 7, 7, 7).");
        input_strings = vec![
            "7".to_string(),
            "7".to_string(),
            "7".to_string(),
            "7".to_string(),
        ];
    }

    let target_g = 24.0;

    let margin = if let Some(err) = abs_error {
        err
    } else if let Some(pct) = pct_error {
        target_g * (pct / 100.0)
    } else {
        1.0
    };

    let target_min = target_g - margin;
    let target_max = target_g + margin;

    let mut active_operations = Vec::new();
    active_operations.extend(operations::standard_operations());
    active_operations.extend(operations::powers_and_roots());
    active_operations.extend(operations::factorials_and_gamma());
    active_operations.extend(operations::calculus());
    active_operations.extend(operations::trig_standard());
    active_operations.extend(operations::trig_cofunctions());
    active_operations.extend(operations::trig_inverse());

    //active_operations.extend(operations::generate_shifts(&[2], &[32]));

    println!("Loaded {} operations.", active_operations.len());

    let mut root_arena = AstArena::default();

    let pool: ArrayVec<Expr, 12> = input_strings
        .iter()
        .map(|s| {
            let parsed_num: f64 = s.parse().unwrap_or_else(|_| {
                eprintln!("Fatal: '{}' is not a valid number.", s);
                std::process::exit(1);
            });

            let value = if parsed_num.fract() == 0.0
                && parsed_num <= i64::MAX as f64
                && parsed_num >= i64::MIN as f64
            {
                Value::Exact(FastRatio::Small(parsed_num as i64, 1))
            } else {
                Value::Approx(parsed_num)
            };

            Expr {
                value,
                tree_idx: root_arena.alloc(ExprTree::Leaf(s.clone())),
                unary_mask: 0,
            }
        })
        .collect();

    let solver = Solver::new(target_g, target_min, target_max, &active_operations, limit);

    println!(
        "Exhaustive search starting... Target: {} (Range: {} - {})",
        target_g, target_min, target_max
    );

    // Track parallel solver execution timing
    let start_time = Instant::now();
    let all_solution_batches = solver.solve_parallel(pool, &root_arena);
    let solver_duration = start_time.elapsed();

    let mut unique_solutions = HashMap::new();

    for (batch, arena) in all_solution_batches {
        for sol in batch {
            let formula_str = arena.format(sol.tree_idx);

            // TODO: Vacuous string deduplication requires structural hash refactor to avoid String allocation overhead
            if !unique_solutions.contains_key(&formula_str) {
                let val = sol.value.to_f64();
                unique_solutions.insert(
                    formula_str.clone(),
                    RenderedSolution {
                        formula: formula_str,
                        value: val,
                        error: val - target_g,
                    },
                );
            }
        }
    }

    let mut exact_sols = Vec::new();
    let mut approx_sols = Vec::new();

    for (_, rendered) in unique_solutions {
        if rendered.error.abs() < 1e-9 {
            exact_sols.push(rendered);
        } else {
            approx_sols.push(rendered);
        }
    }

    exact_sols.sort_unstable_by_key(|e| e.formula.len());
    approx_sols.sort_unstable_by_key(|e| e.formula.len());

    print_solutions("EXACT", &exact_sols, true);
    print_solutions("APPROXIMATE", &approx_sols, false);

    println!("\n[Telemetry] Solver executed in: {:.2?}", solver_duration);
}
