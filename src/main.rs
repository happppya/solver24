//! Main entry point for the exact-math combinatorial solver.

mod expr;
mod operations;
mod solver;

use arrayvec::ArrayVec;
use num_bigint::BigInt;
use num_rational::Ratio;
use std::collections::HashMap;
use std::env;
use std::sync::Arc;

use crate::expr::{Expr, ExprTree, Value};
use crate::solver::Solver;

/// Formats and prints the top and bottom approximate solutions.
fn print_top_bottom_approx(label: &str, sols: &[(Expr, f64)]) {
    println!("\n=== {} Solutions (Total: {}) ===", label, sols.len());
    if sols.is_empty() {
        println!("None found.");
        return;
    }

    println!("--- Top 10 Shortest ---");
    for (i, (sol, err)) in sols.iter().take(10).enumerate() {
        println!(
            "{}. {} = {:.4} ({:+.4})",
            i + 1,
            sol.tree.format(),
            sol.value.to_f64(),
            err
        );
    }

    if sols.len() > 10 {
        println!("--- Top 10 Longest ---");
        let start_idx = sols.len().saturating_sub(10).max(10);
        for (i, (sol, err)) in sols[start_idx..].iter().enumerate() {
            println!(
                "{}. {} = {:.4} ({:+.4})",
                start_idx + i + 1,
                sol.tree.format(),
                sol.value.to_f64(),
                err
            );
        }
    }
}

/// Helper function to format and print the top and bottom 10 exact solutions.
fn print_top_bottom(label: &str, sols: &[Expr]) {
    println!("\n=== {} Solutions (Total: {}) ===", label, sols.len());
    if sols.is_empty() {
        println!("None found.");
        return;
    }

    println!("--- Top 10 Shortest ---");
    for (i, sol) in sols.iter().take(10).enumerate() {
        println!("{}. {} = 24", i + 1, sol.tree.format());
    }

    if sols.len() > 10 {
        println!("--- Top 10 Longest ---");
        let start_idx = sols.len().saturating_sub(10).max(10);
        for (i, sol) in sols[start_idx..].iter().enumerate() {
            println!("{}. {} = 24", start_idx + i + 1, sol.tree.format());
        }
    }
}

fn main() {
    // 1. Parse command-line arguments manually (consider using `clap` for production)
    let mut args = env::args().skip(1);
    let mut input_strings = Vec::new();
    let mut abs_error: Option<f64> = None;
    let mut pct_error: Option<f64> = None;

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
            _ => input_strings.push(arg),
        }
    }

    if abs_error.is_some() && pct_error.is_some() {
        eprintln!("Fatal: --error and --percent-error are mutually exclusive.");
        std::process::exit(1);
    }

    // Default to debug values if no positional arguments are provided
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
    
    // Calculate margin based on mutually exclusive flags
    let margin = if let Some(err) = abs_error {
        err
    } else if let Some(pct) = pct_error {
        target_g * (pct / 100.0)
    } else {
        1.0 // Default 1.0 margin
    };

    let target_min = target_g - margin;
    let target_max = target_g + margin;

    // Load operations
    let mut active_operations = Vec::new();
    active_operations.extend(operations::standard_operations());
    active_operations.extend(operations::powers_and_roots());
    active_operations.extend(operations::factorials_and_gamma());
    active_operations.extend(operations::logarithms());

    // Adding large arrays of shifts causes massive combinatorial explosion
    let bases = vec![2, 10];
    let widths = vec![8, 16];
    active_operations.extend(operations::generate_shifts(&bases, &widths));

    println!("Loaded {} operations.", active_operations.len());

    // Build the initial pool from CLI arguments
    let pool: ArrayVec<Expr, 12> = input_strings
        .iter()
        .map(|s| {
            let parsed_num: f64 = s.parse().unwrap_or_else(|_| {
                eprintln!("Fatal: '{}' is not a valid number.", s);
                std::process::exit(1);
            });
            
            let value = match Ratio::<BigInt>::from_float(parsed_num) {
                Some(exact_ratio) => Value::Exact(exact_ratio),
                None => Value::Approx(parsed_num),
            };

            Expr {
                value,
                tree: Arc::new(ExprTree::Leaf(s.clone())),
                unary_depth: 0,
            }
        })
        .collect();

    let solver = Solver::new(target_g, target_min, target_max, &active_operations);
    println!(
        "Exhaustive search starting... Target: {} (Range: {} - {})",
        target_g, target_min, target_max
    );

    // Run the solver
    let all_solutions = solver.solve_parallel(pool);

    // Deduplicate results
    let mut unique_solutions = HashMap::with_capacity(all_solutions.len());
    for sol in all_solutions {
        unique_solutions.insert(sol.tree.format(), sol);
    }

    // Partition into Exact vs Approximate matches
    let mut exact_sols = Vec::new();
    let mut approx_sols = Vec::new();

    for (_, sol) in unique_solutions {
        let val = sol.value.to_f64();
        let error = val - target_g;

        if error.abs() < 1e-9 {
            exact_sols.push(sol);
        } else {
            approx_sols.push((sol, error));
        }
    }

    exact_sols.sort_unstable_by_key(|e| e.tree.format().len());
    approx_sols.sort_unstable_by_key(|(e, _)| e.tree.format().len());

    // Print the final metrics
    print_top_bottom("EXACT", &exact_sols);
    print_top_bottom_approx("APPROXIMATE", &approx_sols);
}