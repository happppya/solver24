use crate::types::{Op, Value};

/// Evaluates definite integrals for simplified polynomial functions.
/// 
/// Computes analytical solutions using the fundamental theorem of calculus.
pub(crate) fn eval_binary(op: Op, val_a: &Value, val_b: &Value) -> Option<Value> {
    let a = val_a.to_f64();
    let b = val_b.to_f64();

    let res = match op {
        // Solves $\int_a^b x \, dx = \frac{b^2 - a^2}{2}$
        Op::IntegralX => (b.powi(2) - a.powi(2)) / 2.0,
        
        // Solves $\int_a^b x^2 \, dx = \frac{b^3 - a^3}{3}$
        Op::IntegralX2 => (b.powi(3) - a.powi(3)) / 3.0,
        
        _ => unreachable!("Routed non-calculus operation to calculus module"),
    };

    if res.is_finite() {
        Some(Value::Approx(res))
    } else {
        None
    }
}