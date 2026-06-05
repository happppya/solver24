use crate::expr::{Op, Value};

/// Evaluates unary trigonometric operations.
/// 
/// Returns `None` for domain errors (e.g., `acos` of 2.0) or undefined asymptotes.
pub(crate) fn eval_unary(op: Op, val: &Value) -> Option<Value> {
    let v = val.to_f64();
    
    let res = match op {
        Op::Sin => v.sin(),
        Op::Cos => v.cos(),
        Op::Tan => v.tan(),
        Op::Asin if (-1.0..=1.0).contains(&v) => v.asin(),
        Op::Acos if (-1.0..=1.0).contains(&v) => v.acos(),
        Op::Atan => v.atan(),
        Op::Sec => 1.0 / v.cos(),
        Op::Csc => 1.0 / v.sin(),
        Op::Cot => 1.0 / v.tan(),
        Op::Asin | Op::Acos => return None,
        _ => unreachable!("Routed non-trig operation to trig module"),
    };

    // Filter out NaN and Infinities caused by asymptotic bounds (e.g., Csc(0))
    if res.is_finite() {
        Some(Value::Approx(res))
    } else {
        None
    }
}