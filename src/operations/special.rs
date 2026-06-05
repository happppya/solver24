use crate::expr::{FastRatio, Op, Value};
use num_bigint::BigInt;
use num_rational::Ratio;
use num_traits::One;
use statrs::function::gamma::gamma;

#[inline(always)]
pub(crate) fn eval_unary(op: Op, val: &Value) -> Option<Value> {
    match op {
        Op::Factorial => {
            let v = val.to_f64();
            if v >= 0.0 && v <= 100.0 && v.fract() == 0.0 {
                let mut result = BigInt::one();
                for i in 2..=(v as u32) { result *= BigInt::from(i); }
                // Factorials scale rapidly, default to Big allocations
                Some(Value::Exact(FastRatio::Big(Ratio::from_integer(result))))
            } else { None }
        },
        Op::Gamma => {
            let v = val.to_f64();
            if v <= 0.0 && v.fract() == 0.0 { None } else { Some(Value::Approx(gamma(v))) }
        },
        _ => unreachable!("Routed non-special operation to special module"),
    }
}