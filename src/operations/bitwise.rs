use crate::expr::{Op, Value};

#[inline(always)]
pub(crate) fn eval_binary(op: Op, val_a: &Value, val_b: &Value) -> Option<Value> {
    match op {
        Op::LShift(base, width) => {
            let b_f64 = base as f64;
            let w_f64 = width as f64;
            let modulus = b_f64.powf(w_f64);
            let v = val_a.to_f64().trunc().rem_euclid(modulus);
            let s = val_b.to_f64().trunc();
            if s < 0.0 { return None; }
            Some(Value::Approx((v * b_f64.powf(s)).rem_euclid(modulus)))
        },
        Op::RShift(base, _) => {
            let b_f64 = base as f64;
            let v = val_a.to_f64().trunc();
            let s = val_b.to_f64().trunc();
            if s < 0.0 { return None; }
            Some(Value::Approx((v / b_f64.powf(s)).trunc()))
        },
        Op::LCirc(base, width) => {
            let b_f64 = base as f64;
            let w_f64 = width as f64;
            let modulus = b_f64.powf(w_f64);
            let v = val_a.to_f64().trunc().rem_euclid(modulus);
            let s = val_b.to_f64().trunc().rem_euclid(w_f64);
            let left_part = (v * b_f64.powf(s)).rem_euclid(modulus);
            let right_part = (v / b_f64.powf(w_f64 - s)).trunc();
            Some(Value::Approx(left_part + right_part))
        },
        Op::RCirc(base, width) => {
            let b_f64 = base as f64;
            let w_f64 = width as f64;
            let modulus = b_f64.powf(w_f64);
            let v = val_a.to_f64().trunc().rem_euclid(modulus);
            let s = val_b.to_f64().trunc().rem_euclid(w_f64);
            let right_part = (v / b_f64.powf(s)).trunc();
            let left_part = v.rem_euclid(b_f64.powf(s)) * b_f64.powf(w_f64 - s);
            Some(Value::Approx(left_part + right_part))
        },
        _ => unreachable!("Routed non-bitwise operation to bitwise module"),
    }
}