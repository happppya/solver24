use crate::expr::{Op, Value};
use num_bigint::BigInt;
use num_rational::Ratio;
use num_traits::{Zero, One, ToPrimitive};
use statrs::function::gamma::gamma;

#[inline(always)]
pub fn apply_binary(op: Op, val_a: &Value, val_b: &Value) -> Option<Value> {
    match op {
        Op::Add => match (val_a, val_b) {
            (Value::Exact(x), Value::Exact(y)) => Some(Value::Exact(x + y)),
            (a, b) => Some(Value::Approx(a.to_f64() + b.to_f64())),
        },
        Op::Sub => match (val_a, val_b) {
            (Value::Exact(x), Value::Exact(y)) => Some(Value::Exact(x - y)),
            (a, b) => Some(Value::Approx(a.to_f64() - b.to_f64())),
        },
        Op::Mult => match (val_a, val_b) {
            (Value::Exact(x), Value::Exact(y)) => Some(Value::Exact(x * y)),
            (a, b) => Some(Value::Approx(a.to_f64() * b.to_f64())),
        },
        Op::Div => match (val_a, val_b) {
            (Value::Exact(x), Value::Exact(y)) => {
                if !y.numer().is_zero() { Some(Value::Exact(x / y)) } else { None }
            },
            (a, b) => {
                let d = b.to_f64();
                if d.abs() > 1e-9 { Some(Value::Approx(a.to_f64() / d)) } else { None }
            }
        },
        Op::Pow => match (val_a, val_b) {
            (Value::Exact(b), Value::Exact(e)) if e.is_integer() && *e.numer() >= BigInt::zero() && *e.numer() <= BigInt::from(100) => {
                let exp_i32 = e.to_integer().to_i32().unwrap_or(0);
                Some(Value::Exact(b.pow(exp_i32)))
            },
            (a, b) => Some(Value::Approx(a.to_f64().powf(b.to_f64()))),
        },
        Op::Root => {
            let v = val_a.to_f64();
            let d = val_b.to_f64();
            if d.abs() < 1e-9 { return None; } 
            
            let res = if v < 0.0 && d % 2.0 != 0.0 { -(-v).powf(1.0 / d) } 
                      else if v >= 0.0 { v.powf(1.0 / d) } 
                      else { return None; };
            Some(Value::Approx(res))
        },
        Op::Log => {
            let b = val_a.to_f64();
            let o = val_b.to_f64();
            if b <= 0.0 || b == 1.0 || o <= 0.0 { None } else { Some(Value::Approx(o.log(b))) }
        },
        
        // --- Shifting & Rotation Logic ---
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
        _ => None, // Unary operations hit here if incorrectly routed
    }
}

#[inline(always)]
pub fn apply_unary(op: Op, val: &Value) -> Option<Value> {
    match op {
        Op::Factorial => {
            let v = val.to_f64();
            if v >= 0.0 && v <= 100.0 && v.fract() == 0.0 {
                let mut result = BigInt::one();
                for i in 2..=(v as u32) { result *= BigInt::from(i); }
                Some(Value::Exact(Ratio::from_integer(result)))
            } else { None }
        },
        Op::Gamma => {
            let v = val.to_f64();
            if v <= 0.0 && v.fract() == 0.0 { None } else { Some(Value::Approx(gamma(v))) }
        },
        _ => None,
    }
}

// Helper functions for main.rs to grab operation pools
pub fn standard_operations() -> Vec<Op> { vec![Op::Add, Op::Sub, Op::Mult, Op::Div] }
pub fn powers_and_roots() -> Vec<Op> { vec![Op::Pow, Op::Root] }
pub fn logarithms() -> Vec<Op> { vec![Op::Log] }
pub fn factorials_and_gamma() -> Vec<Op> { vec![Op::Factorial, Op::Gamma] }

pub fn generate_shifts(bases: &[u8], widths: &[u32]) -> Vec<Op> {
    let mut ops = Vec::with_capacity(bases.len() * widths.len() * 4);
    for &base in bases {
        for &width in widths {
            ops.push(Op::LShift(base, width));
            ops.push(Op::RShift(base, width));
            ops.push(Op::LCirc(base, width));
            ops.push(Op::RCirc(base, width));
        }
    }
    ops
}