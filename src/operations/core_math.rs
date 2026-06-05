use crate::expr::{FastRatio, Op, Value};
use num_bigint::BigInt;
use num_rational::Ratio;
use num_traits::{ToPrimitive, Zero};

#[inline(always)]
fn gcd(mut a: i64, mut b: i64) -> i64 {
    while b != 0 {
        let t = b;
        b = a % b;
        a = t;
    }
    a.abs()
}

#[inline(always)]
fn simplify_small(n: i64, d: i64) -> FastRatio {
    if d == 0 {
        return FastRatio::Small(n, 1);
    }
    let g = gcd(n, d);
    let (mut sn, mut sd) = (n / g, d / g);
    if sd < 0 {
        sn = -sn;
        sd = -sd;
    }
    FastRatio::Small(sn, sd)
}

fn to_big(r: &FastRatio) -> Ratio<BigInt> {
    match r {
        FastRatio::Small(n, d) => Ratio::new(BigInt::from(*n), BigInt::from(*d)),
        FastRatio::Big(b) => b.clone(),
    }
}

// Zero-overhead arithmetic with fallback bounds
fn fast_add(a: &FastRatio, b: &FastRatio) -> FastRatio {
    if let (FastRatio::Small(n1, d1), FastRatio::Small(n2, d2)) = (a, b) {
        if let (Some(n1d2), Some(n2d1), Some(d1d2)) = (n1.checked_mul(*d2), n2.checked_mul(*d1), d1.checked_mul(*d2)) {
            if let Some(num) = n1d2.checked_add(n2d1) {
                return simplify_small(num, d1d2);
            }
        }
    }
    FastRatio::Big(to_big(a) + to_big(b))
}

fn fast_sub(a: &FastRatio, b: &FastRatio) -> FastRatio {
    if let (FastRatio::Small(n1, d1), FastRatio::Small(n2, d2)) = (a, b) {
        if let (Some(n1d2), Some(n2d1), Some(d1d2)) = (n1.checked_mul(*d2), n2.checked_mul(*d1), d1.checked_mul(*d2)) {
            if let Some(num) = n1d2.checked_sub(n2d1) {
                return simplify_small(num, d1d2);
            }
        }
    }
    FastRatio::Big(to_big(a) - to_big(b))
}

fn fast_mul(a: &FastRatio, b: &FastRatio) -> FastRatio {
    if let (FastRatio::Small(n1, d1), FastRatio::Small(n2, d2)) = (a, b) {
        if let (Some(num), Some(den)) = (n1.checked_mul(*n2), d1.checked_mul(*d2)) {
            return simplify_small(num, den);
        }
    }
    FastRatio::Big(to_big(a) * to_big(b))
}

fn fast_div(a: &FastRatio, b: &FastRatio) -> Option<FastRatio> {
    match b {
        FastRatio::Small(n, _) if *n == 0 => return None,
        FastRatio::Big(r) if r.numer().is_zero() => return None,
        _ => {}
    }

    if let (FastRatio::Small(n1, d1), FastRatio::Small(n2, d2)) = (a, b) {
        if let (Some(num), Some(den)) = (n1.checked_mul(*d2), d1.checked_mul(*n2)) {
            return Some(simplify_small(num, den));
        }
    }
    Some(FastRatio::Big(to_big(a) / to_big(b)))
}

fn is_integer_pow(e: &FastRatio) -> Option<i32> {
    match e {
        FastRatio::Small(n, d) if *d == 1 && *n >= 0 && *n <= 100 => Some(*n as i32),
        FastRatio::Big(r) if r.is_integer() && *r.numer() >= BigInt::zero() && *r.numer() <= BigInt::from(100) => {
            r.to_integer().to_i32()
        }
        _ => None,
    }
}

#[inline(always)]
pub(crate) fn eval_binary(op: Op, val_a: &Value, val_b: &Value) -> Option<Value> {
    match op {
        Op::Add => match (val_a, val_b) {
            (Value::Exact(x), Value::Exact(y)) => Some(Value::Exact(fast_add(x, y))),
            (a, b) => Some(Value::Approx(a.to_f64() + b.to_f64())),
        },
        Op::Sub => match (val_a, val_b) {
            (Value::Exact(x), Value::Exact(y)) => Some(Value::Exact(fast_sub(x, y))),
            (a, b) => Some(Value::Approx(a.to_f64() - b.to_f64())),
        },
        Op::Mult => match (val_a, val_b) {
            (Value::Exact(x), Value::Exact(y)) => Some(Value::Exact(fast_mul(x, y))),
            (a, b) => Some(Value::Approx(a.to_f64() * b.to_f64())),
        },
        Op::Div => match (val_a, val_b) {
            (Value::Exact(x), Value::Exact(y)) => fast_div(x, y).map(Value::Exact),
            (a, b) => {
                let d = b.to_f64();
                if d.abs() > 1e-9 { Some(Value::Approx(a.to_f64() / d)) } else { None }
            }
        },
        Op::Pow => match (val_a, val_b) {
            (Value::Exact(b), Value::Exact(e)) => {
                if let Some(exp) = is_integer_pow(e) {
                    let res = match b {
                        FastRatio::Small(n, d) => {
                            if let (Some(num), Some(den)) = (n.checked_pow(exp as u32), d.checked_pow(exp as u32)) {
                                simplify_small(num, den)
                            } else {
                                FastRatio::Big(to_big(b).pow(exp))
                            }
                        }
                        FastRatio::Big(r) => FastRatio::Big(r.pow(exp)),
                    };
                    Some(Value::Exact(res))
                } else {
                    Some(Value::Approx(val_a.to_f64().powf(val_b.to_f64())))
                }
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
        _ => unreachable!("Routed non-core operation to core_math module"),
    }
}