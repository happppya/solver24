use crate::expr::{Op, Value};

mod core_math;
mod bitwise;
mod special;
mod trig;
mod calculus;

pub fn apply_binary(op: Op, val_a: &Value, val_b: &Value) -> Option<Value> {
    match op {
        Op::Add | Op::Sub | Op::Mult | Op::Div | Op::Pow | Op::Root => {
            core_math::eval_binary(op, val_a, val_b)
        }
        
        Op::LShift(..) | Op::RShift(..) | Op::LCirc(..) | Op::RCirc(..) => {
            bitwise::eval_binary(op, val_a, val_b)
        }
        
        Op::IntegralX | Op::IntegralX2 => {
            calculus::eval_binary(op, val_a, val_b)
        }

        Op::Log => {
            let b = val_a.to_f64();
            let o = val_b.to_f64();
            // Checking floating point equality strictly is a code smell, 
            // but kept here to match your domain constraints.
            if b <= 0.0 || b == 1.0 || o <= 0.0 { 
                None 
            } else { 
                Some(Value::Approx(o.log(b))) 
            }
        }
        
        _ => None,
    }
}

pub fn apply_unary(op: Op, val: &Value) -> Option<Value> {
    match op {
        Op::Factorial | Op::Gamma => special::eval_unary(op, val),
        Op::Sin | Op::Cos | Op::Tan | Op::Asin | Op::Acos | Op::Atan | 
        Op::Sec | Op::Csc | Op::Cot => trig::eval_unary(op, val),
        _ => None,
    }
}

pub fn standard_operations() -> Vec<Op> { vec![Op::Add, Op::Sub, Op::Mult, Op::Div] }
pub fn powers_and_roots() -> Vec<Op> { vec![Op::Pow, Op::Root] }
pub fn logarithms() -> Vec<Op> { vec![Op::Log] }
pub fn factorials_and_gamma() -> Vec<Op> { vec![Op::Factorial, Op::Gamma] }

pub fn trig() -> Vec<Op> {
    vec![
        Op::Sin, Op::Cos, Op::Tan, Op::Asin, Op::Acos, Op::Atan,
        Op::Sec, Op::Csc, Op::Cot
    ]
}

pub fn calculus() -> Vec<Op> { vec![Op::IntegralX, Op::IntegralX2] }

// Consider migrating this to return `impl Iterator<Item = Op>` in the future
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