use num_bigint::BigInt;
use num_rational::Ratio;
use num_traits::ToPrimitive;
use std::sync::Arc;

#[derive(Clone, Debug, PartialEq)]
pub enum Value {
    Exact(Ratio<BigInt>),
    Approx(f64),
}

impl Value {
    pub fn to_f64(&self) -> f64 {
        match self {
            Value::Exact(r) => {
                let n = r.numer().to_f64().unwrap_or(f64::INFINITY);
                let d = r.denom().to_f64().unwrap_or(f64::INFINITY);
                n / d
            }
            Value::Approx(f) => *f,
        }
    }

    pub fn is_too_large(&self) -> bool {
        const MAX_BITS: u64 = 8192;
        match self {
            Value::Exact(r) => r.numer().bits() > MAX_BITS || r.denom().bits() > MAX_BITS,
            Value::Approx(f) => f.is_infinite() || f.is_nan(),
        }
    }
}

/// AST representation of the mathematical operations applied.
#[derive(Clone, Debug)]
pub enum ExprTree {
    Leaf(String),
    Node(Op, Arc<ExprTree>, Option<Arc<ExprTree>>),
}

impl ExprTree {
    /// Recursively builds the formatted string only when explicitly requested.
    pub fn format(&self) -> String {
        match self {
            ExprTree::Leaf(s) => s.clone(),
            ExprTree::Node(op, left, right) => {
                let left_str = left.format();
                let right_str = right.as_ref().map(|r| r.format());
                op.format_str(&left_str, right_str.as_deref())
            }
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Op {
    // Binary
    Add,
    Sub,
    Mult,
    Div,
    Pow,
    Root,
    Log,

    // Binary with baked-in data
    LShift(u8, u32),
    RShift(u8, u32),
    LCirc(u8, u32),
    RCirc(u8, u32),

    // Unary
    Factorial,
    Gamma,
}

impl Op {
    #[inline(always)]
    pub fn is_commutative(&self) -> bool {
        matches!(self, Op::Add | Op::Mult)
    }

    /// Handles string formatting for the operation.
    pub fn format_str(&self, a: &str, b: Option<&str>) -> String {
        match self {
            Op::Add => format!("({} + {})", a, b.unwrap()),
            Op::Sub => format!("({} - {})", a, b.unwrap()),
            Op::Mult => format!("({} * {})", a, b.unwrap()),
            Op::Div => format!("({} / {})", a, b.unwrap()),
            Op::Pow => format!("({} ^ {})", a, b.unwrap()),
            Op::Root => format!("root({}, {})", a, b.unwrap()),
            Op::Log => format!("log_{}({})", a, b.unwrap()),
            Op::LShift(base, width) => format!("({} <<_{}^{} {})", a, base, width, b.unwrap()),
            Op::RShift(base, width) => format!("({} >>_{}^{} {})", a, base, width, b.unwrap()),
            Op::LCirc(base, width) => format!("({} LCIRC_{}^{} {})", a, base, width, b.unwrap()),
            Op::RCirc(base, width) => format!("({} RCIRC_{}^{} {})", a, base, width, b.unwrap()),
            Op::Factorial => format!("{}!", a),
            Op::Gamma => format!("Gamma({})", a),
        }
    }
}

#[derive(Clone, Debug)]
pub struct Expr {
    pub value: Value,
    pub tree: Arc<ExprTree>,
    pub unary_depth: u8,
}
