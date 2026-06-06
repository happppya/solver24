use num_bigint::BigInt;
use num_rational::Ratio;
use num_traits::ToPrimitive;

#[derive(Clone, Debug, PartialEq)]
pub enum FastRatio {
    Small(i64, i64),
    Big(Ratio<BigInt>),
}

impl FastRatio {
    #[inline(always)]
    pub fn to_f64(&self) -> f64 {
        match self {
            Self::Small(n, d) => (*n as f64) / (*d as f64),
            Self::Big(r) => {
                let n = r.numer().to_f64().unwrap_or(f64::INFINITY);
                let d = r.denom().to_f64().unwrap_or(f64::INFINITY);
                n / d
            }
        }
    }

    #[inline(always)]
    pub fn is_too_large(&self) -> bool {
        const MAX_BITS: u64 = 8192;
        match self {
            Self::Small(_, _) => false,
            Self::Big(r) => r.numer().bits() > MAX_BITS || r.denom().bits() > MAX_BITS,
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub enum Value {
    Exact(FastRatio),
    Approx(f64),
}

impl Value {
    #[inline(always)]
    pub fn to_f64(&self) -> f64 {
        match self {
            Value::Exact(r) => r.to_f64(),
            Value::Approx(f) => *f,
        }
    }

    /// Enforces strict mathematical barriers. NaN and Infinity states are viral
    /// and will poison the hashing cache if allowed to propagate.
    #[inline(always)]
    pub fn is_too_large(&self) -> bool {
        match self {
            Value::Exact(r) => r.is_too_large(),
            Value::Approx(f) => f.is_infinite() || f.is_nan(),
        }
    }
}

#[derive(Clone, Debug)]
pub enum ExprTree {
    // Note: Storing an owned String here bloats the node size to 32 bytes and 
    // triggers heap allocations. Consider migrating to Interned strings or numeric IDs.
    Leaf(String),
    Node(Op, u32, Option<u32>),
}

/// A contiguous, flat-memory arena for AST generation.
#[derive(Clone, Default, Debug)]
pub struct AstArena {
    nodes: Vec<ExprTree>,
}

impl AstArena {
    /// Pre-allocates arena capacity to prevent reallocation overhead during first-generation mapping.
    #[inline(always)]
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            nodes: Vec::with_capacity(capacity),
        }
    }

    #[inline(always)]
    pub fn alloc(&mut self, node: ExprTree) -> u32 {
        let idx = self.nodes.len() as u32;
        self.nodes.push(node);
        idx
    }

    #[inline(always)]
    pub fn len(&self) -> u32 {
        self.nodes.len() as u32
    }

    #[inline(always)]
    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }

    #[inline(always)]
    pub fn truncate(&mut self, len: u32) {
        self.nodes.truncate(len as usize);
    }

    /// Deep copies a node lineage from a transient arena into this persistent one.
    pub fn copy_from(&mut self, source: &AstArena, root_idx: u32) -> u32 {
        match &source.nodes[root_idx as usize] {
            ExprTree::Leaf(s) => self.alloc(ExprTree::Leaf(s.clone())),
            ExprTree::Node(op, left, right) => {
                let new_left = self.copy_from(source, *left);
                let new_right = right.map(|r| self.copy_from(source, r));
                self.alloc(ExprTree::Node(*op, new_left, new_right))
            }
        }
    }

    pub fn format(&self, idx: u32) -> String {
        match &self.nodes[idx as usize] {
            ExprTree::Leaf(s) => s.clone(),
            ExprTree::Node(op, left, right) => {
                let left_str = self.format(*left);
                let right_str = right.map(|r| self.format(r));
                op.format_str(&left_str, right_str.as_deref())
            }
        }
    }
}

/// Defines all supported mathematical, bitwise, and special operations.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Op {
    // Standard Math
    Add, Sub, Mult, Div, Pow, Root, Log,
    
    // Bitwise Operations
    LShift(u8, u32), RShift(u8, u32), LCirc(u8, u32), RCirc(u8, u32),
    
    // Special Functions
    Factorial, Gamma,
    
    // Trigonometry
    Sin, Cos, Tan, Asin, Acos, Atan, Sec, Csc, Cot,
    
    // Calculus (Definite Integrals)
    IntegralX, IntegralX2,
}

impl Op {
    /// Determines if the operation satisfies the commutative property.
    #[inline]
    pub fn is_commutative(&self) -> bool {
        matches!(self, Op::Add | Op::Mult)
    }

    /// Identifies whether the operation has an arity of one.
    #[inline]
    pub fn is_unary(&self) -> bool {
        matches!(
            self,
            Op::Factorial
                | Op::Gamma
                | Op::Sin
                | Op::Cos
                | Op::Tan
                | Op::Asin
                | Op::Acos
                | Op::Atan
                | Op::Sec
                | Op::Csc
                | Op::Cot
        )
    }

    /// Returns a bitmask representing the operation's asymptotic family.
    /// Used by the solver to outright prune cyclical loops (e.g., sin(cos(x)) or sin(asin(x))).
    #[inline]
    pub fn family_mask(&self) -> u32 {
        match self {
            // Standard Trig: Bit 0
            Op::Sin | Op::Cos | Op::Tan | Op::Sec | Op::Csc | Op::Cot => 1 << 0,
            // Inverse Trig: Bit 1
            Op::Asin | Op::Acos | Op::Atan => 1 << 1,
            // Gamma / Factorial: Bit 2
            Op::Factorial | Op::Gamma => 1 << 2,
            _ => 0,
        }
    }

    /// Formats the operation into a human-readable string representation.
    ///
    /// # Panics
    /// Panics if a binary operation is provided but `b` is `None`.
    pub fn format_str(&self, a: &str, b: Option<&str>) -> String {
        match self {
            // Unary Operations
            Op::Factorial => format!("{}!", a),
            Op::Gamma => format!("Gamma({})", a),
            Op::Sin => format!("sin({})", a),
            Op::Cos => format!("cos({})", a),
            Op::Tan => format!("tan({})", a),
            Op::Asin => format!("asin({})", a),
            Op::Acos => format!("acos({})", a),
            Op::Atan => format!("atan({})", a),
            Op::Sec => format!("sec({})", a),
            Op::Csc => format!("csc({})", a),
            Op::Cot => format!("cot({})", a),

            // Binary Operations
            _ => {
                let b_str = b.expect("CRITICAL: Binary operation missing second operand");
                match self {
                    Op::Add => format!("({} + {})", a, b_str),
                    Op::Sub => format!("({} - {})", a, b_str),
                    Op::Mult => format!("({} * {})", a, b_str),
                    Op::Div => format!("({} / {})", a, b_str),
                    Op::Pow => format!("({} ^ {})", a, b_str),
                    Op::Root => format!("root({}, {})", a, b_str),
                    Op::Log => format!("log_{}({})", a, b_str),
                    
                    Op::LShift(base, width) => format!("({} <<_{}^{} {})", a, base, width, b_str),
                    Op::RShift(base, width) => format!("({} >>_{}^{} {})", a, base, width, b_str),
                    Op::LCirc(base, width) => format!("({} LCIRC_{}^{} {})", a, base, width, b_str),
                    Op::RCirc(base, width) => format!("({} RCIRC_{}^{} {})", a, base, width, b_str),
                    
                    Op::IntegralX => format!("int_{}^{} x dx", a, b_str),
                    Op::IntegralX2 => format!("int_{}^{} x^2 dx", a, b_str),
                    
                    _ => unreachable!("Unhandled binary formatting variant"),
                }
            }
        }
    }
}

/// Represents an evaluated node or partial branch within the expression tree.
#[derive(Clone, Debug)]
pub struct Expr {
    /// The exact or approximate numerical value of this node.
    pub value: Value,
    /// The unique index mapping this expression to its position in the AST.
    pub tree_idx: u32,
    /// The bitmask tracking the execution history of unary families to prevent cyclical explosion.
    pub unary_mask: u32,
}