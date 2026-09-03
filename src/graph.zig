pub const UnaryOp = enum {
    neg,
    sqrt,
    exp,
    log,
    sin,
    cos,
    abs,
    sum,
};

pub const BinaryOp = enum {
    add,
    sub,
    mul,
    div,
    atan2,
    dot,
};

pub const CastOp = enum {
    get_element,
    set_element,
};

const Shape = union(enum) {
    scalar,
    vector: usize,
    matrix: struct { rows: usize, cols: usize },
};

const Node = struct {
    shape: Shape,
    op: union(enum) {
        parameter: usize,
        scalar_constant: f64,
        vector_constant: []const f64,
        matrix_constant: []const []const f64,
        unary: struct { input: usize, op: UnaryOp },
        binary: struct { lhs: usize, rhs: usize, op: BinaryOp },
        cast: struct { input: usize, op: CastOp },
    },
};

const Graph = struct {
    output_index: []const usize,
    nodes: []Node,
};
