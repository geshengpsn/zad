const Context = struct {};

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

const Shape = union(enum) {
    scalar,
    vector: usize,
    matrix: struct { rows: usize, cols: usize },
};

const Node = struct {
    shape: Shape,
    op: union(enum) {
        parameter: *Node,
        scalar_constant: f64,
        vector_constant: []const f64,
        matrix_constant: []const []const f64,
        unary: struct { input: *Node, op: UnaryOp },
        binary: struct { lhs: *Node, rhs: *Node, op: BinaryOp },
    },
};

const Scalar = struct {
    constant_values: ?f64 = null,
    // index: usize = 0,
    fn init(constant_values: f64) Scalar {
        return .{ .constant_values = constant_values };
    }
    fn parameter() Scalar {
        return Scalar{ .constant_values = null };
    }
};

fn Vector(comptime len: usize) type {
    return struct {
        constant_values: ?[len]f64 = null,
        // index: usize = 0,
        fn init(constant_values: [len]f64) Vector(len) {
            return .{ .constant_values = constant_values };
        }

        fn parameter() Vector(len) {
            return Vector(len){ .constant_values = null };
        }
    };
}

fn Matrix(comptime rows: usize, comptime cols: usize) type {
    return struct {
        constant_values: ?[rows][cols]f64 = null,
        // index: usize = 0,
        fn init(constant_values: [rows][cols]f64) Matrix(rows, cols) {
            return .{ .constant_values = constant_values };
        }
    };
}

const Q = Matrix(2, 2).init(.{
    .{ 1, 0 },
    .{ 0, 1 },
});

const p = Vector(2).init(.{ 0, 0 });

fn qp(x: Vector(2)) Scalar {
    return Q.mul(x).dot(x).add(p.dot(x));
}

test "user input" {
    const x = Vector(2).parameter();
    const result = qp(x);
    try std.testing.expectEqual(result.constant_values, 0);
}
