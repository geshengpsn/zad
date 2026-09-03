const IRCode = @import("ir.zig").IRCode;

const UnaryOp = union(enum) {
    neg,
    sqrt,
    exp,
    log,
    sin,
    cos,
    abs,
    sum,
    get_vec: usize,
    get_mat: struct { row: usize, col: usize },
};

pub const BinaryOp = union(enum) {
    add,
    sub,
    mul,
    div,
    atan2,
    dot,
    outer_product,
    set_vec: usize,
    set_mat: struct { row: usize, col: usize },
};

const Shape = union(enum) {
    scalar,
    vector: usize,
    matrix: struct { rows: usize, cols: usize },
};

fn Node(comptime T: type) type {
    return struct {
        shape: Shape,
        op: union(enum) {
            parameter: usize,
            output: usize,
            scalar_constant: T,
            vector_constant: []const T,
            matrix_constant: []const []const T,
            unary: struct { input: usize, op: UnaryOp },
            binary: struct { lhs: usize, rhs: usize, op: BinaryOp },
        },
    };
}

fn Graph(comptime T: type) type {
    return struct {
        nodes: []const Node(T),
    };
}

fn buildIRInner(comptime T: type, comptime graph: Graph(T)) usize {
    var result = 0;
    for (graph.nodes) |node| {
        switch (node.op) {
            .matrix_constant => |values| {
                result += values[0].len;
            },
            .unary => |v| {
                switch (v.op) {}
            },
            else => {
                result += 1;
            },
        }
    }
    return 0;
}

fn IRCodeLen(comptime T: type, comptime graph: Graph(T)) usize {
    _ = graph;
    return 0;
}

fn buildIR(comptime T: type, comptime graph: Graph(T)) []IRCode(T) {
    _ = graph;
}

test "graph" {
    // const graph = Graph{
    //     .output_index = &[_]usize{0},
    //     .nodes = &[_]Node{
    //         Node{
    //             .shape = Shape.scalar,
    //         },
    //     },
    // };
}
