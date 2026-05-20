const std = @import("std");

pub const Op1 = enum {
    neg,
    abs,
    exp,
    log,
    sqrt,
    sin,
    cos,
    tan,
};

pub const Op2 = enum {
    add,
    sub,
    mul,
    div,
};

pub fn DAGNode(comptime T: type) type {
    return union(enum) {
        scalar_constant: T,
        scalar_parameter: usize,
        op1: struct {
            node: usize,
            op: Op1,
        },
        op2: struct {
            lhs: usize,
            rhs: usize,
            op: Op2,
        },
        output: struct {
            index: usize,
            node: usize,
        },
    };
}

pub fn op2_node(comptime T: type, lhs: usize, rhs: usize, op: Op2) DAGNode(T) {
    return switch (op) {
        .add, .mul => if (lhs > rhs)
            .{ .op2 = .{ .lhs = rhs, .rhs = lhs, .op = op } }
        else
            .{ .op2 = .{ .lhs = lhs, .rhs = rhs, .op = op } },
        else => .{ .op2 = .{ .lhs = lhs, .rhs = rhs, .op = op } },
    };
}

test "op2_node normalizes commutative operands" {
    try std.testing.expectEqual(
        DAGNode(f64){ .op2 = .{ .lhs = 2, .rhs = 5, .op = .add } },
        op2_node(f64, 5, 2, .add),
    );
    try std.testing.expectEqual(
        DAGNode(f64){ .op2 = .{ .lhs = 5, .rhs = 2, .op = .sub } },
        op2_node(f64, 5, 2, .sub),
    );
}

fn validate_node_ref(comptime node_index: usize, comptime ref_index: usize, comptime label: []const u8) void {
    if (ref_index >= node_index) {
        @compileError("node " ++ std.fmt.comptimePrint("{}", .{node_index}) ++ " has invalid " ++ label ++ " reference " ++ std.fmt.comptimePrint("{}", .{ref_index}) ++ "; references must point to earlier nodes");
    }
}

pub fn validate_dag(comptime T: type, comptime dag: []const DAGNode(T)) void {
    @setEvalBranchQuota(dag.len);
    inline for (dag, 0..) |node, i| {
        switch (node) {
            .scalar_constant, .scalar_parameter => {},
            .op1 => |op| {
                validate_node_ref(i, op.node, "op1 operand");
            },
            .op2 => |op| {
                switch (op.op) {
                    .add, .mul => {
                        if (op.lhs > op.rhs) {
                            @compileError("op2 lhs index " ++ std.fmt.comptimePrint("{}", .{op.lhs}) ++ " is greater than rhs index " ++ std.fmt.comptimePrint("{}", .{op.rhs}));
                        }
                    },
                    else => {},
                }

                validate_node_ref(i, op.lhs, "op2 lhs");
                validate_node_ref(i, op.rhs, "op2 rhs");
            },
            .output => |out| {
                validate_node_ref(i, out.node, "output node");
            },
        }
    }
}

pub fn input_size(comptime T: type, comptime dag: []const DAGNode(T)) usize {
    @setEvalBranchQuota(dag.len);
    var size: usize = 0;
    inline for (dag) |node| {
        switch (node) {
            .scalar_parameter => |i| size = @max(size, i + 1),
            else => {},
        }
    }
    return size;
}

pub fn output_size(comptime T: type, comptime dag: []const DAGNode(T)) usize {
    @setEvalBranchQuota(dag.len);
    var size: usize = 0;
    inline for (dag) |node| {
        switch (node) {
            .output => |o| size = @max(size, o.index + 1),
            else => {},
        }
    }
    return size;
}

test "size" {
    const test_dag = [_]DAGNode(f64){
        DAGNode(f64){ .scalar_parameter = 0 },
        DAGNode(f64){ .op1 = .{ .node = 0, .op = .log } },
        DAGNode(f64){ .scalar_parameter = 1 },
        DAGNode(f64){ .op2 = .{ .lhs = 0, .rhs = 2, .op = .mul } },
        DAGNode(f64){ .op2 = .{ .lhs = 1, .rhs = 3, .op = .add } },
        DAGNode(f64){ .op1 = .{ .node = 2, .op = .sin } },
        DAGNode(f64){ .op2 = .{ .lhs = 4, .rhs = 5, .op = .sub } },
        DAGNode(f64){ .output = .{ .index = 0, .node = 6 } },
    };
    try std.testing.expectEqual(input_size(f64, &test_dag), 2);
    try std.testing.expectEqual(output_size(f64, &test_dag), 1);
    validate_dag(f64, &test_dag);
}
