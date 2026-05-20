const std = @import("std");
const dag_mod = @import("../dag.zig");
const DAGNode = dag_mod.DAGNode;
const Builder = @import("../dag_builder.zig").Builder;

const op2_simplify = enum {
    add_zero, // x + 0 = x, 0 + x = x
    sub_zero, // x - 0 = x, 0 - x = -x
    self_sub, // x - x = 0
    mul_zero, // x * 0 = 0, 0 * x = 0
    mul_one, // x * 1 = x, 1 * x = x
    mul_neg, // x * -1 = -x, -1 * x = -x
    div_one, // x / 1 = x
    zero_div, // 0 / x = 0, unsafe
    self_div, // x / x = 1, unsafe
};

const complex_op2_simplify = enum {
    sin_div_cos, // sin(x) / cos(x) = tan(x)
    sin_sq_add_cos_sq, // sin(x) * sin(x) + cos(x) * cos(x) = 1
};

fn is_zero(comptime T: type, node: DAGNode(T)) bool {
    switch (node) {
        .scalar_constant => |val| return val == 0,
        else => return false,
    }
}

fn is_one(comptime T: type, node: DAGNode(T)) bool {
    switch (node) {
        .scalar_constant => |val| return val == 1,
        else => return false,
    }
}

fn is_neg_one(comptime T: type, node: DAGNode(T)) bool {
    switch (node) {
        .scalar_constant => |val| return val == -1,
        else => return false,
    }
}

fn is_same_node(lhs: usize, rhs: usize) bool {
    return lhs == rhs;
}

fn resolve_map(map: []const usize, index: usize) usize {
    var current = index;
    while (map[current] != current) {
        current = map[current];
    }
    return current;
}

fn rewrite_refs(comptime T: type, nodes: []DAGNode(T), map: []const usize) void {
    for (nodes, 0..) |node, i| {
        nodes[i] = switch (node) {
            .op1 => |op1| DAGNode(T){ .op1 = .{ .node = resolve_map(map, op1.node), .op = op1.op } },
            .op2 => |op2| dag_mod.op2_node(T, resolve_map(map, op2.lhs), resolve_map(map, op2.rhs), op2.op),
            .output => |out| DAGNode(T){ .output = .{ .index = out.index, .node = resolve_map(map, out.node) } },
            else => node,
        };
    }
}

fn base_result(comptime T: type, comptime dag: []const DAGNode(T)) struct { nodes: [dag.len]DAGNode(T), map: [dag.len]usize } {
    const result: [dag.len]DAGNode(T) = dag[0..dag.len].*;
    var map: [dag.len]usize = undefined;
    inline for (0..dag.len) |i| {
        map[i] = i;
    }
    return .{ .nodes = result, .map = map };
}

test is_zero {
    try std.testing.expect(is_zero(f32, DAGNode(f32){ .scalar_constant = 0 }));
    try std.testing.expect(!is_zero(f32, DAGNode(f32){ .scalar_constant = 1 }));
}

pub fn has_add_zero(comptime T: type, comptime dag: []const DAGNode(T)) bool {
    for (dag) |node| {
        switch (node) {
            .op2 => |op| {
                if (op.op == .add and (is_zero(T, dag[op.lhs]) or is_zero(T, dag[op.rhs]))) {
                    return true;
                }
            },
            else => {},
        }
    }
    return false;
}

pub fn has_sub_zero(comptime T: type, comptime dag: []const DAGNode(T)) bool {
    for (dag) |node| {
        switch (node) {
            .op2 => |op| {
                if (op.op == .sub and (is_zero(T, dag[op.lhs]) or is_zero(T, dag[op.rhs]))) {
                    return true;
                }
            },
            else => {},
        }
    }
    return false;
}

pub fn has_self_sub(comptime T: type, comptime dag: []const DAGNode(T)) bool {
    for (dag) |node| {
        switch (node) {
            .op2 => |op| {
                if (op.op == .sub and is_same_node(op.lhs, op.rhs)) {
                    return true;
                }
            },
            else => {},
        }
    }
    return false;
}

pub fn has_mul_zero(comptime T: type, comptime dag: []const DAGNode(T)) bool {
    for (dag) |node| {
        switch (node) {
            .op2 => |op| {
                if (op.op == .mul and (is_zero(T, dag[op.lhs]) or is_zero(T, dag[op.rhs]))) {
                    return true;
                }
            },
            else => {},
        }
    }
    return false;
}

pub fn has_mul_one(comptime T: type, comptime dag: []const DAGNode(T)) bool {
    for (dag) |node| {
        switch (node) {
            .op2 => |op| {
                if (op.op == .mul and (is_one(T, dag[op.lhs]) or is_one(T, dag[op.rhs]))) {
                    return true;
                }
            },
            else => {},
        }
    }
    return false;
}

pub fn has_mul_neg(comptime T: type, comptime dag: []const DAGNode(T)) bool {
    for (dag) |node| {
        switch (node) {
            .op2 => |op| {
                if (op.op == .mul and (is_neg_one(T, dag[op.lhs]) or is_neg_one(T, dag[op.rhs]))) {
                    return true;
                }
            },
            else => {},
        }
    }
    return false;
}

pub fn has_div_one(comptime T: type, comptime dag: []const DAGNode(T)) bool {
    for (dag) |node| {
        switch (node) {
            .op2 => |op| {
                if (op.op == .div and is_one(T, dag[op.rhs])) {
                    return true;
                }
            },
            else => {},
        }
    }
    return false;
}

pub fn has_zero_div(comptime T: type, comptime dag: []const DAGNode(T)) bool {
    for (dag) |node| {
        switch (node) {
            .op2 => |op| {
                if (op.op == .div and is_zero(T, dag[op.lhs])) {
                    return true;
                }
            },
            else => {},
        }
    }
    return false;
}

pub fn has_self_div(comptime T: type, comptime dag: []const DAGNode(T)) bool {
    for (dag) |node| {
        switch (node) {
            .op2 => |op| {
                if (op.op == .div and is_same_node(op.lhs, op.rhs)) {
                    return true;
                }
            },
            else => {},
        }
    }
    return false;
}

// x + 0 = x, 0 + x = x
pub fn add_zero(comptime T: type, comptime dag: []const DAGNode(T)) [dag.len]DAGNode(T) {
    var state = base_result(T, dag);
    inline for (dag, 0..) |node, i| {
        switch (node) {
            .op2 => |op| {
                if (op.op == .add and is_zero(T, dag[op.lhs])) {
                    state.map[i] = op.rhs;
                } else if (op.op == .add and is_zero(T, dag[op.rhs])) {
                    state.map[i] = op.lhs;
                }
            },
            else => {},
        }
    }
    rewrite_refs(T, state.nodes[0..], &state.map);
    return state.nodes;
}

// x - 0 = x, 0 - x = -x
pub fn sub_zero(comptime T: type, comptime dag: []const DAGNode(T)) [dag.len]DAGNode(T) {
    var state = base_result(T, dag);
    inline for (dag, 0..) |node, i| {
        switch (node) {
            .op2 => |op| {
                if (op.op == .sub and is_zero(T, dag[op.rhs])) {
                    state.map[i] = op.lhs;
                } else if (op.op == .sub and is_zero(T, dag[op.lhs])) {
                    state.nodes[i] = DAGNode(T){ .op1 = .{ .node = op.rhs, .op = .neg } };
                }
            },
            else => {},
        }
    }
    rewrite_refs(T, state.nodes[0..], &state.map);
    return state.nodes;
}

// x - x = 0
pub fn self_sub(comptime T: type, comptime dag: []const DAGNode(T)) [dag.len]DAGNode(T) {
    var state = base_result(T, dag);
    inline for (dag, 0..) |node, i| {
        switch (node) {
            .op2 => |op| {
                if (op.op == .sub and op.lhs == op.rhs) {
                    state.nodes[i] = DAGNode(T){ .scalar_constant = 0 };
                }
            },
            else => {},
        }
    }
    rewrite_refs(T, state.nodes[0..], &state.map);
    return state.nodes;
}

// x * 0 = 0, 0 * x = 0
pub fn mul_zero(comptime T: type, comptime dag: []const DAGNode(T)) [dag.len]DAGNode(T) {
    var state = base_result(T, dag);
    inline for (dag, 0..) |node, i| {
        switch (node) {
            .op2 => |op| {
                if (op.op == .mul and is_zero(T, dag[op.lhs])) {
                    state.map[i] = op.lhs;
                } else if (op.op == .mul and is_zero(T, dag[op.rhs])) {
                    state.map[i] = op.rhs;
                }
            },
            else => {},
        }
    }
    rewrite_refs(T, state.nodes[0..], &state.map);
    return state.nodes;
}

// x * 1 = x, 1 * x = x
pub fn mul_one(comptime T: type, comptime dag: []const DAGNode(T)) [dag.len]DAGNode(T) {
    var state = base_result(T, dag);
    inline for (dag, 0..) |node, i| {
        switch (node) {
            .op2 => |op| {
                if (op.op == .mul and is_one(T, dag[op.lhs])) {
                    state.map[i] = op.rhs;
                } else if (op.op == .mul and is_one(T, dag[op.rhs])) {
                    state.map[i] = op.lhs;
                }
            },
            else => {},
        }
    }
    rewrite_refs(T, state.nodes[0..], &state.map);
    return state.nodes;
}

// x * -1 = -x, -1 * x = -x
pub fn mul_neg(comptime T: type, comptime dag: []const DAGNode(T)) [dag.len]DAGNode(T) {
    var state = base_result(T, dag);
    inline for (dag, 0..) |node, i| {
        switch (node) {
            .op2 => |op| {
                if (op.op == .mul and is_neg_one(T, dag[op.lhs])) {
                    state.nodes[i] = DAGNode(T){ .op1 = .{ .node = op.rhs, .op = .neg } };
                } else if (op.op == .mul and is_neg_one(T, dag[op.rhs])) {
                    state.nodes[i] = DAGNode(T){ .op1 = .{ .node = op.lhs, .op = .neg } };
                }
            },
            else => {},
        }
    }
    rewrite_refs(T, state.nodes[0..], &state.map);
    return state.nodes;
}

// x / 1 = x
pub fn div_one(comptime T: type, comptime dag: []const DAGNode(T)) [dag.len]DAGNode(T) {
    var state = base_result(T, dag);
    inline for (dag, 0..) |node, i| {
        switch (node) {
            .op2 => |op| {
                if (op.op == .div and is_one(T, dag[op.rhs])) {
                    state.map[i] = op.lhs;
                }
            },
            else => {},
        }
    }
    rewrite_refs(T, state.nodes[0..], &state.map);
    return state.nodes;
}

// 0 / x = 0, unsafe
pub fn zero_div(comptime T: type, comptime dag: []const DAGNode(T)) [dag.len]DAGNode(T) {
    var state = base_result(T, dag);
    inline for (dag, 0..) |node, i| {
        switch (node) {
            .op2 => |op| {
                if (op.op == .div and is_zero(T, dag[op.lhs])) {
                    state.map[i] = op.lhs;
                }
            },
            else => {},
        }
    }
    rewrite_refs(T, state.nodes[0..], &state.map);
    return state.nodes;
}

// x / x = 1, unsafe
pub fn self_div(comptime T: type, comptime dag: []const DAGNode(T)) [dag.len]DAGNode(T) {
    var state = base_result(T, dag);
    inline for (dag, 0..) |node, i| {
        switch (node) {
            .op2 => |op| {
                if (op.op == .div and op.lhs == op.rhs) {
                    state.nodes[i] = DAGNode(T){ .scalar_constant = 1 };
                }
            },
            else => {},
        }
    }
    rewrite_refs(T, state.nodes[0..], &state.map);
    return state.nodes;
}

test "has_add_zero" {
    const dag = comptime blk: {
        var b = Builder(f32, 10){};
        const v1 = b.x();
        const v2 = b.c(0.0);
        const v3 = b.add(v1, v2);
        b.output(v3);
        break :blk b.dag();
    };
    try std.testing.expect(has_add_zero(f32, &dag));

    const dag_b = comptime blk: {
        var b = Builder(f32, 10){};
        const v1 = b.c(0.0);
        const v2 = b.x();
        const v3 = b.add(v1, v2);
        b.output(v3);
        break :blk b.dag();
    };
    try std.testing.expect(has_add_zero(f32, &dag_b));
}

test "has_sub_zero" {
    const dag = comptime blk: {
        var b = Builder(f32, 10){};
        const v1 = b.x();
        const v2 = b.c(0.0);
        const v3 = b.sub(v1, v2);
        b.output(v3);
        break :blk b.dag();
    };
    try std.testing.expect(has_sub_zero(f32, &dag));

    const dag_b = comptime blk: {
        var b = Builder(f32, 10){};
        const v1 = b.c(0.0);
        const v2 = b.x();
        const v3 = b.sub(v1, v2);
        b.output(v3);
        break :blk b.dag();
    };
    try std.testing.expect(has_sub_zero(f32, &dag_b));
}

test "has_self_sub" {
    const dag = comptime blk: {
        var b = Builder(f32, 10){};
        const v1 = b.x();
        const v2 = b.sub(v1, v1);
        b.output(v2);
        break :blk b.dag();
    };
    try std.testing.expect(has_self_sub(f32, &dag));
}

test "has_mul_zero" {
    const dag = comptime blk: {
        var b = Builder(f32, 10){};
        const v1 = b.x();
        const v2 = b.c(0.0);
        const v3 = b.mul(v1, v2);
        b.output(v3);
        break :blk b.dag();
    };
    try std.testing.expect(has_mul_zero(f32, &dag));

    const dag_b = comptime blk: {
        var b = Builder(f32, 10){};
        const v1 = b.c(0.0);
        const v2 = b.x();
        const v3 = b.mul(v1, v2);
        b.output(v3);
        break :blk b.dag();
    };
    try std.testing.expect(has_mul_zero(f32, &dag_b));
}

test "has_mul_one" {
    const dag = comptime blk: {
        var b = Builder(f32, 10){};
        const v1 = b.x();
        const v2 = b.c(1.0);
        const v3 = b.mul(v1, v2);
        b.output(v3);
        break :blk b.dag();
    };
    try std.testing.expect(has_mul_one(f32, &dag));

    const dag_b = comptime blk: {
        var b = Builder(f32, 10){};
        const v1 = b.c(1.0);
        const v2 = b.x();
        const v3 = b.mul(v1, v2);
        b.output(v3);
        break :blk b.dag();
    };
    try std.testing.expect(has_mul_one(f32, &dag_b));
}

test "has_mul_neg" {
    const dag = comptime blk: {
        var b = Builder(f32, 10){};
        const v1 = b.x();
        const v2 = b.c(-1.0);
        const v3 = b.mul(v1, v2);
        b.output(v3);
        break :blk b.dag();
    };
    try std.testing.expect(has_mul_neg(f32, &dag));

    const dag_b = comptime blk: {
        var b = Builder(f32, 10){};
        const v1 = b.c(-1.0);
        const v2 = b.x();
        const v3 = b.mul(v1, v2);
        b.output(v3);
        break :blk b.dag();
    };
    try std.testing.expect(has_mul_neg(f32, &dag_b));
}

test "has_div_one" {
    const dag = comptime blk: {
        var b = Builder(f32, 10){};
        const v1 = b.x();
        const v2 = b.c(1.0);
        const v3 = b.div(v1, v2);
        b.output(v3);
        break :blk b.dag();
    };
    try std.testing.expect(has_div_one(f32, &dag));
}

test "has_zero_div" {
    const dag = comptime blk: {
        var b = Builder(f32, 10){};
        const v1 = b.c(0.0);
        const v2 = b.x();
        const v3 = b.div(v1, v2);
        b.output(v3);
        break :blk b.dag();
    };
    try std.testing.expect(has_zero_div(f32, &dag));
}

test "has_self_div" {
    const dag = comptime blk: {
        var b = Builder(f32, 10){};
        const v1 = b.x();
        const v2 = b.div(v1, v1);
        b.output(v2);
        break :blk b.dag();
    };
    try std.testing.expect(has_self_div(f32, &dag));
}

test "add_zero" {
    const test_dag = comptime blk: {
        var b = Builder(f32, 10){};
        const v1 = b.x();
        const v2 = b.c(0.0);
        const v3 = b.add(v1, v2);
        b.output(v3);
        break :blk b.dag();
    };
    const expected_dag = comptime blk: {
        var b = Builder(f32, 10){};
        const v1 = b.x();
        const v2 = b.c(0.0);
        _ = b.add(v1, v2);
        b.output(v1);
        break :blk b.dag();
    };
    const simplified = add_zero(f32, &test_dag);
    for (simplified, expected_dag) |node, expected| {
        try std.testing.expectEqual(expected, node);
    }
}

test "sub_zero" {
    const test_dag = comptime blk: {
        var b = Builder(f32, 10){};
        const v1 = b.x();
        const v2 = b.c(0.0);
        const v3 = b.sub(v1, v2);
        b.output(v3);
        break :blk b.dag();
    };
    const expected_dag = comptime blk: {
        var b = Builder(f32, 10){};
        const v1 = b.x();
        const v2 = b.c(0.0);
        _ = b.sub(v1, v2);
        b.output(v1);
        break :blk b.dag();
    };
    const simplified = sub_zero(f32, &test_dag);
    for (simplified, expected_dag) |node, expected| {
        try std.testing.expectEqual(expected, node);
    }
}

test "self_sub" {
    const test_dag = comptime blk: {
        var b = Builder(f32, 10){};
        const v1 = b.x();
        const v2 = b.sub(v1, v1);
        b.output(v2);
        break :blk b.dag();
    };
    const expected_dag = comptime blk: {
        var b = Builder(f32, 10){};
        _ = b.x();
        const v2 = b.c(0.0);
        b.output(v2);
        break :blk b.dag();
    };
    const simplified = self_sub(f32, &test_dag);
    for (simplified, expected_dag) |node, expected| {
        try std.testing.expectEqual(expected, node);
    }
}

test "mul_zero" {
    const test_dag = comptime blk: {
        var b = Builder(f32, 10){};
        const v1 = b.x();
        const v2 = b.c(0.0);
        const v3 = b.mul(v1, v2);
        b.output(v3);
        break :blk b.dag();
    };
    const expected_dag = comptime blk: {
        var b = Builder(f32, 10){};
        const v1 = b.x();
        const v2 = b.c(0.0);
        _ = b.mul(v1, v2);
        b.output(v2);
        break :blk b.dag();
    };
    const simplified = mul_zero(f32, &test_dag);
    for (simplified, expected_dag) |node, expected| {
        try std.testing.expectEqual(expected, node);
    }
}

test "mul_one" {
    const test_dag = comptime blk: {
        var b = Builder(f32, 10){};
        const v1 = b.x();
        const v2 = b.c(1.0);
        const v3 = b.mul(v1, v2);
        b.output(v3);
        break :blk b.dag();
    };
    const expected_dag = comptime blk: {
        var b = Builder(f32, 10){};
        const v1 = b.x();
        const v2 = b.c(1.0);
        _ = b.mul(v1, v2);
        b.output(v1);
        break :blk b.dag();
    };
    const simplified = mul_one(f32, &test_dag);
    for (simplified, expected_dag) |node, expected| {
        try std.testing.expectEqual(expected, node);
    }
}

test "mul_neg" {
    const test_dag = comptime blk: {
        var b = Builder(f32, 10){};
        const v1 = b.x();
        const v2 = b.c(-1.0);
        const v3 = b.mul(v1, v2);
        b.output(v3);
        break :blk b.dag();
    };
    const expected_dag = comptime blk: {
        var b = Builder(f32, 10){};
        const v1 = b.x();
        _ = b.c(-1.0);
        const v3 = b.neg(v1);
        b.output(v3);
        break :blk b.dag();
    };
    const simplified = mul_neg(f32, &test_dag);
    for (simplified, expected_dag) |node, expected| {
        try std.testing.expectEqual(expected, node);
    }
}

test "div_one" {
    const test_dag = comptime blk: {
        var b = Builder(f32, 10){};
        const v1 = b.x();
        const v2 = b.c(1.0);
        const v3 = b.div(v1, v2);
        b.output(v3);
        break :blk b.dag();
    };
    const expected_dag = comptime blk: {
        var b = Builder(f32, 10){};
        const v1 = b.x();
        const v2 = b.c(1.0);
        _ = b.div(v1, v2);
        b.output(v1);
        break :blk b.dag();
    };
    const simplified = div_one(f32, &test_dag);
    for (simplified, expected_dag) |node, expected| {
        try std.testing.expectEqual(expected, node);
    }
}

test "zero_div" {
    const test_dag = comptime blk: {
        var b = Builder(f32, 10){};
        const v1 = b.c(0.0);
        const v2 = b.x();
        const v3 = b.div(v1, v2);
        b.output(v3);
        break :blk b.dag();
    };
    const expected_dag = comptime blk: {
        var b = Builder(f32, 10){};
        const v1 = b.c(0.0);
        const v2 = b.x();
        _ = b.div(v1, v2);
        b.output(v1);
        break :blk b.dag();
    };
    const simplified = zero_div(f32, &test_dag);
    for (simplified, expected_dag) |node, expected| {
        try std.testing.expectEqual(expected, node);
    }
}

test "self_div" {
    const test_dag = comptime blk: {
        var b = Builder(f32, 10){};
        const v1 = b.x();
        const v2 = b.div(v1, v1);
        b.output(v2);
        break :blk b.dag();
    };
    const expected_dag = comptime blk: {
        var b = Builder(f32, 10){};
        _ = b.x();
        const v2 = b.c(1.0);
        b.output(v2);
        break :blk b.dag();
    };
    const simplified = self_div(f32, &test_dag);
    for (simplified, expected_dag) |node, expected| {
        try std.testing.expectEqual(expected, node);
    }
}
