const std = @import("std");
const dag_mod = @import("../dag.zig");
const DAGNode = dag_mod.DAGNode;
const DAGWriter = @import("../dag_writer.zig").DAGWriter;
const Op1 = @import("../dag.zig").Op1;

const op1_simplify = enum {
    neg_neg, // -(-x) = x
    abs_abs, // abs(abs(x)) = abs(x)
    cos_neg, // cos(-x) = cos(x)
    log_exp, // log(exp(x)) = x
    exp_log, // exp(log(x)) = x, x > 0
};

fn is_op1(comptime T: type, node: DAGNode(T), comptime op: Op1) bool {
    return switch (node) {
        .op1 => |op1| op1.op == op,
        else => false,
    };
}

fn inner_op1_node(comptime T: type, node: DAGNode(T)) usize {
    return switch (node) {
        .op1 => |op1| op1.node,
        else => unreachable,
    };
}

fn has_pattern(comptime T: type, comptime dag: []const DAGNode(T), comptime outer: Op1, comptime inner: Op1) bool {
    @setEvalBranchQuota(dag.len);
    for (dag) |node| {
        switch (node) {
            .op1 => |op1| {
                if (op1.op == outer and is_op1(T, dag[op1.node], inner)) return true;
            },
            else => {},
        }
    }
    return false;
}

fn pattern_index(comptime T: type, comptime dag: []const DAGNode(T), comptime outer: Op1, comptime inner: Op1) ?usize {
    @setEvalBranchQuota(dag.len);
    for (dag, 0..) |node, i| {
        switch (node) {
            .op1 => |op1| {
                if (is_op1(T, dag[op1.node], outer) and is_op1(T, dag[inner_op1_node(dag[op1.node])], inner)) return i;
            },
            else => {},
        }
    }
    return null;
}

pub fn has_neg_neg(comptime T: type, comptime dag: []const DAGNode(T)) bool {
    return has_pattern(T, dag, .neg, .neg);
}

pub fn has_abs_abs(comptime T: type, comptime dag: []const DAGNode(T)) bool {
    return has_pattern(T, dag, .abs, .abs);
}

pub fn has_cos_neg(comptime T: type, comptime dag: []const DAGNode(T)) bool {
    return has_pattern(T, dag, .cos, .neg);
}

pub fn has_log_exp(comptime T: type, comptime dag: []const DAGNode(T)) bool {
    return has_pattern(T, dag, .log, .exp);
}

pub fn has_exp_log(comptime T: type, comptime dag: []const DAGNode(T)) bool {
    return has_pattern(T, dag, .exp, .log);
}

fn simplify_target(comptime T: type, comptime dag: []const DAGNode(T), node: DAGNode(T), comptime rule: op1_simplify) ?usize {
    return switch (node) {
        .op1 => |op1| switch (rule) {
            .neg_neg => if (op1.op == .neg and is_op1(T, dag[op1.node], .neg)) inner_op1_node(T, dag[op1.node]) else null,
            .abs_abs => if (op1.op == .abs and is_op1(T, dag[op1.node], .abs)) op1.node else null,
            .cos_neg => if (op1.op == .cos and is_op1(T, dag[op1.node], .neg)) inner_op1_node(T, dag[op1.node]) else null,
            .log_exp => if (op1.op == .log and is_op1(T, dag[op1.node], .exp)) inner_op1_node(T, dag[op1.node]) else null,
            .exp_log => if (op1.op == .exp and is_op1(T, dag[op1.node], .log)) inner_op1_node(T, dag[op1.node]) else null,
        },
        else => null,
    };
}

fn resolve_map(map: []const usize, index: usize) usize {
    var current = index;
    while (map[current] != current) {
        current = map[current];
    }
    return current;
}

fn apply_op1_simplify(comptime T: type, comptime dag: []const DAGNode(T), comptime rule: op1_simplify) [dag.len]DAGNode(T) {
    var result: [dag.len]DAGNode(T) = dag[0..dag.len].*;
    var map: [dag.len]usize = undefined;
    inline for (0..dag.len) |i| {
        map[i] = i;
    }

    inline for (dag, 0..) |node, i| {
        switch (rule) {
            .cos_neg => switch (node) {
                .op1 => |op1| {
                    if (op1.op == .cos and is_op1(T, dag[op1.node], .neg)) {
                        result[i] = DAGNode(T){ .op1 = .{ .node = inner_op1_node(T, dag[op1.node]), .op = .cos } };
                    }
                },
                else => {},
            },
            else => if (simplify_target(T, dag, node, rule)) |target| {
                map[i] = target;
            },
        }
    }

    inline for (result, 0..) |node, i| {
        result[i] = switch (node) {
            .op1 => |op1| DAGNode(T){ .op1 = .{ .node = resolve_map(&map, op1.node), .op = op1.op } },
            .op2 => |op2| dag_mod.op2_node(T, resolve_map(&map, op2.lhs), resolve_map(&map, op2.rhs), op2.op),
            .output => |output| DAGNode(T){ .output = .{ .index = output.index, .node = resolve_map(&map, output.node) } },
            else => node,
        };
    }
    return result;
}

// fn my_apply_op1_simplify(comptime T: type, comptime dag: []const DAGNode(T), comptime rule: op1_simplify) [dag.len]DAGNode(T) {
//     var map: [dag.len]usize = @import("rewire.zig").default_map(dag.len);
//     switch (rule) {
//         .neg_neg => {
//             if (pattern_index(T, dag, .neg, .neg)) |index| {
//                 map[index] = inner_op1_node(T, dag[inner_op1_node(T, dag[index])]);
//             }
//         },
//         .abs_abs => {
//             if (pattern_index(T, dag, .abs, .abs)) |index| {
//                 map[index] = inner_op1_node(T, dag[index]);
//             }
//         },
//         .cos_neg => {
//             if (node == .op1 and is_op1(T, dag[node.op1.node], .cos) and is_op1(T, dag[node.op1.node], .neg)) {
//                 map[i] = inner_op1_node(T, dag[node.op1.node]);
//             }
//         },
//         .log_exp => {
//             if (node == .op1 and is_op1(T, dag[node.op1.node], .log) and is_op1(T, dag[node.op1.node], .exp)) {
//                 map[i] = inner_op1_node(T, dag[node.op1.node]);
//             }
//         },
//         .exp_log => {
//             if (node == .op1 and is_op1(T, dag[node.op1.node], .exp) and is_op1(T, dag[node.op1.node], .log)) {
//                 map[i] = inner_op1_node(T, dag[node.op1.node]);
//             }
//         },
//         else => {},
//     }
//     return @import("rewire.zig").rewire(T, dag, map);
// }

pub fn neg_neg(comptime T: type, comptime dag: []const DAGNode(T)) [dag.len]DAGNode(T) {
    return apply_op1_simplify(T, dag, .neg_neg);
}

pub fn abs_abs(comptime T: type, comptime dag: []const DAGNode(T)) [dag.len]DAGNode(T) {
    return apply_op1_simplify(T, dag, .abs_abs);
}

pub fn cos_neg(comptime T: type, comptime dag: []const DAGNode(T)) [dag.len]DAGNode(T) {
    return apply_op1_simplify(T, dag, .cos_neg);
}

pub fn log_exp(comptime T: type, comptime dag: []const DAGNode(T)) [dag.len]DAGNode(T) {
    return apply_op1_simplify(T, dag, .log_exp);
}

pub fn exp_log(comptime T: type, comptime dag: []const DAGNode(T)) [dag.len]DAGNode(T) {
    return apply_op1_simplify(T, dag, .exp_log);
}

test "neg_neg" {
    const test_dag = comptime blk: {
        var b = DAGWriter(f64, 100){};
        const v1 = b.x();
        const v2 = b.neg(v1);
        const v3 = b.neg(v2);
        b.output(v3);
        break :blk b.dag();
    };

    const expected_dag = comptime blk: {
        var b = DAGWriter(f64, 100){};
        const v1 = b.x();
        const v2 = b.neg(v1);
        _ = b.neg(v2);
        b.output(v1);
        break :blk b.dag();
    };
    try std.testing.expect(has_neg_neg(f64, &test_dag));
    const simplified = neg_neg(f64, &test_dag);
    for (simplified, expected_dag) |node, expected| {
        try std.testing.expectEqual(expected, node);
    }
}

test "abs_abs" {
    const test_dag = comptime blk: {
        var b = DAGWriter(f64, 100){};
        const v1 = b.x();
        const v2 = b.abs(v1);
        const v3 = b.abs(v2);
        b.output(v3);
        break :blk b.dag();
    };
    const expected_dag = comptime blk: {
        var b = DAGWriter(f64, 100){};
        const v1 = b.x();
        const v2 = b.abs(v1);
        _ = b.abs(v2);
        b.output(v2);
        break :blk b.dag();
    };
    try std.testing.expect(has_abs_abs(f64, &test_dag));
    const simplified = abs_abs(f64, &test_dag);
    for (simplified, expected_dag) |node, expected| {
        try std.testing.expectEqual(expected, node);
    }
}

test "cos_neg" {
    const test_dag = comptime blk: {
        var b = DAGWriter(f64, 100){};
        const v1 = b.x();
        const v2 = b.neg(v1);
        const v3 = b.cos(v2);
        b.output(v3);
        break :blk b.dag();
    };
    const expected_dag = comptime blk: {
        var b = DAGWriter(f64, 100){};
        const v1 = b.x();
        _ = b.neg(v1);
        const v3 = b.cos(v1);
        b.output(v3);
        break :blk b.dag();
    };
    try std.testing.expect(has_cos_neg(f64, &test_dag));
    const simplified = cos_neg(f64, &test_dag);
    for (simplified, expected_dag) |node, expected| {
        try std.testing.expectEqual(expected, node);
    }
}

test "log_exp" {
    const test_dag = comptime blk: {
        var b = DAGWriter(f64, 100){};
        const v1 = b.x();
        const v2 = b.exp(v1);
        const v3 = b.log(v2);
        b.output(v3);
        break :blk b.dag();
    };
    const expected_dag = comptime blk: {
        var b = DAGWriter(f64, 100){};
        const v1 = b.x();
        const v2 = b.exp(v1);
        _ = b.log(v2);
        b.output(v1);
        break :blk b.dag();
    };
    try std.testing.expect(has_log_exp(f64, &test_dag));
    const simplified = log_exp(f64, &test_dag);
    for (simplified, expected_dag) |node, expected| {
        try std.testing.expectEqual(expected, node);
    }
}

test "exp_log" {
    const test_dag = comptime blk: {
        var b = DAGWriter(f64, 100){};
        const v1 = b.x();
        const v2 = b.log(v1);
        const v3 = b.exp(v2);
        b.output(v3);
        break :blk b.dag();
    };
    const expected_dag = comptime blk: {
        var b = DAGWriter(f64, 100){};
        const v1 = b.x();
        const v2 = b.log(v1);
        _ = b.exp(v2);
        b.output(v1);
        break :blk b.dag();
    };
    try std.testing.expect(has_exp_log(f64, &test_dag));
    const simplified = exp_log(f64, &test_dag);
    for (simplified, expected_dag) |node, expected| {
        try std.testing.expectEqual(expected, node);
    }
}
