const std = @import("std");
const dag_mod = @import("dag.zig");
const DAGNode = dag_mod.DAGNode;
const Builder = @import("dag_builder.zig").Builder;
const eval = @import("eval.zig").eval;

fn dag_constant(comptime T: type, comptime value: comptime_float) T {
    return switch (@typeInfo(T)) {
        .float, .comptime_float => @as(T, value),
        .vector => |info| @splat(@as(info.child, value)),
        else => @compileError("grad only supports float or vector-of-float DAG values"),
    };
}

test "dag_constant" {
    try std.testing.expectEqual(@as(f64, 2.0), dag_constant(f64, 2.0));

    const v = dag_constant(@Vector(2, f64), 2.0);
    try std.testing.expectEqual(@Vector(2, f64){ 2.0, 2.0 }, v);
}

fn unary_derivative(comptime T: type, b: anytype, op: dag_mod.Op1, node: usize) usize {
    return switch (op) {
        .neg => b.c(dag_constant(T, -1.0)),
        .abs => b.div(node, b.abs(node)),
        .exp => b.exp(node),
        .log => b.div(b.c(dag_constant(T, 1.0)), node),
        .sqrt => b.div(b.c(dag_constant(T, 1.0)), b.mul(b.c(dag_constant(T, 2.0)), b.sqrt(node))),
        .sin => b.cos(node),
        .cos => b.neg(b.sin(node)),
        .tan => blk: {
            const cos_node = b.cos(node);
            break :blk b.div(b.c(dag_constant(T, 1.0)), b.mul(cos_node, cos_node));
        },
    };
}

test "unary_derivative" {
    inline for ([_]dag_mod.Op1{ .neg, .abs, .exp, .log, .sqrt, .sin, .cos, .tan }) |op| {
        const derivative_dag = comptime blk: {
            var b = Builder(f64, 16){};
            const x = b.x();
            const d = unary_derivative(f64, &b, op, x);
            b.output(d);
            break :blk b.dag();
        };

        var input = [_]f64{2.0};
        const actual = eval(f64, &derivative_dag, &input)[0];
        const expected = switch (op) {
            .neg => -1.0,
            .abs => 1.0,
            .exp => @exp(input[0]),
            .log => 1.0 / input[0],
            .sqrt => 1.0 / (2.0 * @sqrt(input[0])),
            .sin => @cos(input[0]),
            .cos => -@sin(input[0]),
            .tan => 1.0 / (@cos(input[0]) * @cos(input[0])),
        };
        try std.testing.expectApproxEqAbs(expected, actual, 1e-12);
    }
}

fn binary_derivative(comptime T: type, b: anytype, op: dag_mod.Op2, lhs: usize, rhs: usize) struct {
    lhs_part: usize,
    rhs_part: usize,
} {
    return switch (op) {
        .add => .{
            .lhs_part = b.c(dag_constant(T, 1.0)),
            .rhs_part = b.c(dag_constant(T, 1.0)),
        },
        .sub => .{
            .lhs_part = b.c(dag_constant(T, 1.0)),
            .rhs_part = b.c(dag_constant(T, -1.0)),
        },
        .mul => .{
            .lhs_part = rhs,
            .rhs_part = lhs,
        },
        .div => .{
            .lhs_part = b.div(b.c(dag_constant(T, 1.0)), rhs),
            .rhs_part = b.neg(b.div(lhs, b.mul(rhs, rhs))),
        },
    };
}

test "binary_derivative" {
    inline for ([_]dag_mod.Op2{ .add, .sub, .mul, .div }) |op| {
        const derivative_dag = comptime blk: {
            var b = Builder(f64, 32){};
            const x = b.x();
            const y = b.x();
            const d = binary_derivative(f64, &b, op, x, y);
            b.output(d.lhs_part);
            b.output(d.rhs_part);
            break :blk b.dag();
        };

        var input = [_]f64{ 6.0, 3.0 };
        const actual = eval(f64, &derivative_dag, &input);
        const expected = switch (op) {
            .add => .{ 1.0, 1.0 },
            .sub => .{ 1.0, -1.0 },
            .mul => .{ input[1], input[0] },
            .div => .{ 1.0 / input[1], -input[0] / (input[1] * input[1]) },
        };
        try std.testing.expectApproxEqAbs(expected[0], actual[0], 1e-12);
        try std.testing.expectApproxEqAbs(expected[1], actual[1], 1e-12);
    }
}

fn graph_counts(comptime T: type, comptime dag: []const DAGNode(T)) struct {
    values: usize,
    outputs: usize,
    op1: usize,
    op2: usize,
} {
    @setEvalBranchQuota(dag.len);
    var values: usize = 0;
    var op1: usize = 0;
    var op2: usize = 0;

    inline for (dag) |node| {
        switch (node) {
            .output => {},
            .op1 => {
                values += 1;
                op1 += 1;
            },
            .op2 => {
                values += 1;
                op2 += 1;
            },
            else => values += 1,
        }
    }

    return .{
        .values = values,
        .outputs = dag_mod.output_size(T, dag),
        .op1 = op1,
        .op2 = op2,
    };
}

fn has_normalized_commutative_op2(comptime T: type, comptime dag: []const DAGNode(T)) bool {
    inline for (dag) |node| {
        switch (node) {
            .op2 => |op| switch (op.op) {
                .add, .mul => if (op.lhs > op.rhs) return false,
                else => {},
            },
            else => {},
        }
    }
    return true;
}

test "has_normalized_commutative_op2" {
    const valid = comptime [_]DAGNode(f64){dag_mod.op2_node(f64, 2, 1, .add)};
    try std.testing.expect(has_normalized_commutative_op2(f64, &valid));

    const invalid = comptime [_]DAGNode(f64){.{ .op2 = .{ .lhs = 2, .rhs = 1, .op = .mul } }};
    try std.testing.expect(!has_normalized_commutative_op2(f64, &invalid));
}

fn eval_branch_quota(comptime T: type, comptime dag: []const DAGNode(T)) comptime_int {
    const counts = graph_counts(T, dag);
    comptime var values: usize = counts.values;
    comptime var outputs: usize = counts.outputs + 1;
    if (values == 0) values = 1;
    if (outputs == 0) outputs = 1;

    comptime var scale: usize = values;
    scale *= outputs;

    comptime var quota: usize = scale;
    quota *= scale;
    quota *= 100;
    return @max(quota, 100_000);
}

fn constant_equal(comptime T: type, value: T, comptime expected: comptime_float) bool {
    return switch (@typeInfo(T)) {
        .float, .comptime_float => value == @as(T, expected),
        .vector => |info| @reduce(.And, value == @as(T, @splat(@as(info.child, expected)))),
        else => false,
    };
}

test "constant_equal" {
    try std.testing.expect(constant_equal(f64, 1.0, 1.0));
    try std.testing.expect(!constant_equal(f64, 2.0, 1.0));
    try std.testing.expect(constant_equal(@Vector(2, f64), .{ 0.0, 0.0 }, 0.0));
}

fn node_is_constant(comptime T: type, b: anytype, index: usize, comptime expected: comptime_float) bool {
    return switch (b.nodes[index]) {
        .scalar_constant => |value| constant_equal(T, value, expected),
        else => false,
    };
}

test "node_is_constant" {
    comptime {
        var b = Builder(f64, 4){};
        const zero = b.c(0.0);
        const one = b.c(1.0);
        try std.testing.expect(node_is_constant(f64, &b, zero, 0.0));
        try std.testing.expect(!node_is_constant(f64, &b, one, 0.0));
    }
}

fn add_node(comptime T: type, b: anytype, lhs: usize, rhs: usize) usize {
    if (node_is_constant(T, b, lhs, 0.0)) return rhs;
    if (node_is_constant(T, b, rhs, 0.0)) return lhs;
    return b.add(lhs, rhs);
}

test "add_node" {
    const test_dag = comptime blk: {
        var b = Builder(f64, 8){};
        const x = b.x();
        const zero = b.c(0.0);
        try std.testing.expectEqual(x, add_node(f64, &b, x, zero));
        try std.testing.expectEqual(x, add_node(f64, &b, zero, x));
        break :blk b.dag();
    };
    try std.testing.expectEqual(@as(usize, 2), test_dag.len);
}

fn neg_node(comptime T: type, b: anytype, node: usize, zero: usize) usize {
    if (node_is_constant(T, b, node, 0.0)) return zero;
    switch (b.nodes[node]) {
        .op1 => |op| if (op.op == .neg) return op.node,
        else => {},
    }
    return b.neg(node);
}

test "neg_node" {
    const test_dag = comptime blk: {
        var b = Builder(f64, 8){};
        const x = b.x();
        const zero = b.c(0.0);
        const neg_x = b.neg(x);
        try std.testing.expectEqual(zero, neg_node(f64, &b, zero, zero));
        try std.testing.expectEqual(x, neg_node(f64, &b, neg_x, zero));
        break :blk b.dag();
    };
    try std.testing.expectEqual(@as(usize, 3), test_dag.len);
}

fn mul_node(comptime T: type, b: anytype, lhs: usize, rhs: usize, zero: usize) usize {
    if (node_is_constant(T, b, lhs, 0.0) or node_is_constant(T, b, rhs, 0.0)) return zero;
    if (node_is_constant(T, b, lhs, 1.0)) return rhs;
    if (node_is_constant(T, b, rhs, 1.0)) return lhs;
    if (node_is_constant(T, b, lhs, -1.0)) return neg_node(T, b, rhs, zero);
    if (node_is_constant(T, b, rhs, -1.0)) return neg_node(T, b, lhs, zero);
    return b.mul(lhs, rhs);
}

test "mul_node" {
    const test_dag = comptime blk: {
        var b = Builder(f64, 8){};
        const x = b.x();
        const zero = b.c(0.0);
        const one = b.c(1.0);
        const neg_one = b.c(-1.0);
        try std.testing.expectEqual(zero, mul_node(f64, &b, x, zero, zero));
        try std.testing.expectEqual(x, mul_node(f64, &b, x, one, zero));
        _ = mul_node(f64, &b, x, neg_one, zero);
        break :blk b.dag();
    };
    try std.testing.expectEqual(@as(usize, 5), test_dag.len);
}

fn div_node(comptime T: type, b: anytype, lhs: usize, rhs: usize, zero: usize) usize {
    if (node_is_constant(T, b, lhs, 0.0)) return zero;
    if (node_is_constant(T, b, rhs, 1.0)) return lhs;
    return b.div(lhs, rhs);
}

test "div_node" {
    const test_dag = comptime blk: {
        var b = Builder(f64, 8){};
        const x = b.x();
        const zero = b.c(0.0);
        const one = b.c(1.0);
        try std.testing.expectEqual(zero, div_node(f64, &b, zero, x, zero));
        try std.testing.expectEqual(x, div_node(f64, &b, x, one, zero));
        break :blk b.dag();
    };
    try std.testing.expectEqual(@as(usize, 3), test_dag.len);
}

fn add_adjoint(comptime T: type, b: anytype, derivatives: []usize, target: usize, adjoint: usize, local: usize, zero: usize, one: usize) void {
    if (adjoint == zero or local == zero) return;

    const contribution = if (adjoint == one)
        local
    else if (local == one)
        adjoint
    else
        mul_node(T, b, adjoint, local, zero);
    derivatives[target] = add_node(T, b, derivatives[target], contribution);
}

fn add_negative_adjoint(comptime T: type, b: anytype, derivatives: []usize, target: usize, adjoint: usize, zero: usize) void {
    if (adjoint == zero) return;

    const contribution = neg_node(T, b, adjoint, zero);
    derivatives[target] = add_node(T, b, derivatives[target], contribution);
}

fn apply_unary_adjoint(comptime T: type, b: anytype, derivatives: []usize, op: dag_mod.Op1, child: usize, current: usize, child_derivative: usize, current_derivative: usize, zero: usize, one: usize) void {
    if (current_derivative == zero) return;

    if (op == .neg) {
        add_negative_adjoint(T, b, derivatives, child_derivative, current_derivative, zero);
        return;
    }

    const local = switch (op) {
        .neg => unreachable,
        .abs => div_node(T, b, child, current, zero),
        .exp => current,
        .log => div_node(T, b, one, child, zero),
        .sqrt => div_node(T, b, one, mul_node(T, b, b.c(dag_constant(T, 2.0)), current, zero), zero),
        .sin => b.cos(child),
        .cos => b.neg(b.sin(child)),
        .tan => add_node(T, b, one, mul_node(T, b, current, current, zero)),
    };
    add_adjoint(T, b, derivatives, child_derivative, current_derivative, local, zero, one);
}

test "apply_unary_adjoint" {
    const test_dag = comptime blk: {
        var b = Builder(f64, 16){};
        const x = b.x();
        const y = b.exp(x);
        const zero = b.c(0.0);
        const one = b.c(1.0);
        var derivatives = [_]usize{ zero, one };
        apply_unary_adjoint(f64, &b, &derivatives, .exp, x, y, 0, derivatives[1], zero, one);
        b.output(derivatives[0]);
        break :blk b.dag();
    };
    var input = [_]f64{2.0};
    const actual = eval(f64, &test_dag, &input)[0];
    try std.testing.expectApproxEqAbs(@exp(input[0]), actual, 1e-12);
}

fn apply_binary_adjoint(comptime T: type, b: anytype, derivatives: []usize, op: dag_mod.Op2, lhs: usize, rhs: usize, lhs_derivative: usize, rhs_derivative: usize, current_derivative: usize, zero: usize, one: usize) void {
    if (current_derivative == zero) return;

    switch (op) {
        .add => {
            add_adjoint(T, b, derivatives, lhs_derivative, current_derivative, one, zero, one);
            add_adjoint(T, b, derivatives, rhs_derivative, current_derivative, one, zero, one);
        },
        .sub => {
            add_adjoint(T, b, derivatives, lhs_derivative, current_derivative, one, zero, one);
            add_negative_adjoint(T, b, derivatives, rhs_derivative, current_derivative, zero);
        },
        .mul => {
            add_adjoint(T, b, derivatives, lhs_derivative, current_derivative, rhs, zero, one);
            add_adjoint(T, b, derivatives, rhs_derivative, current_derivative, lhs, zero, one);
        },
        .div => {
            add_adjoint(T, b, derivatives, lhs_derivative, current_derivative, div_node(T, b, one, rhs, zero), zero, one);
            add_adjoint(T, b, derivatives, rhs_derivative, current_derivative, neg_node(T, b, div_node(T, b, lhs, mul_node(T, b, rhs, rhs, zero), zero), zero), zero, one);
        },
    }
}

test "apply_binary_adjoint" {
    const test_dag = comptime blk: {
        var b = Builder(f64, 16){};
        const x = b.x();
        const y = b.x();
        const zero = b.c(0.0);
        const one = b.c(1.0);
        var derivatives = [_]usize{ zero, zero, one };
        apply_binary_adjoint(f64, &b, &derivatives, .mul, x, y, 0, 1, derivatives[2], zero, one);
        b.output(derivatives[0]);
        b.output(derivatives[1]);
        break :blk b.dag();
    };
    var input = [_]f64{ 2.0, 3.0 };
    const actual = eval(f64, &test_dag, &input);
    try std.testing.expectApproxEqAbs(input[1], actual[0], 1e-12);
    try std.testing.expectApproxEqAbs(input[0], actual[1], 1e-12);
}

test "add_adjoint" {
    const test_dag = comptime blk: {
        var b = Builder(f64, 16){};
        const x = b.x();
        const zero = b.c(0.0);
        const one = b.c(1.0);
        var derivatives = [_]usize{zero};
        add_adjoint(f64, &b, &derivatives, 0, zero, x, zero, one);
        try std.testing.expectEqual(zero, derivatives[0]);
        add_adjoint(f64, &b, &derivatives, 0, one, x, zero, one);
        try std.testing.expectEqual(x, derivatives[0]);
        break :blk b.dag();
    };
    const expected_dag = [_]DAGNode(f64){
        .{ .scalar_parameter = 0 },
        .{ .scalar_constant = 0.0 },
        .{ .scalar_constant = 1.0 },
    };
    try std.testing.expectEqualSlices(DAGNode(f64), &expected_dag, &test_dag);
}

test "graph_counts" {
    const test_dag = comptime blk: {
        var b = Builder(f64, 8){};
        const x = b.x();
        const y = b.x();
        const z = b.add(b.sin(x), y);
        b.output(z);
        break :blk b.dag();
    };
    const counts = graph_counts(f64, &test_dag);
    try std.testing.expectEqual(@as(usize, 4), counts.values);
    try std.testing.expectEqual(@as(usize, 1), counts.outputs);
    try std.testing.expectEqual(@as(usize, 1), counts.op1);
    try std.testing.expectEqual(@as(usize, 1), counts.op2);
}

fn grad_capacity(comptime T: type, comptime dag: []const DAGNode(T)) usize {
    const counts = graph_counts(T, dag);
    return counts.values + 2 + counts.outputs * (counts.values * counts.values * 16 + dag_mod.input_size(T, dag));
}

test "grad_capacity" {
    const test_dag = [_]DAGNode(f64){
        .{ .scalar_parameter = 0 },
        .{ .op1 = .{ .node = 0, .op = .sin } },
        .{ .output = .{ .index = 0, .node = 1 } },
    };
    try std.testing.expectEqual(@as(usize, 69), grad_capacity(f64, &test_dag));
}

fn output_nodes(comptime T: type, comptime dag: []const DAGNode(T)) [dag_mod.output_size(T, dag)]usize {
    @setEvalBranchQuota(dag.len);
    var nodes: [dag_mod.output_size(T, dag)]usize = undefined;
    inline for (dag) |node| {
        switch (node) {
            .output => |out| nodes[out.index] = out.node,
            else => {},
        }
    }
    return nodes;
}

fn mark_needed(comptime T: type, comptime dag: []const DAGNode(T), needed: *[dag.len]bool, index: usize) void {
    if (needed[index]) return;
    needed[index] = true;
    switch (dag[index]) {
        .op1 => |op| mark_needed(T, dag, needed, op.node),
        .op2 => |op| {
            mark_needed(T, dag, needed, op.lhs);
            mark_needed(T, dag, needed, op.rhs);
        },
        else => {},
    }
}

fn grad_primal_needed(comptime T: type, comptime dag: []const DAGNode(T)) [dag.len]bool {
    @setEvalBranchQuota(eval_branch_quota(T, dag));
    var needed = [_]bool{false} ** dag.len;

    inline for (dag, 0..) |node, i| {
        switch (node) {
            .scalar_parameter => {},
            .op1 => |op| switch (op.op) {
                .abs, .exp, .sqrt, .tan => mark_needed(T, dag, &needed, i),
                else => {},
            },
            .op2 => |op| switch (op.op) {
                .mul, .div => {
                    mark_needed(T, dag, &needed, op.lhs);
                    mark_needed(T, dag, &needed, op.rhs);
                },
                else => {},
            },
            .output => {},
            .scalar_constant => {},
        }
    }

    return needed;
}

test "grad_primal_needed" {
    const test_dag = comptime blk: {
        var b = Builder(f64, 16){};
        const x = b.x();
        const y = b.x();
        const unused_output_chain = b.add(x, y);
        const product = b.mul(x, y);
        b.output(unused_output_chain);
        b.output(product);
        break :blk b.dag();
    };
    const needed = grad_primal_needed(f64, &test_dag);
    try std.testing.expectEqualSlices(bool, &[_]bool{ true, true, false, false, false, false }, &needed);
}

test "output_nodes" {
    const test_dag = comptime blk: {
        var b = Builder(f64, 8){};
        const x = b.x();
        const y = b.sin(x);
        b.output(y);
        b.output(x);
        break :blk b.dag();
    };
    const nodes = output_nodes(f64, &test_dag);
    try std.testing.expectEqualSlices(usize, &[_]usize{ 1, 0 }, &nodes);
}

fn build_grad_nodes(comptime T: type, comptime dag: []const DAGNode(T), comptime capacity: usize) struct {
    len: usize,
    nodes: [capacity]DAGNode(T),
} {
    @setEvalBranchQuota(eval_branch_quota(T, dag));
    dag_mod.validate_dag(T, dag);

    var b = Builder(T, capacity){};
    var old_to_new: [dag.len]usize = undefined;
    const primal_needed = grad_primal_needed(T, dag);

    inline for (dag, 0..) |node, i| {
        switch (node) {
            .scalar_constant => |value| {
                if (primal_needed[i]) old_to_new[i] = b.c(value);
            },
            .scalar_parameter => |index| old_to_new[i] = b.append(.{ .scalar_parameter = index }),
            .op1 => |op| {
                if (primal_needed[i]) old_to_new[i] = b.op1(old_to_new[op.node], op.op);
            },
            .op2 => |op| {
                if (primal_needed[i]) old_to_new[i] = b.op2(old_to_new[op.lhs], old_to_new[op.rhs], op.op);
            },
            .output => {},
        }
    }

    const zero = b.c(dag_constant(T, 0.0));
    const one = b.c(dag_constant(T, 1.0));
    const outs = output_nodes(T, dag);

    inline for (outs) |target_node| {
        var derivatives: [dag.len]usize = undefined;
        for (&derivatives) |*derivative| {
            derivative.* = zero;
        }
        derivatives[target_node] = one;

        inline for (0..dag.len) |offset| {
            const i = dag.len - 1 - offset;
            switch (dag[i]) {
                .op1 => |op| {
                    const child = old_to_new[op.node];
                    const current = old_to_new[i];
                    apply_unary_adjoint(T, &b, &derivatives, op.op, child, current, op.node, derivatives[i], zero, one);
                },
                .op2 => |op| {
                    const lhs = old_to_new[op.lhs];
                    const rhs = old_to_new[op.rhs];
                    apply_binary_adjoint(T, &b, &derivatives, op.op, lhs, rhs, op.lhs, op.rhs, derivatives[i], zero, one);
                },
                else => {},
            }
        }

        inline for (dag, 0..) |node, i| {
            switch (node) {
                .scalar_parameter => |input_index| {
                    _ = input_index;
                    b.output(derivatives[i]);
                },
                else => {},
            }
        }
    }

    return .{ .len = b.len, .nodes = b.nodes };
}

test "build_grad_nodes" {
    const test_dag = comptime blk: {
        var b = Builder(f64, 16){};
        const x = b.x();
        const y = b.x();
        b.output(b.mul(x, y));
        break :blk b.dag();
    };
    const built = comptime build_grad_nodes(f64, &test_dag, grad_capacity(f64, &test_dag));
    const nodes = built.nodes[0..built.len];
    var input = [_]f64{ 2.0, 3.0 };
    const actual = eval(f64, nodes, &input);
    try std.testing.expectEqual(@as(usize, 2), actual.len);
    try std.testing.expectApproxEqAbs(3.0, actual[0], 1e-12);
    try std.testing.expectApproxEqAbs(2.0, actual[1], 1e-12);
}

fn grad_node_count(comptime T: type, comptime dag: []const DAGNode(T)) usize {
    @setEvalBranchQuota(eval_branch_quota(T, dag));
    const built = comptime build_grad_nodes(T, dag, grad_capacity(T, dag));
    return built.len;
}

test "grad_node_count" {
    const test_dag = [_]DAGNode(f64){
        .{ .scalar_parameter = 0 },
        .{ .op1 = .{ .node = 0, .op = .sin } },
        .{ .output = .{ .index = 0, .node = 1 } },
    };
    try std.testing.expectEqual(@as(usize, 5), grad_node_count(f64, &test_dag));
}

pub fn grad(comptime T: type, comptime dag: []const DAGNode(T)) struct {
    rows: usize,
    cols: usize,
    nodes: [grad_node_count(T, dag)]DAGNode(T),
} {
    @setEvalBranchQuota(eval_branch_quota(T, dag));
    const built = comptime build_grad_nodes(T, dag, grad_capacity(T, dag));
    const len = comptime grad_node_count(T, dag);
    return .{
        .rows = dag_mod.output_size(T, dag),
        .cols = dag_mod.input_size(T, dag),
        .nodes = built.nodes[0..len].*,
    };
}

test "grad" {
    const test_dag = comptime blk: {
        var b = Builder(f64, 16){};
        const x = b.x();
        const y = b.x();
        b.output(b.add(b.log(x), b.mul(x, y)));
        b.output(b.sub(b.sin(y), x));
        break :blk b.dag();
    };

    const g = comptime grad(f64, &test_dag);
    try std.testing.expect(has_normalized_commutative_op2(f64, &g.nodes));
    try std.testing.expectEqual(@as(usize, 2), g.rows);
    try std.testing.expectEqual(@as(usize, 2), g.cols);

    var input = [_]f64{ 2.0, 3.0 };
    const actual = eval(f64, &g.nodes, &input);

    try std.testing.expectEqual(@as(usize, 4), actual.len);
    try std.testing.expectApproxEqAbs(1.0 / input[0] + input[1], actual[0], 1e-12);
    try std.testing.expectApproxEqAbs(input[0], actual[1], 1e-12);
    try std.testing.expectApproxEqAbs(-1.0, actual[2], 1e-12);
    try std.testing.expectApproxEqAbs(@cos(input[1]), actual[3], 1e-12);
}

test "grad reuses needed primal op1" {
    const test_dag = comptime blk: {
        var b = Builder(f64, 8){};
        const x = b.x();
        b.output(b.exp(x));
        break :blk b.dag();
    };

    const g = comptime grad(f64, &test_dag);
    var input = [_]f64{2.0};
    const actual = eval(f64, &g.nodes, &input);
    try std.testing.expectEqual(@as(usize, 1), actual.len);
    try std.testing.expectApproxEqAbs(@exp(input[0]), actual[0], 1e-12);
}
