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
    var nodes: [dag_mod.output_size(T, dag)]usize = undefined;
    inline for (dag) |node| {
        switch (node) {
            .output => |out| nodes[out.index] = out.node,
            else => {},
        }
    }
    return nodes;
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
    dag_mod.validate_dag(T, dag);

    var b = Builder(T, capacity){};
    var old_to_new: [dag.len]usize = undefined;

    inline for (dag, 0..) |node, i| {
        switch (node) {
            .scalar_constant => |value| old_to_new[i] = b.c(value),
            .scalar_parameter => |index| old_to_new[i] = b.append(.{ .scalar_parameter = index }),
            .op1 => |op| old_to_new[i] = b.op1(old_to_new[op.node], op.op),
            .op2 => |op| old_to_new[i] = b.op2(old_to_new[op.lhs], old_to_new[op.rhs], op.op),
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
                    const local = unary_derivative(T, &b, op.op, child);
                    derivatives[op.node] = b.add(derivatives[op.node], b.mul(derivatives[i], local));
                },
                .op2 => |op| {
                    const lhs = old_to_new[op.lhs];
                    const rhs = old_to_new[op.rhs];
                    const local = binary_derivative(T, &b, op.op, lhs, rhs);
                    derivatives[op.lhs] = b.add(derivatives[op.lhs], b.mul(derivatives[i], local.lhs_part));
                    derivatives[op.rhs] = b.add(derivatives[op.rhs], b.mul(derivatives[i], local.rhs_part));
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
    const built = comptime build_grad_nodes(T, dag, grad_capacity(T, dag));
    return built.len;
}

test "grad_node_count" {
    const test_dag = [_]DAGNode(f64){
        .{ .scalar_parameter = 0 },
        .{ .op1 = .{ .node = 0, .op = .sin } },
        .{ .output = .{ .index = 0, .node = 1 } },
    };
    try std.testing.expectEqual(@as(usize, 8), grad_node_count(f64, &test_dag));
}

pub fn grad(comptime T: type, comptime dag: []const DAGNode(T)) struct {
    rows: usize,
    cols: usize,
    nodes: [grad_node_count(T, dag)]DAGNode(T),
} {
    @setEvalBranchQuota(100000);

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
