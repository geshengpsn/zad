const std = @import("std");
const Builder = @import("../dag_builder.zig").Builder;
const DAGNode = @import("../dag.zig").DAGNode;

fn is_constant(comptime T: type, node: DAGNode(T)) bool {
    return switch (node) {
        .scalar_constant => true,
        else => false,
    };
}

fn get_constant(comptime T: type, node: DAGNode(T)) T {
    return switch (node) {
        .scalar_constant => |val| val,
        else => unreachable,
    };
}

pub fn has_unfold_constant(comptime T: type, dag: []const DAGNode(T)) bool {
    for (dag) |node| {
        switch (node) {
            .op1 => |op1| {
                if (is_constant(T, dag[op1.node])) {
                    return true;
                }
            },
            .op2 => |op2| {
                if (is_constant(T, dag[op2.lhs]) and is_constant(T, dag[op2.rhs])) {
                    return true;
                }
            },
            else => continue,
        }
    }
    return false;
}

// constant_fold always returns dag with deadcode
pub fn constant_fold(comptime T: type, comptime dag: []const DAGNode(T)) [dag.len]DAGNode(T) {
    var result: [dag.len]DAGNode(T) = undefined;
    for (dag, 0..) |node, i| {
        result[i] = node;
    }
    inline for (dag, 0..) |node, i| {
        switch (node) {
            .op1 => |op1| {
                if (is_constant(T, dag[op1.node])) {
                    const val = get_constant(T, dag[op1.node]);
                    const computed_val = switch (op1.op) {
                        .neg => -val,
                        .abs => @abs(val),
                        .exp => @exp(val),
                        .log => @log(val),
                        .sqrt => @sqrt(val),
                        .sin => @sin(val),
                        .cos => @cos(val),
                        .tan => @tan(val),
                    };
                    result[i] = DAGNode(T){
                        .scalar_constant = computed_val,
                    };
                }
            },
            .op2 => |op2| {
                if (is_constant(T, dag[op2.lhs]) and is_constant(T, dag[op2.rhs])) {
                    const lhs_val = get_constant(T, dag[op2.lhs]);
                    const rhs_val = get_constant(T, dag[op2.rhs]);
                    const computed_val = switch (op2.op) {
                        .add => lhs_val + rhs_val,
                        .sub => lhs_val - rhs_val,
                        .mul => lhs_val * rhs_val,
                        .div => lhs_val / rhs_val,
                    };
                    result[i] = DAGNode(T){
                        .scalar_constant = computed_val,
                    };
                }
            },
            else => continue,
        }
    }
    return result;
}

test "is_constant" {
    const node = DAGNode(f32){
        .scalar_constant = 1.0,
    };
    try std.testing.expect(is_constant(f32, node));
}

test "has_unfold_constant" {
    const test_dag_a = comptime blk: {
        var b = Builder(f32, 100){};
        const v1 = b.c(1.0);
        const v2 = b.c(2.0);
        _ = b.add(v1, v2);
        break :blk b.dag();
    };
    try std.testing.expect(has_unfold_constant(f32, &test_dag_a));
    const test_dag_b = comptime blk: {
        var b = Builder(f32, 100){};
        const v1 = b.x();
        const v2 = b.x();
        _ = b.add(v1, v2);
        break :blk b.dag();
    };
    try std.testing.expect(!has_unfold_constant(f32, &test_dag_b));
}

test "constant_fold" {
    const test_dag_a = comptime blk: {
        var b = Builder(f32, 100){};
        const v1 = b.c(std.math.pi / 2.0);
        const v2 = b.sin(v1);
        const v3 = b.c(1.0);
        const v4 = b.add(v2, v3);
        const v5 = b.x();
        const v6 = b.mul(v5, v4);
        b.output(v6);
        break :blk b.dag();
    };
    const test_dag_b = comptime blk: {
        var b = Builder(f32, 100){};
        _ = b.c(std.math.pi / 2.0);
        const v2 = b.c(@sin(std.math.pi / 2.0));
        const v3 = b.c(1.0);
        const v4 = b.add(v2, v3);
        const v5 = b.x();
        const v6 = b.mul(v5, v4);
        b.output(v6);
        break :blk b.dag();
    };
    const folded_dag_a = constant_fold(f32, &test_dag_a);
    for (folded_dag_a, test_dag_b) |node_a, node_b| {
        try std.testing.expectEqual(node_a, node_b);
    }
}
