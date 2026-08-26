const std = @import("std");
const dag_mod = @import("../dag.zig");
const DAGNode = dag_mod.DAGNode;
const DAGWriter = @import("../dag_writer.zig").DAGWriter;

pub fn default_map(comptime size: usize) [size]usize {
    var result: [size]usize = undefined;
    for (0..size) |i| {
        result[i] = i;
    }
    return result;
}

test "default_map" {
    const map = default_map(10);
    try std.testing.expectEqualSlices(usize, &[10]usize{ 0, 1, 2, 3, 4, 5, 6, 7, 8, 9 }, &map);
}

pub fn rewire(comptime T: type, comptime dag: []const DAGNode(T), map: [dag.len]usize) [dag.len]DAGNode(T) {
    var result: [dag.len]DAGNode(T) = dag[0..dag.len].*;
    inline for (0..dag.len) |i| {
        result[i] = switch (result[i]) {
            .op1 => |op1| DAGNode(T){ .op1 = .{ .node = map[op1.node], .op = op1.op } },
            .op2 => |op2| dag_mod.op2_node(T, map[op2.lhs], map[op2.rhs], op2.op),
            .output => |output| DAGNode(T){ .output = .{ .index = output.index, .node = map[output.node] } },
            else => result[i],
        };
    }
    return result;
}

test "rewire" {
    const test_dag = comptime blk: {
        var b = DAGWriter(f64, 100){};
        const v1 = b.x();
        const v2 = b.neg(v1);
        const v3 = b.neg(v2);
        b.output(v3);
        break :blk b.dag();
    };
    const map = [_]usize{ 0, 1, 0, 3 };
    const result = rewire(f64, &test_dag, map);

    const expected_dag = comptime blk: {
        var b = DAGWriter(f64, 100){};
        const v1 = b.x();
        const v2 = b.neg(v1);
        _ = b.neg(v2);
        b.output(v1);
        break :blk b.dag();
    };
    try std.testing.expectEqualSlices(DAGNode(f64), &expected_dag, &result);
}

test "rewire normalizes commutative op2" {
    const test_dag = [_]DAGNode(f64){
        .{ .scalar_parameter = 0 },
        .{ .scalar_parameter = 1 },
        .{ .op2 = .{ .lhs = 0, .rhs = 1, .op = .add } },
    };
    const map = [_]usize{ 1, 0, 2 };
    const result = rewire(f64, &test_dag, map);
    try std.testing.expectEqual(
        DAGNode(f64){ .op2 = .{ .lhs = 0, .rhs = 1, .op = .add } },
        result[2],
    );
}
