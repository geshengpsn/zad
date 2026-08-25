const std = @import("std");
const Builder = @import("../dag_builder.zig").Builder;
const dag_mod = @import("../dag.zig");
const DAGNode = dag_mod.DAGNode;

pub fn get_active_nodes(comptime T: type, comptime dag: []const DAGNode(T)) [dag.len]bool {
    var active_nodes_: [dag.len]bool = .{false} ** dag.len;
    inline for (0..dag.len) |i| {
        const reverse_index = dag.len - i - 1;
        const node = dag[reverse_index];
        switch (node) {
            .scalar_parameter => {
                active_nodes_[reverse_index] = true;
            },
            .output => |o| {
                active_nodes_[reverse_index] = true;
                active_nodes_[o.node] = true;
            },
            .op1 => |op| {
                if (active_nodes_[reverse_index]) {
                    active_nodes_[op.node] = true;
                }
            },
            .op2 => |op| {
                if (active_nodes_[reverse_index]) {
                    active_nodes_[op.lhs] = true;
                    active_nodes_[op.rhs] = true;
                }
            },
            else => {},
        }
    }
    return active_nodes_;
}

fn number_of_true(array: []const bool) usize {
    var count: usize = 0;
    for (array) |e| {
        if (e) {
            count += 1;
        }
    }
    return count;
}

fn get_old_to_new_index_map(comptime array: []const bool) [array.len]usize {
    var new_old_index_map: [array.len]usize = undefined;
    var map_count: usize = 0;
    for (0..array.len) |index| {
        if (array[index]) {
            new_old_index_map[index] = map_count;
            map_count += 1;
        }
    }
    return new_old_index_map;
}

fn remove_elements_by_boolean_array(comptime T: type, comptime dag: []const DAGNode(T), comptime active_nodes: []const bool) [number_of_true(active_nodes)]DAGNode(T) {
    if (active_nodes.len != dag.len) {
        @compileError("active_nodes and dag must have the same length");
    }
    const map: [dag.len]usize = get_old_to_new_index_map(active_nodes);
    const new_dag_len = comptime number_of_true(active_nodes);
    var new_dag: [new_dag_len]DAGNode(T) = undefined;
    for (0..dag.len) |i| {
        if (active_nodes[i]) {
            const node = dag[i];
            new_dag[map[i]] = switch (node) {
                .op1 => |op| DAGNode(T){ .op1 = .{ .node = map[op.node], .op = op.op } },
                .op2 => |op| dag_mod.op2_node(T, map[op.lhs], map[op.rhs], op.op),
                .output => |o| DAGNode(T){ .output = .{ .node = map[o.node], .index = o.index } },
                else => node,
            };
        }
    }
    return new_dag;
}

pub fn deadcode_elimination(comptime T: type, comptime dag: []const DAGNode(T)) [number_of_true(&get_active_nodes(T, dag))]DAGNode(T) {
    const active_nodes = comptime get_active_nodes(T, dag);
    const new_dag = remove_elements_by_boolean_array(T, dag, &active_nodes);
    return new_dag;
}

pub fn has_deadcode(comptime T: type, comptime dag: []const DAGNode(T)) bool {
    const active_nodes = comptime get_active_nodes(T, dag);
    return number_of_true(&active_nodes) != dag.len;
}

const test_dag = blk: {
    var b = Builder(f64, 9){};
    const x1 = b.x();
    const x2 = b.x();
    const v1 = b.log(x1);
    const v2 = b.mul(x1, x2);
    const v3 = b.sin(x2);
    _ = b.cos(x1); // dead code
    const v4 = b.add(v1, v2);
    const v5 = b.sub(v4, v3);
    b.output(v5);
    break :blk b.dag();
};

test "active_nodes" {
    const computed_nodes = get_active_nodes(f64, &test_dag);
    const expect_nodes = [_]bool{ true, true, true, true, true, false, true, true, true };
    for (0..expect_nodes.len) |i| {
        try std.testing.expectEqual(expect_nodes[i], computed_nodes[i]);
    }
}

test "number_of_true" {
    try std.testing.expectEqual(number_of_true(&get_active_nodes(f64, &test_dag)), 8);
}

test "old_to_new_index_map" {
    const old_to_new_index_map = get_old_to_new_index_map(&get_active_nodes(f64, &test_dag));
    const expect_map = [_]usize{ 0, 1, 2, 3, 4, 12297829382473034410, 5, 6, 7 };
    for (0..old_to_new_index_map.len) |i| {
        try std.testing.expectEqual(expect_map[i], old_to_new_index_map[i]);
    }
}

test "remove_elements" {
    const active_nodes = comptime get_active_nodes(f64, &test_dag);
    const new_dag = remove_elements_by_boolean_array(f64, &test_dag, &active_nodes);
    const expect_dag = comptime blk: {
        var b = Builder(f64, 9){};
        const x1 = b.x();
        const x2 = b.x();
        const v1 = b.log(x1);
        const v2 = b.mul(x1, x2);
        const v3 = b.sin(x2);
        const v4 = b.add(v1, v2);
        const v5 = b.sub(v4, v3);
        b.output(v5);
        break :blk b.dag();
    };
    for (0..expect_dag.len) |i| {
        try std.testing.expectEqual(expect_dag[i], new_dag[i]);
    }
}

test "deadcode_elimination" {
    const new_dag = deadcode_elimination(f64, &test_dag);
    const expect_dag = comptime blk: {
        var b = Builder(f64, 9){};
        const x1 = b.x();
        const x2 = b.x();
        const v1 = b.log(x1);
        const v2 = b.mul(x1, x2);
        const v3 = b.sin(x2);
        const v4 = b.add(v1, v2);
        const v5 = b.sub(v4, v3);
        b.output(v5);
        break :blk b.dag();
    };
    for (0..expect_dag.len) |i| {
        try std.testing.expectEqual(expect_dag[i], new_dag[i]);
    }
}

test "deadcode normalizes commutative op2 after reindex" {
    const reindex_dag = [_]DAGNode(f64){
        .{ .scalar_constant = 0.0 },
        .{ .scalar_parameter = 0 },
        .{ .scalar_parameter = 1 },
        .{ .op2 = .{ .lhs = 2, .rhs = 1, .op = .mul } },
        .{ .output = .{ .index = 0, .node = 3 } },
    };
    const result = deadcode_elimination(f64, &reindex_dag);
    try std.testing.expectEqual(
        DAGNode(f64){ .op2 = .{ .lhs = 0, .rhs = 1, .op = .mul } },
        result[2],
    );
}

test "has_deadcode" {
    try std.testing.expectEqual(has_deadcode(f64, &test_dag), true);
    const expect_dag = comptime blk: {
        var b = Builder(f64, 9){};
        const x1 = b.x();
        const x2 = b.x();
        const v1 = b.log(x1);
        const v2 = b.mul(x1, x2);
        const v3 = b.sin(x2);
        const v4 = b.add(v1, v2);
        const v5 = b.sub(v4, v3);
        b.output(v5);
        break :blk b.dag();
    };
    try std.testing.expectEqual(has_deadcode(f64, &expect_dag), false);
}
