const std = @import("std");
const Op1 = @import("../dag.zig").Op1;
const Op2 = @import("../dag.zig").Op2;
const DAGNode = @import("../dag.zig").DAGNode;
const output_size = @import("../dag.zig").output_size;
const Builder = @import("../dag_builder.zig").Builder;
const wyhash = std.hash.Wyhash;

const exprHash = u64;

fn param_hash(index: usize) exprHash {
    var hasher = wyhash.init(0);
    hasher.update(&std.mem.toBytes(@as(u16, 1212))); // little-endian tag, distinct from const_hash
    hasher.update(&std.mem.toBytes(index));
    return hasher.final();
}

fn const_hash(comptime T: type, val: T) exprHash {
    var hasher = wyhash.init(0);
    hasher.update(&std.mem.toBytes(@as(u16, 2121)));
    hasher.update(@typeName(T));
    hasher.update(&std.mem.toBytes(val));
    return hasher.final();
}

fn op1_hash(op: Op1, child_hash: exprHash) exprHash {
    var hasher = wyhash.init(0);
    hasher.update(&[1]u8{@intFromEnum(op)});
    hasher.update(&std.mem.toBytes(child_hash));
    return hasher.final();
}

test "op1_hash" {
    const op = Op1.abs;
    const child_hash = 123;
    const hash = op1_hash(op, child_hash);
    try std.testing.expectEqual(hash, 17858335559295060212);
}

fn op2_hash(op: Op2, lhs_hash: exprHash, rhs_hash: exprHash) exprHash {
    var hasher = wyhash.init(0);
    hasher.update(&[1]u8{@intFromEnum(op)});
    hasher.update(&std.mem.toBytes(lhs_hash));
    hasher.update(&std.mem.toBytes(rhs_hash));
    return hasher.final();
}

test "op2_hash" {
    const op = Op2.add;
    const lhs_hash = 123;
    const rhs_hash = 456;
    const hash = op2_hash(op, lhs_hash, rhs_hash);
    try std.testing.expectEqual(hash, 4011655355291119311);
}

fn dag_hash_array(comptime T: type, comptime dag: []const DAGNode(T)) [dag.len]exprHash {
    var hash_array: [dag.len]exprHash = undefined;
    inline for (dag, 0..) |node, i| {
        hash_array[i] = switch (node) {
            .scalar_constant => |c| const_hash(T, c),
            .scalar_parameter => |p| param_hash(p),
            .op1 => |op| op1_hash(op.op, hash_array[op.node]),
            .op2 => |op| op2_hash(op.op, hash_array[op.lhs], hash_array[op.rhs]),
            .output => |out| hash_array[out.node],
        };
    }
    return hash_array;
}

test "dag_hash_array" {
    const test_dag = comptime blk: {
        var b = Builder(f64, 100){};
        const v1 = b.x();
        const v2 = b.log(v1);
        const v3 = b.add(v1, v2);
        b.output(v3);
        break :blk b.dag();
    };
    const hash_array = dag_hash_array(f64, &test_dag);
    const expected_hashes = [_]exprHash{
        15825087117685866637,
        11463455980673280080,
        1660360994614574136,
        1660360994614574136,
    };
    for (hash_array, 0..) |hash, i| {
        try std.testing.expectEqual(hash, expected_hashes[i]);
    }
}

const CSEContext = struct {
    pub fn hash(ctx: @This(), key: exprHash) u64 {
        _ = ctx;
        return key;
    }
    pub fn eql(ctx: @This(), a: exprHash, b: exprHash) bool {
        _ = ctx;
        return a == b;
    }
};

fn hash_map_buffer_size(comptime entry_count: usize) usize {
    const cap = @max(
        @as(usize, 8),
        std.math.ceilPowerOfTwo(usize, entry_count * 2 + 1) catch unreachable,
    );
    return @as(usize, 24) + @as(usize, 9) * cap;
}

pub fn has_common_subexpression(comptime T: type, comptime dag: []const DAGNode(T)) bool {
    const active_nodes = @import("deadcode.zig").get_active_nodes(T, dag);
    const hash_array = dag_hash_array(T, dag);
    for (dag, 0..) |_, i| {
        if (!active_nodes[i]) continue;
        switch (dag[i]) {
            .output => continue,
            else => {},
        }
        for (0..i) |j| {
            if (!active_nodes[j]) continue;
            switch (dag[j]) {
                .output => continue,
                else => {},
            }
            if (hash_array[i] == hash_array[j]) {
                return true;
            }
        }
    }
    return false;
}

test "has_common_subexpression" {
    const test_dag = comptime blk: {
        var b = Builder(f64, 100){};
        const v1 = b.x();
        const v2 = b.log(v1);
        const v3 = b.add(v1, v2);
        _ = b.add(v1, v2);
        b.output(v3);
        break :blk b.dag();
    };
    try std.testing.expectEqual(has_common_subexpression(f64, &test_dag), false);
    const test_dag2 = comptime blk: {
        var b = Builder(f64, 100){};
        const v1 = b.x();
        const v2 = b.log(v1);
        const v3 = b.add(v1, v2);
        const v4 = b.add(v1, v2);
        const v5 = b.add(v4, v3);
        b.output(v5);
        break :blk b.dag();
    };
    try std.testing.expectEqual(has_common_subexpression(f64, &test_dag2), true);
    const test_dag3 = [_]DAGNode(f64){
        .{ .scalar_constant = 2 },
        .{ .scalar_constant = 1 },
        .{ .output = .{ .index = 0, .node = 1 } },
        .{ .output = .{ .index = 1, .node = 0 } },
        .{ .scalar_constant = 1 },
        .{ .scalar_constant = 2 },
        .{ .output = .{ .index = 2, .node = 5 } },
        .{ .output = .{ .index = 3, .node = 4 } },
    };
    try std.testing.expectEqual(has_common_subexpression(f64, &test_dag3), true);
}

pub fn cse(comptime T: type, comptime dag: []const DAGNode(T)) [dag.len]DAGNode(T) {
    const active_nodes = @import("deadcode.zig").get_active_nodes(T, dag);
    const hash_array = dag_hash_array(T, dag);
    var rewire_map = @import("rewire.zig").default_map(dag.len);
    for (dag, 0..) |_, i| {
        if (!active_nodes[i]) continue;
        switch (dag[i]) {
            .output => continue,
            else => {},
        }
        for (0..i) |j| {
            if (!active_nodes[j]) continue;
            switch (dag[j]) {
                .output => continue,
                else => {},
            }
            if (hash_array[i] == hash_array[j]) {
                rewire_map[i] = j;
                break;
            }
        }
    }
    return @import("rewire.zig").rewire(T, dag, rewire_map);
}

test "cse" {
    const test_dag = comptime blk: {
        var b = Builder(f64, 100){};
        const v1 = b.x();
        const v2 = b.log(v1);
        const v3 = b.add(v1, v2);
        const v4 = b.add(v1, v2);
        const v5 = b.add(v4, v3);
        b.output(v5);
        break :blk b.dag();
    };
    const expected_dag = comptime blk: {
        var b = Builder(f64, 100){};
        const v1 = b.x();
        const v2 = b.log(v1);
        const v3 = b.add(v1, v2);
        _ = b.add(v1, v2);
        const v5 = b.add(v3, v3);
        b.output(v5);
        break :blk b.dag();
    };
    const result = cse(f64, &test_dag);
    try std.testing.expectEqualSlices(DAGNode(f64), &expected_dag, &result);
}
