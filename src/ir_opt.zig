const std = @import("std");
const ir = @import("ir.zig");

fn constantIs(comptime T: type, comptime program: anytype, index: usize, expected: T) bool {
    const node = program.nodes[index];
    return switch (node.op) {
        .scalar_constant => |value| value == expected,
        .tensor_constant => |values| blk: {
            for (values) |value| if (value != expected) break :blk false;
            break :blk true;
        },
        .fill => |value| value == expected,
        else => false,
    };
}

fn rewriteNode(comptime T: type, node: ir.Node(T), map: []const usize) ir.Node(T) {
    var result = node;
    result.op = switch (node.op) {
        .unary => |unary| .{ .unary = .{ .input = map[unary.input], .op = unary.op } },
        .binary => |binary| .{ .binary = .{ .lhs = map[binary.lhs], .rhs = map[binary.rhs], .op = binary.op } },
        .scale => |scale| .{ .scale = .{ .value = map[scale.value], .scalar = map[scale.scalar] } },
        .reduce_dot => |dot| .{ .reduce_dot = .{ .lhs = map[dot.lhs], .rhs = map[dot.rhs] } },
        .mat_vec => |mat_vec| .{ .mat_vec = .{ .matrix = map[mat_vec.matrix], .vector = map[mat_vec.vector] } },
        .transpose_mat_vec => |mat_vec| .{ .transpose_mat_vec = .{ .matrix = map[mat_vec.matrix], .vector = map[mat_vec.vector] } },
        .outer => |outer| .{ .outer = .{ .lhs = map[outer.lhs], .rhs = map[outer.rhs] } },
        .extract => |extract| .{ .extract = .{ .input = map[extract.input], .index = extract.index } },
        else => node.op,
    };
    return result;
}

fn equivalent(comptime T: type, lhs: ir.Node(T), rhs: ir.Node(T)) bool {
    if (!lhs.shape.eql(rhs.shape) or lhs.kernel != rhs.kernel or std.meta.activeTag(lhs.op) != std.meta.activeTag(rhs.op)) return false;
    return switch (lhs.op) {
        .parameter => |value| value == rhs.op.parameter,
        .scalar_constant => |value| value == rhs.op.scalar_constant,
        .tensor_constant => |values| std.mem.eql(T, values, rhs.op.tensor_constant),
        .fill => |value| value == rhs.op.fill,
        .basis => |basis| basis.index == rhs.op.basis.index and basis.value == rhs.op.basis.value,
        .unary => |unary| unary.input == rhs.op.unary.input and unary.op == rhs.op.unary.op,
        .binary => |binary| binary.lhs == rhs.op.binary.lhs and binary.rhs == rhs.op.binary.rhs and binary.op == rhs.op.binary.op,
        .scale => |scale| scale.value == rhs.op.scale.value and scale.scalar == rhs.op.scale.scalar,
        .reduce_dot => |dot| dot.lhs == rhs.op.reduce_dot.lhs and dot.rhs == rhs.op.reduce_dot.rhs,
        .mat_vec => |mat_vec| mat_vec.matrix == rhs.op.mat_vec.matrix and mat_vec.vector == rhs.op.mat_vec.vector,
        .transpose_mat_vec => |mat_vec| mat_vec.matrix == rhs.op.transpose_mat_vec.matrix and mat_vec.vector == rhs.op.transpose_mat_vec.vector,
        .outer => |outer| outer.lhs == rhs.op.outer.lhs and outer.rhs == rhs.op.outer.rhs,
        .extract => |extract| extract.input == rhs.op.extract.input and extract.index == rhs.op.extract.index,
    };
}

fn findEquivalent(comptime T: type, comptime program: anytype, node: ir.Node(T)) ?usize {
    for (program.nodes[0..program.len], 0..) |existing, index| {
        if (equivalent(T, existing, node)) return index;
    }
    return null;
}

fn localRewrite(comptime T: type, writer: anytype, node: ir.Node(T)) ?usize {
    return switch (node.op) {
        .unary => |unary| if (unary.op == .neg) switch (writer.program.nodes[unary.input].op) {
            .unary => |inner| if (inner.op == .neg) inner.input else null,
            else => null,
        } else null,
        .binary => |binary| switch (binary.op) {
            .add => if (constantIs(T, writer.program, binary.lhs, 0))
                binary.rhs
            else if (constantIs(T, writer.program, binary.rhs, 0))
                binary.lhs
            else
                null,
            .sub => if (constantIs(T, writer.program, binary.rhs, 0)) binary.lhs else null,
            .mul => if (constantIs(T, writer.program, binary.lhs, 0))
                binary.lhs
            else if (constantIs(T, writer.program, binary.rhs, 0))
                binary.rhs
            else if (constantIs(T, writer.program, binary.lhs, 1))
                binary.rhs
            else if (constantIs(T, writer.program, binary.rhs, 1))
                binary.lhs
            else
                null,
            .div => if (constantIs(T, writer.program, binary.rhs, 1)) binary.lhs else null,
        },
        .scale => |scale| if (constantIs(T, writer.program, scale.scalar, 1)) scale.value else null,
        else => null,
    };
}

fn markDependencies(comptime T: type, program: anytype, active: []bool, node_index: usize) void {
    if (active[node_index]) return;
    active[node_index] = true;
    switch (program.nodes[node_index].op) {
        .unary => |unary| markDependencies(T, program, active, unary.input),
        .binary => |binary| {
            markDependencies(T, program, active, binary.lhs);
            markDependencies(T, program, active, binary.rhs);
        },
        .scale => |scale| {
            markDependencies(T, program, active, scale.value);
            markDependencies(T, program, active, scale.scalar);
        },
        .reduce_dot => |dot| {
            markDependencies(T, program, active, dot.lhs);
            markDependencies(T, program, active, dot.rhs);
        },
        .mat_vec => |mat_vec| {
            markDependencies(T, program, active, mat_vec.matrix);
            markDependencies(T, program, active, mat_vec.vector);
        },
        .transpose_mat_vec => |mat_vec| {
            markDependencies(T, program, active, mat_vec.matrix);
            markDependencies(T, program, active, mat_vec.vector);
        },
        .outer => |outer| {
            markDependencies(T, program, active, outer.lhs);
            markDependencies(T, program, active, outer.rhs);
        },
        .extract => |extract| markDependencies(T, program, active, extract.input),
        else => {},
    }
}

pub fn optimize(comptime T: type, comptime source: anytype) @TypeOf(source) {
    @setEvalBranchQuota(1_000_000);
    var writer = ir.Writer(T, @TypeOf(source).node_cap, @TypeOf(source).output_cap).init(
        source.input_size,
        source.tensor_backend,
        source.vector_bits,
    );
    writer.program.input_shapes = source.input_shapes;
    writer.program.result_shape = source.result_shape;
    var map: [@TypeOf(source).node_cap]usize = undefined;

    inline for (source.nodes[0..source.len], 0..) |source_node, source_index| {
        const node = rewriteNode(T, source_node, &map);
        if (localRewrite(T, &writer, node)) |alias| {
            map[source_index] = alias;
            continue;
        }
        if (findEquivalent(T, writer.program, node)) |existing| {
            map[source_index] = existing;
            continue;
        }
        map[source_index] = writer.append(node);
    }
    inline for (source.outputs[0..source.output_len]) |output| writer.output(map[output]);

    var active: [@TypeOf(source).node_cap]bool = @splat(false);
    inline for (writer.program.outputs[0..writer.program.output_len]) |output| markDependencies(T, writer.program, &active, output);

    var compact = ir.Writer(T, @TypeOf(source).node_cap, @TypeOf(source).output_cap).init(
        source.input_size,
        source.tensor_backend,
        source.vector_bits,
    );
    compact.program.input_shapes = source.input_shapes;
    compact.program.result_shape = source.result_shape;
    var compact_map: [@TypeOf(source).node_cap]usize = undefined;
    inline for (writer.program.nodes[0..writer.program.len], 0..) |node, index| {
        if (active[index]) compact_map[index] = compact.append(rewriteNode(T, node, &compact_map));
    }
    inline for (writer.program.outputs[0..writer.program.output_len]) |output| compact.output(compact_map[output]);
    ir.validate(T, compact.program);
    return compact.program;
}

test "IR optimization removes identities and dead constants" {
    const dag = @import("dag.zig");
    const vm = @import("vm.zig");
    const S = dag.Scalar(f32);
    const V = dag.Vector(f32, 4);
    const model = struct {
        fn call(x: *const V) V {
            const one = S.c(1);
            const zero = V.c(.{ 0, 0, 0, 0 });
            const scaled = x.mul(&one);
            return scaled.add(&zero);
        }
    }.call;
    const graph = comptime dag.toDag(f32, model);
    const source = comptime ir.lower(f32, graph, .{});
    const result = comptime optimize(f32, source);
    try std.testing.expect(result.len < source.len);

    var inputs = [_]f32{ 1, 2, 3, 4 };
    try std.testing.expectEqual(vm.evalFlat(f32, source, &inputs), vm.evalFlat(f32, result, &inputs));
}
