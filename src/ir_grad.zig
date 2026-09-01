const std = @import("std");
const ir = @import("ir.zig");

pub const Mode = enum {
    auto,
    forward,
    reverse,
};

pub const Selection = union(enum) {
    all,
    index: usize,
    range: struct {
        start: usize,
        len: usize,
    },
    indices: []const usize,
};

pub const Options = struct {
    wrt: Selection = .all,
    outputs: Selection = .all,
    mode: Mode = .auto,
    optimize: bool = true,
};

fn selectionLen(comptime selection: Selection, comptime total: usize, comptime label: []const u8) usize {
    return switch (selection) {
        .all => total,
        .index => |index| blk: {
            if (index >= total) @compileError(label ++ " index is out of bounds");
            break :blk 1;
        },
        .range => |range| blk: {
            if (range.start > total or range.len > total - range.start) @compileError(label ++ " range is out of bounds");
            break :blk range.len;
        },
        .indices => |indices| blk: {
            inline for (indices, 0..) |index, i| {
                if (index >= total) @compileError(label ++ " index is out of bounds");
                inline for (indices[0..i]) |previous| if (index == previous) @compileError(label ++ " indices must be unique");
            }
            break :blk indices.len;
        },
    };
}

fn selectionAt(comptime selection: Selection, comptime total: usize, comptime position: usize, comptime label: []const u8) usize {
    _ = selectionLen(selection, total, label);
    return switch (selection) {
        .all => position,
        .index => |index| index,
        .range => |range| range.start + position,
        .indices => |indices| indices[position],
    };
}

fn resolvedMode(comptime T: type, comptime source: anytype, comptime options: Options) Mode {
    const rows = selectionLen(options.outputs, ir.outputSize(T, source), "gradient output");
    const cols = selectionLen(options.wrt, source.input_size, "gradient input");
    return switch (options.mode) {
        .auto => if (cols < rows) .forward else .reverse,
        else => options.mode,
    };
}

const Component = struct {
    node: usize,
    index: usize,
};

fn outputComponent(comptime T: type, comptime source: anytype, flat_index: usize) Component {
    _ = T;
    var offset: usize = 0;
    for (source.outputs[0..source.output_len]) |node_index| {
        const len = source.nodes[node_index].shape.size();
        if (flat_index < offset + len) return .{ .node = node_index, .index = flat_index - offset };
        offset += len;
    }
    unreachable;
}

fn inputComponent(comptime T: type, comptime source: anytype, node_index: usize, flat_index: usize) ?usize {
    _ = T;
    return switch (source.nodes[node_index].op) {
        .parameter => |offset| if (flat_index >= offset and flat_index < offset + source.nodes[node_index].shape.size()) flat_index - offset else null,
        else => null,
    };
}

fn one(comptime T: type) T {
    return @as(T, 1);
}

fn zero(comptime T: type) T {
    return @as(T, 0);
}

fn localUnary(comptime T: type, writer: anytype, input: usize, current: usize, op: ir.UnaryOp) usize {
    const shape = writer.program.nodes[input].shape;
    const one_node = writer.fill(shape, one(T));
    return switch (op) {
        .neg => writer.fill(shape, @as(T, -1)),
        .abs => writer.binary(input, current, .div),
        .exp => current,
        .log => writer.binary(one_node, input, .div),
        .sqrt => blk: {
            const two = writer.fill(shape, @as(T, 2));
            break :blk writer.binary(one_node, writer.binary(two, current, .mul), .div);
        },
        .sin => writer.unary(input, .cos),
        .cos => writer.unary(writer.unary(input, .sin), .neg),
        .tan => writer.binary(one_node, writer.binary(current, current, .mul), .add),
    };
}

fn addAdjoint(writer: anytype, adjoints: []usize, target: usize, contribution: usize, none: usize) void {
    if (adjoints[target] == none) {
        adjoints[target] = contribution;
    } else {
        adjoints[target] = writer.binary(adjoints[target], contribution, .add);
    }
}

fn forwardNode(comptime T: type, writer: anytype, source: anytype, tangents: []usize, index: usize) usize {
    const node = source.nodes[index];
    return switch (node.op) {
        .parameter, .scalar_constant, .tensor_constant, .fill, .basis => unreachable,
        .unary => |unary| writer.binary(
            tangents[unary.input],
            localUnary(T, writer, unary.input, index, unary.op),
            .mul,
        ),
        .binary => |binary| switch (binary.op) {
            .add => writer.binary(tangents[binary.lhs], tangents[binary.rhs], .add),
            .sub => writer.binary(tangents[binary.lhs], tangents[binary.rhs], .sub),
            .mul => writer.binary(
                writer.binary(tangents[binary.lhs], binary.rhs, .mul),
                writer.binary(binary.lhs, tangents[binary.rhs], .mul),
                .add,
            ),
            .div => writer.binary(
                writer.binary(
                    writer.binary(tangents[binary.lhs], binary.rhs, .mul),
                    writer.binary(binary.lhs, tangents[binary.rhs], .mul),
                    .sub,
                ),
                writer.binary(binary.rhs, binary.rhs, .mul),
                .div,
            ),
        },
        .scale => |scale| writer.binary(
            writer.scale(tangents[scale.value], scale.scalar),
            writer.scale(scale.value, tangents[scale.scalar]),
            .add,
        ),
        .reduce_dot => |dot| writer.binary(
            writer.reduceDot(tangents[dot.lhs], dot.rhs),
            writer.reduceDot(dot.lhs, tangents[dot.rhs]),
            .add,
        ),
        .mat_vec => |mat_vec| writer.binary(
            writer.matVec(tangents[mat_vec.matrix], mat_vec.vector),
            writer.matVec(mat_vec.matrix, tangents[mat_vec.vector]),
            .add,
        ),
        .transpose_mat_vec => |mat_vec| writer.binary(
            writer.transposeMatVec(tangents[mat_vec.matrix], mat_vec.vector),
            writer.transposeMatVec(mat_vec.matrix, tangents[mat_vec.vector]),
            .add,
        ),
        .outer => |outer| writer.binary(
            writer.outer(tangents[outer.lhs], outer.rhs),
            writer.outer(outer.lhs, tangents[outer.rhs]),
            .add,
        ),
        .extract => |extract| writer.extract(tangents[extract.input], extract.index),
    };
}

fn reverseNode(comptime T: type, writer: anytype, source: anytype, adjoints: []usize, index: usize, none: usize) void {
    const adjoint = adjoints[index];
    if (adjoint == none) return;
    switch (source.nodes[index].op) {
        .parameter, .scalar_constant, .tensor_constant, .fill, .basis => {},
        .unary => |unary| addAdjoint(
            writer,
            adjoints,
            unary.input,
            writer.binary(adjoint, localUnary(T, writer, unary.input, index, unary.op), .mul),
            none,
        ),
        .binary => |binary| switch (binary.op) {
            .add => {
                addAdjoint(writer, adjoints, binary.lhs, adjoint, none);
                addAdjoint(writer, adjoints, binary.rhs, adjoint, none);
            },
            .sub => {
                addAdjoint(writer, adjoints, binary.lhs, adjoint, none);
                addAdjoint(writer, adjoints, binary.rhs, writer.unary(adjoint, .neg), none);
            },
            .mul => {
                addAdjoint(writer, adjoints, binary.lhs, writer.binary(adjoint, binary.rhs, .mul), none);
                addAdjoint(writer, adjoints, binary.rhs, writer.binary(adjoint, binary.lhs, .mul), none);
            },
            .div => {
                addAdjoint(writer, adjoints, binary.lhs, writer.binary(adjoint, binary.rhs, .div), none);
                const denominator = writer.binary(binary.rhs, binary.rhs, .mul);
                const numerator = writer.binary(adjoint, binary.lhs, .mul);
                addAdjoint(writer, adjoints, binary.rhs, writer.unary(writer.binary(numerator, denominator, .div), .neg), none);
            },
        },
        .scale => |scale| {
            addAdjoint(writer, adjoints, scale.value, writer.scale(adjoint, scale.scalar), none);
            addAdjoint(writer, adjoints, scale.scalar, writer.reduceDot(adjoint, scale.value), none);
        },
        .reduce_dot => |dot| {
            addAdjoint(writer, adjoints, dot.lhs, writer.scale(dot.rhs, adjoint), none);
            addAdjoint(writer, adjoints, dot.rhs, writer.scale(dot.lhs, adjoint), none);
        },
        .mat_vec => |mat_vec| {
            addAdjoint(writer, adjoints, mat_vec.matrix, writer.outer(adjoint, mat_vec.vector), none);
            addAdjoint(writer, adjoints, mat_vec.vector, writer.transposeMatVec(mat_vec.matrix, adjoint), none);
        },
        .transpose_mat_vec => |mat_vec| {
            addAdjoint(writer, adjoints, mat_vec.matrix, writer.outer(mat_vec.vector, adjoint), none);
            addAdjoint(writer, adjoints, mat_vec.vector, writer.matVec(mat_vec.matrix, adjoint), none);
        },
        .outer => |outer| {
            addAdjoint(writer, adjoints, outer.lhs, writer.matVec(adjoint, outer.rhs), none);
            addAdjoint(writer, adjoints, outer.rhs, writer.transposeMatVec(adjoint, outer.lhs), none);
        },
        .extract => |extract| {
            const basis = writer.basis(source.nodes[extract.input].shape, extract.index, one(T));
            addAdjoint(writer, adjoints, extract.input, writer.scale(basis, adjoint), none);
        },
    }
}

fn resultNodeCapacity(comptime T: type, comptime source: anytype, comptime options: Options) usize {
    const rows = selectionLen(options.outputs, ir.outputSize(T, source), "gradient output");
    const cols = selectionLen(options.wrt, source.input_size, "gradient input");
    const mode = resolvedMode(T, source, options);
    const passes = if (mode == .forward) cols else rows;
    const reverse_extraction = if (mode == .reverse) source.len * cols * 2 else 0;
    return @TypeOf(source).node_cap + 8 + passes * (source.len * 20 + rows + cols + reverse_extraction + 8);
}

fn resultOutputCapacity(comptime T: type, comptime source: anytype, comptime options: Options) usize {
    return selectionLen(options.outputs, ir.outputSize(T, source), "gradient output") *
        selectionLen(options.wrt, source.input_size, "gradient input");
}

fn jacobianShape(rows: usize, cols: usize) ir.Shape {
    if (rows == 0 or cols == 0) return .{ .matrix = .{ .rows = rows, .cols = cols } };
    if (rows == 1 and cols == 1) return .scalar;
    if (rows == 1) return .{ .vector = cols };
    if (cols == 1) return .{ .vector = rows };
    return .{ .matrix = .{ .rows = rows, .cols = cols } };
}

pub fn differentiate(
    comptime T: type,
    comptime source: anytype,
    comptime options: Options,
) ir.Program(T, resultNodeCapacity(T, source, options), resultOutputCapacity(T, source, options)) {
    @setEvalBranchQuota(10_000_000);
    ir.validate(T, source);
    const rows = selectionLen(options.outputs, ir.outputSize(T, source), "gradient output");
    const cols = selectionLen(options.wrt, source.input_size, "gradient input");
    const capacity = resultNodeCapacity(T, source, options);
    const output_capacity = resultOutputCapacity(T, source, options);
    var writer = ir.Writer(T, capacity, output_capacity).init(source.input_size, source.tensor_backend, source.vector_bits);
    writer.program.input_shapes = source.input_shapes;
    writer.program.result_shape = jacobianShape(rows, cols);
    inline for (source.nodes[0..source.len]) |node| _ = writer.append(node);

    var jacobian: [rows * cols]usize = undefined;
    switch (resolvedMode(T, source, options)) {
        .auto => unreachable,
        .forward => inline for (0..cols) |col| {
            const input_index = selectionAt(options.wrt, source.input_size, col, "gradient input");
            var tangents: [@TypeOf(source).node_cap]usize = undefined;
            inline for (source.nodes[0..source.len], 0..) |node, index| {
                tangents[index] = switch (node.op) {
                    .parameter => |offset| if (input_index >= offset and input_index < offset + node.shape.size())
                        writer.basis(node.shape, input_index - offset, one(T))
                    else
                        writer.fill(node.shape, zero(T)),
                    .scalar_constant, .tensor_constant, .fill, .basis => writer.fill(node.shape, zero(T)),
                    else => forwardNode(T, &writer, source, &tangents, index),
                };
            }
            inline for (0..rows) |row| {
                const output_index = selectionAt(options.outputs, ir.outputSize(T, source), row, "gradient output");
                const component = outputComponent(T, source, output_index);
                jacobian[row * cols + col] = if (source.nodes[component.node].shape == .scalar)
                    tangents[component.node]
                else
                    writer.extract(tangents[component.node], component.index);
            }
        },
        .reverse => inline for (0..rows) |row| {
            const none = std.math.maxInt(usize);
            var adjoints: [@TypeOf(source).node_cap]usize = @splat(none);
            const output_index = selectionAt(options.outputs, ir.outputSize(T, source), row, "gradient output");
            const component = outputComponent(T, source, output_index);
            adjoints[component.node] = writer.basis(source.nodes[component.node].shape, component.index, one(T));

            inline for (0..source.len) |offset| {
                const index = source.len - 1 - offset;
                reverseNode(T, &writer, source, &adjoints, index, none);
            }

            inline for (0..cols) |col| {
                const input_index = selectionAt(options.wrt, source.input_size, col, "gradient input");
                var derivative: ?usize = null;
                inline for (source.nodes[0..source.len], 0..) |_, index| {
                    if (inputComponent(T, source, index, input_index)) |component_index| {
                        if (adjoints[index] != none) {
                            const contribution = if (source.nodes[index].shape == .scalar)
                                adjoints[index]
                            else
                                writer.extract(adjoints[index], component_index);
                            derivative = if (derivative) |existing| writer.binary(existing, contribution, .add) else contribution;
                        }
                    }
                }
                jacobian[row * cols + col] = derivative orelse writer.fill(.scalar, zero(T));
            }
        },
    }
    inline for (jacobian) |node| writer.output(node);
    ir.validate(T, writer.program);
    return writer.program;
}

test "IR reverse gradient preserves matrix and vector operations" {
    const dag = @import("dag.zig");
    const vm = @import("vm.zig");
    const opt = @import("ir_opt.zig");
    const V = dag.Vector(f32, 4);
    const M = dag.Matrix(f32, 2, 4);
    const Y = dag.Vector(f32, 2);
    const model = struct {
        fn call(matrix: *const M, x: *const V, y: *const Y) dag.Scalar(f32) {
            const product = matrix.matMul(x);
            return product.dot(y);
        }
    }.call;
    const graph = comptime dag.toDag(f32, model);
    const primal = comptime ir.lower(f32, graph, .{});
    const raw_gradient = comptime differentiate(f32, primal, .{ .wrt = .{ .range = .{ .start = 8, .len = 4 } } });
    const gradient = comptime opt.optimize(f32, raw_gradient);

    var inputs = [_]f32{
        1, 2, 3, 4,
        5, 6, 7, 8,
        1, 2, 3, 4,
        2, 3,
    };
    const actual = vm.evalFlat(f32, gradient, &inputs);
    try std.testing.expectEqual([_]f32{ 17, 22, 27, 32 }, actual);
}

test "IR forward mode computes a vector-output Jacobian" {
    const dag = @import("dag.zig");
    const vm = @import("vm.zig");
    const S = dag.Scalar(f64);
    const V = dag.Vector(f64, 3);
    const model = struct {
        fn call(x: *const S) V {
            const coefficients = V.c(.{ 1, 2, 3 });
            return coefficients.mul(x);
        }
    }.call;
    const graph = comptime dag.toDag(f64, model);
    const primal = comptime ir.lower(f64, graph, .{});
    const gradient = comptime differentiate(f64, primal, .{});
    var inputs = [_]f64{4};
    try std.testing.expectEqual([_]f64{ 1, 2, 3 }, vm.evalFlat(f64, gradient, &inputs));
}

test "generated gradient IR can be differentiated for a Hessian" {
    const dag = @import("dag.zig");
    const vm = @import("vm.zig");
    const S = dag.Scalar(f64);
    const model = struct {
        fn call(x: *const S) S {
            const square = x.mul(x);
            return square.mul(x);
        }
    }.call;
    const graph = comptime dag.toDag(f64, model);
    const primal = comptime ir.lower(f64, graph, .{});
    const gradient = comptime differentiate(f64, primal, .{});
    const hessian = comptime differentiate(f64, gradient, .{});
    var inputs = [_]f64{2};
    try std.testing.expectEqual(@as(f64, 12), vm.evalFlat(f64, hessian, &inputs)[0]);
}

test "reverse differentiation handles overlapping raw parameter ranges" {
    const vm = @import("vm.zig");
    const shape = ir.Shape{ .vector = 4 };
    const source = comptime blk: {
        var writer = ir.Writer(f32, 3, 1).init(4, .simd, 1024);
        const lhs = writer.append(.{ .shape = shape, .kernel = .simd, .op = .{ .parameter = 0 } });
        const rhs = writer.append(.{ .shape = shape, .kernel = .simd, .op = .{ .parameter = 0 } });
        writer.output(writer.binary(lhs, rhs, .add));
        writer.program.input_shapes = &.{shape};
        writer.program.result_shape = shape;
        break :blk writer.program;
    };
    const gradient = comptime differentiate(f32, source, .{ .mode = .reverse });
    const actual = vm.eval(gradient, .{[4]f32{ 1, 2, 3, 4 }});
    const expected = [4][4]f32{
        .{ 2, 0, 0, 0 },
        .{ 0, 2, 0, 0 },
        .{ 0, 0, 2, 0 },
        .{ 0, 0, 0, 2 },
    };
    try std.testing.expectEqual(expected, actual);
}

test "empty selections preserve Jacobian dimensions" {
    const dag = @import("dag.zig");
    const vm = @import("vm.zig");
    const V = dag.Vector(f32, 2);
    const model = struct {
        fn call(x: *const V) V {
            return x.neg();
        }
    }.call;
    const graph = comptime dag.toDag(f32, model);
    const source = comptime ir.lower(f32, graph, .{});
    const gradient = comptime differentiate(f32, source, .{
        .wrt = .{ .range = .{ .start = 0, .len = 0 } },
    });
    try std.testing.expect(gradient.result_shape.eql(.{ .matrix = .{ .rows = 2, .cols = 0 } }));
    try std.testing.expectEqual([2][0]f32{ .{}, .{} }, vm.eval(gradient, .{[2]f32{ 1, 2 }}));
}
