const std = @import("std");
const ir = @import("ir.zig");

fn unaryScalar(comptime T: type, op: ir.UnaryOp, value: T) T {
    return switch (op) {
        .neg => -value,
        .abs => @abs(value),
        .exp => @exp(value),
        .log => @log(value),
        .sqrt => @sqrt(value),
        .sin => @sin(value),
        .cos => @cos(value),
        .tan => @tan(value),
    };
}

fn unaryVector(comptime T: type, comptime lanes: usize, op: ir.UnaryOp, value: @Vector(lanes, T)) @Vector(lanes, T) {
    return switch (op) {
        .neg => -value,
        .abs => @abs(value),
        .exp => @exp(value),
        .log => @log(value),
        .sqrt => @sqrt(value),
        .sin => @sin(value),
        .cos => @cos(value),
        .tan => @tan(value),
    };
}

fn binaryScalar(comptime T: type, op: ir.BinaryOp, lhs: T, rhs: T) T {
    return switch (op) {
        .add => lhs + rhs,
        .sub => lhs - rhs,
        .mul => lhs * rhs,
        .div => lhs / rhs,
    };
}

fn binaryVector(comptime T: type, comptime lanes: usize, op: ir.BinaryOp, lhs: @Vector(lanes, T), rhs: @Vector(lanes, T)) @Vector(lanes, T) {
    return switch (op) {
        .add => lhs + rhs,
        .sub => lhs - rhs,
        .mul => lhs * rhs,
        .div => lhs / rhs,
    };
}

fn validLanes(comptime T: type, comptime program: anytype, comptime node_index: usize, comptime local_slot: usize) comptime_int {
    const lanes = ir.stackLanes(T, program.tensor_backend, program.vector_bits);
    const shape = program.nodes[node_index].shape;
    const columns = switch (shape) {
        .vector => |len| len,
        .matrix => |matrix| matrix.cols,
        .scalar => unreachable,
    };
    const chunk = switch (shape) {
        .vector => local_slot,
        .matrix => local_slot % ir.chunksPerRow(T, shape, program.vector_bits),
        .scalar => unreachable,
    };
    return @min(lanes, columns - chunk * lanes);
}

fn clearPadding(
    comptime T: type,
    comptime program: anytype,
    comptime node_index: usize,
    comptime local_slot: usize,
    value: @Vector(ir.stackLanes(T, program.tensor_backend, program.vector_bits), T),
) @Vector(ir.stackLanes(T, program.tensor_backend, program.vector_bits), T) {
    const lanes = ir.stackLanes(T, program.tensor_backend, program.vector_bits);
    const valid = comptime validLanes(T, program, node_index, local_slot);
    if (valid == lanes) return value;
    var mask: @Vector(lanes, bool) = @splat(false);
    inline for (0..valid) |lane| mask[lane] = true;
    return @select(T, mask, value, @as(@Vector(lanes, T), @splat(0)));
}

fn pack(
    comptime T: type,
    comptime lanes: usize,
    values: []const T,
    offset: usize,
    comptime valid: usize,
) @Vector(lanes, T) {
    var result: @Vector(lanes, T) = @splat(0);
    inline for (0..valid) |lane| result[lane] = values[offset + lane];
    return result;
}

fn unpack(
    comptime T: type,
    comptime lanes: usize,
    value: @Vector(lanes, T),
    outputs: []T,
    offset: usize,
    comptime valid: usize,
) void {
    inline for (0..valid) |lane| outputs[offset + lane] = value[lane];
}

fn vectorSlotForElement(comptime T: type, comptime program: anytype, comptime node_index: usize, comptime index: usize) struct { slot: usize, lane: usize } {
    const lanes = comptime ir.stackLanes(T, program.tensor_backend, program.vector_bits);
    const shape = program.nodes[node_index].shape;
    return switch (shape) {
        .vector => .{
            .slot = ir.vectorOffset(program, node_index) + index / lanes,
            .lane = index % lanes,
        },
        .matrix => |matrix| blk: {
            const row = index / matrix.cols;
            const col = index % matrix.cols;
            break :blk .{
                .slot = ir.vectorOffset(program, node_index) + row * ir.chunksPerRow(T, shape, program.vector_bits) + col / lanes,
                .lane = col % lanes,
            };
        },
        .scalar => unreachable,
    };
}

fn tensorElement(
    comptime T: type,
    comptime program: anytype,
    comptime node_index: usize,
    comptime index: usize,
    scalar_stack: []const T,
    vector_stack: []const @Vector(ir.stackLanes(T, program.tensor_backend, program.vector_bits), T),
) T {
    if (comptime ir.usesVectorStack(program.nodes[node_index])) {
        const location = comptime vectorSlotForElement(T, program, node_index, index);
        return vector_stack[location.slot][comptime location.lane];
    }
    return scalar_stack[ir.scalarOffset(program, node_index) + index];
}

fn setTensorElement(
    comptime T: type,
    comptime program: anytype,
    comptime node_index: usize,
    comptime index: usize,
    value: T,
    scalar_stack: []T,
    vector_stack: []@Vector(ir.stackLanes(T, program.tensor_backend, program.vector_bits), T),
) void {
    if (comptime ir.usesVectorStack(program.nodes[node_index])) {
        const location = comptime vectorSlotForElement(T, program, node_index, index);
        vector_stack[location.slot][comptime location.lane] = value;
    } else {
        scalar_stack[ir.scalarOffset(program, node_index) + index] = value;
    }
}

fn loadBoundaryTensor(
    comptime T: type,
    comptime program: anytype,
    comptime node_index: usize,
    values: []const T,
    value_offset: usize,
    scalar_stack: []T,
    vector_stack: []@Vector(ir.stackLanes(T, program.tensor_backend, program.vector_bits), T),
) void {
    const node = program.nodes[node_index];
    if (comptime ir.usesVectorStack(node)) {
        const base = ir.vectorOffset(program, node_index);
        const slots = comptime ir.vectorSlots(T, node.shape, program.vector_bits);
        const lanes = comptime ir.stackLanes(T, program.tensor_backend, program.vector_bits);
        const chunks = comptime ir.chunksPerRow(T, node.shape, program.vector_bits);
        const columns = switch (node.shape) {
            .vector => |len| len,
            .matrix => |matrix| matrix.cols,
            .scalar => unreachable,
        };
        inline for (0..slots) |local_slot| {
            const row = switch (node.shape) {
                .vector => 0,
                .matrix => local_slot / chunks,
                .scalar => unreachable,
            };
            const chunk = local_slot % chunks;
            const valid = comptime @min(lanes, columns - chunk * lanes);
            vector_stack[base + local_slot] = pack(T, lanes, values, value_offset + row * columns + chunk * lanes, valid);
        }
    } else {
        const len = node.shape.size();
        @memcpy(scalar_stack[ir.scalarOffset(program, node_index)..][0..len], values[value_offset..][0..len]);
    }
}

fn storeBoundaryTensor(
    comptime T: type,
    comptime program: anytype,
    comptime node_index: usize,
    scalar_stack: []const T,
    vector_stack: []const @Vector(ir.stackLanes(T, program.tensor_backend, program.vector_bits), T),
    outputs: []T,
    output_offset: usize,
) void {
    const node = program.nodes[node_index];
    if (comptime ir.usesVectorStack(node)) {
        const base = ir.vectorOffset(program, node_index);
        const slots = comptime ir.vectorSlots(T, node.shape, program.vector_bits);
        const lanes = comptime ir.stackLanes(T, program.tensor_backend, program.vector_bits);
        const chunks = comptime ir.chunksPerRow(T, node.shape, program.vector_bits);
        const columns = switch (node.shape) {
            .vector => |len| len,
            .matrix => |matrix| matrix.cols,
            .scalar => unreachable,
        };
        inline for (0..slots) |local_slot| {
            const row = switch (node.shape) {
                .vector => 0,
                .matrix => local_slot / chunks,
                .scalar => unreachable,
            };
            const chunk = local_slot % chunks;
            const valid = comptime @min(lanes, columns - chunk * lanes);
            unpack(T, lanes, vector_stack[base + local_slot], outputs, output_offset + row * columns + chunk * lanes, valid);
        }
    } else {
        const len = node.shape.size();
        @memcpy(outputs[output_offset..][0..len], scalar_stack[ir.scalarOffset(program, node_index)..][0..len]);
    }
}

fn executeNode(
    comptime T: type,
    comptime program: anytype,
    comptime node_index: usize,
    inputs: []const T,
    scalar_stack: []T,
    vector_stack: []@Vector(ir.stackLanes(T, program.tensor_backend, program.vector_bits), T),
) void {
    const node = comptime program.nodes[node_index];
    const lanes = comptime ir.stackLanes(T, program.tensor_backend, program.vector_bits);
    const scalar_dst = if (comptime !ir.usesVectorStack(node)) ir.scalarOffset(program, node_index) else 0;
    const vector_dst = if (comptime ir.usesVectorStack(node)) ir.vectorOffset(program, node_index) else 0;
    const slots = comptime if (ir.usesVectorStack(node)) ir.vectorSlots(T, node.shape, program.vector_bits) else 0;

    switch (node.op) {
        .parameter => |offset| loadBoundaryTensor(T, program, node_index, inputs, offset, scalar_stack, vector_stack),
        .scalar_constant => |value| scalar_stack[scalar_dst] = value,
        .tensor_constant => |values| loadBoundaryTensor(T, program, node_index, values, 0, scalar_stack, vector_stack),
        .fill => |value| if (comptime ir.usesVectorStack(node)) {
            inline for (0..slots) |slot| vector_stack[vector_dst + slot] = clearPadding(T, program, node_index, slot, @splat(value));
        } else {
            @memset(scalar_stack[scalar_dst..][0..node.shape.size()], value);
        },
        .basis => |basis| {
            if (comptime ir.usesVectorStack(node)) {
                inline for (0..slots) |slot| vector_stack[vector_dst + slot] = @splat(0);
            } else {
                @memset(scalar_stack[scalar_dst..][0..node.shape.size()], 0);
            }
            setTensorElement(T, program, node_index, basis.index, basis.value, scalar_stack, vector_stack);
        },
        .unary => |unary| if (comptime ir.usesVectorStack(node)) {
            const src = ir.vectorOffset(program, unary.input);
            inline for (0..slots) |slot| {
                vector_stack[vector_dst + slot] = clearPadding(T, program, node_index, slot, unaryVector(T, lanes, unary.op, vector_stack[src + slot]));
            }
        } else {
            const src = ir.scalarOffset(program, unary.input);
            for (0..node.shape.size()) |index| scalar_stack[scalar_dst + index] = unaryScalar(T, unary.op, scalar_stack[src + index]);
        },
        .binary => |binary| if (comptime ir.usesVectorStack(node)) {
            const lhs = ir.vectorOffset(program, binary.lhs);
            const rhs = ir.vectorOffset(program, binary.rhs);
            inline for (0..slots) |slot| {
                vector_stack[vector_dst + slot] = clearPadding(T, program, node_index, slot, binaryVector(T, lanes, binary.op, vector_stack[lhs + slot], vector_stack[rhs + slot]));
            }
        } else {
            const lhs = ir.scalarOffset(program, binary.lhs);
            const rhs = ir.scalarOffset(program, binary.rhs);
            for (0..node.shape.size()) |index| scalar_stack[scalar_dst + index] = binaryScalar(T, binary.op, scalar_stack[lhs + index], scalar_stack[rhs + index]);
        },
        .scale => |scale| {
            const scalar = scalar_stack[ir.scalarOffset(program, scale.scalar)];
            if (comptime ir.usesVectorStack(node)) {
                const src = ir.vectorOffset(program, scale.value);
                const factor: @Vector(lanes, T) = @splat(scalar);
                inline for (0..slots) |slot| vector_stack[vector_dst + slot] = clearPadding(T, program, node_index, slot, vector_stack[src + slot] * factor);
            } else {
                const src = ir.scalarOffset(program, scale.value);
                for (0..node.shape.size()) |index| scalar_stack[scalar_dst + index] = scalar_stack[src + index] * scalar;
            }
        },
        .reduce_dot => |dot| {
            var result: T = 0;
            if (comptime ir.usesVectorStack(program.nodes[dot.lhs])) {
                const lhs = ir.vectorOffset(program, dot.lhs);
                const rhs = ir.vectorOffset(program, dot.rhs);
                const operand_slots = comptime ir.vectorSlots(T, program.nodes[dot.lhs].shape, program.vector_bits);
                var sum: @Vector(lanes, T) = @splat(0);
                inline for (0..operand_slots) |slot| sum += vector_stack[lhs + slot] * vector_stack[rhs + slot];
                result = @reduce(.Add, sum);
            } else {
                const lhs = ir.scalarOffset(program, dot.lhs);
                const rhs = ir.scalarOffset(program, dot.rhs);
                for (0..program.nodes[dot.lhs].shape.size()) |index| result += scalar_stack[lhs + index] * scalar_stack[rhs + index];
            }
            scalar_stack[scalar_dst] = result;
        },
        .mat_vec => |mat_vec| {
            const matrix_shape = program.nodes[mat_vec.matrix].shape.matrix;
            if (comptime ir.usesVectorStack(node)) {
                inline for (0..slots) |slot| vector_stack[vector_dst + slot] = @splat(0);
                const matrix = ir.vectorOffset(program, mat_vec.matrix);
                const vector = ir.vectorOffset(program, mat_vec.vector);
                const row_chunks = comptime ir.chunksPerRow(T, program.nodes[mat_vec.matrix].shape, program.vector_bits);
                inline for (0..matrix_shape.rows) |row| {
                    var sum: @Vector(lanes, T) = @splat(0);
                    inline for (0..row_chunks) |chunk| sum += vector_stack[matrix + row * row_chunks + chunk] * vector_stack[vector + chunk];
                    setTensorElement(T, program, node_index, row, @reduce(.Add, sum), scalar_stack, vector_stack);
                }
            } else {
                const matrix = ir.scalarOffset(program, mat_vec.matrix);
                const vector = ir.scalarOffset(program, mat_vec.vector);
                for (0..matrix_shape.rows) |row| {
                    var sum: T = 0;
                    for (0..matrix_shape.cols) |col| sum += scalar_stack[matrix + row * matrix_shape.cols + col] * scalar_stack[vector + col];
                    scalar_stack[scalar_dst + row] = sum;
                }
            }
        },
        .transpose_mat_vec => |mat_vec| {
            const matrix_shape = program.nodes[mat_vec.matrix].shape.matrix;
            if (comptime ir.usesVectorStack(node)) {
                inline for (0..slots) |slot| vector_stack[vector_dst + slot] = @splat(0);
                const matrix = ir.vectorOffset(program, mat_vec.matrix);
                const row_chunks = comptime ir.chunksPerRow(T, program.nodes[mat_vec.matrix].shape, program.vector_bits);
                inline for (0..matrix_shape.rows) |row| {
                    const scalar: @Vector(lanes, T) = @splat(tensorElement(T, program, mat_vec.vector, row, scalar_stack, vector_stack));
                    inline for (0..row_chunks) |chunk| vector_stack[vector_dst + chunk] += vector_stack[matrix + row * row_chunks + chunk] * scalar;
                }
                inline for (0..slots) |slot| vector_stack[vector_dst + slot] = clearPadding(T, program, node_index, slot, vector_stack[vector_dst + slot]);
            } else {
                const matrix = ir.scalarOffset(program, mat_vec.matrix);
                const vector = ir.scalarOffset(program, mat_vec.vector);
                @memset(scalar_stack[scalar_dst..][0..matrix_shape.cols], 0);
                for (0..matrix_shape.rows) |row| {
                    for (0..matrix_shape.cols) |col| scalar_stack[scalar_dst + col] += scalar_stack[matrix + row * matrix_shape.cols + col] * scalar_stack[vector + row];
                }
            }
        },
        .outer => |outer| {
            const rows = program.nodes[outer.lhs].shape.vector;
            const cols = program.nodes[outer.rhs].shape.vector;
            if (comptime ir.usesVectorStack(node)) {
                const rhs = ir.vectorOffset(program, outer.rhs);
                const row_chunks = comptime ir.chunksPerRow(T, node.shape, program.vector_bits);
                inline for (0..rows) |row| {
                    const factor: @Vector(lanes, T) = @splat(tensorElement(T, program, outer.lhs, row, scalar_stack, vector_stack));
                    inline for (0..row_chunks) |chunk| {
                        const slot = row * row_chunks + chunk;
                        vector_stack[vector_dst + slot] = clearPadding(T, program, node_index, slot, factor * vector_stack[rhs + chunk]);
                    }
                }
            } else {
                const lhs = ir.scalarOffset(program, outer.lhs);
                const rhs = ir.scalarOffset(program, outer.rhs);
                for (0..rows) |row| {
                    for (0..cols) |col| scalar_stack[scalar_dst + row * cols + col] = scalar_stack[lhs + row] * scalar_stack[rhs + col];
                }
            }
        },
        .extract => |extract| scalar_stack[scalar_dst] = tensorElement(T, program, extract.input, extract.index, scalar_stack, vector_stack),
    }
}

pub fn Frame(comptime program: anytype) type {
    const T = @TypeOf(program).scalar_type;
    const V = @Vector(ir.stackLanes(T, program.tensor_backend, program.vector_bits), T);
    return struct {
        scalar_stack: [program.scalar_stack_size]T = undefined,
        vector_stack: [program.vector_stack_size]V = undefined,
    };
}

fn executeInto(
    comptime T: type,
    comptime program: anytype,
    inputs: []const T,
    frame: *Frame(program),
    outputs: []T,
) void {
    ir.validate(T, program);
    if (inputs.len < program.input_size) @panic("VM input slice is too small");
    if (outputs.len < ir.outputSize(T, program)) @panic("VM output workspace is too small");
    inline for (0..program.len) |node_index| executeNode(T, program, node_index, inputs, &frame.scalar_stack, &frame.vector_stack);

    var output_offset: usize = 0;
    inline for (program.outputs[0..program.output_len]) |output_node| {
        storeBoundaryTensor(T, program, output_node, &frame.scalar_stack, &frame.vector_stack, outputs, output_offset);
        output_offset += program.nodes[output_node].shape.size();
    }
}

pub fn evalFlatInto(
    comptime T: type,
    comptime program: anytype,
    inputs: []const T,
    frame: *Frame(program),
    outputs: *[ir.outputSize(T, program)]T,
) void {
    executeInto(T, program, inputs, frame, outputs);
}

pub fn evalFlatWithWorkspace(
    comptime T: type,
    comptime program: anytype,
    inputs: []const T,
    frame: *Frame(program),
) [ir.outputSize(T, program)]T {
    var outputs: [ir.outputSize(T, program)]T = undefined;
    executeInto(T, program, inputs, frame, &outputs);
    return outputs;
}

fn frameBytes(comptime program: anytype) usize {
    return @sizeOf(Frame(program));
}

pub fn evalFlat(comptime T: type, comptime program: anytype, inputs: []const T) [ir.outputSize(T, program)]T {
    if (comptime frameBytes(program) > 1024 * 1024) @compileError("VM stacks exceed 1 MiB; use evalFlatWithWorkspace");
    var frame: Frame(program) = .{};
    return evalFlatWithWorkspace(T, program, inputs, &frame);
}

pub fn Value(comptime T: type, comptime shape: ir.Shape) type {
    return switch (shape) {
        .scalar => T,
        .vector => |len| [len]T,
        .matrix => |matrix| [matrix.rows][matrix.cols]T,
    };
}

pub fn Inputs(comptime program: anytype) type {
    const T = @TypeOf(program).scalar_type;
    var types: [program.input_shapes.len]type = undefined;
    inline for (program.input_shapes, 0..) |shape, index| types[index] = Value(T, shape);
    return @Tuple(&types);
}

pub fn Result(comptime program: anytype) type {
    return Value(@TypeOf(program).scalar_type, program.result_shape);
}

fn flattenInputs(comptime program: anytype, inputs: Inputs(program), flat: []@TypeOf(program).scalar_type) void {
    if (flat.len < program.input_size) @panic("VM input workspace is too small");
    var offset: usize = 0;
    inline for (program.input_shapes, 0..) |shape, input_index| switch (shape) {
        .scalar => {
            flat[offset] = inputs[input_index];
            offset += 1;
        },
        .vector => |len| {
            for (0..len) |index| flat[offset + index] = inputs[input_index][index];
            offset += len;
        },
        .matrix => |matrix| {
            for (0..matrix.rows) |row| {
                for (0..matrix.cols) |col| flat[offset + row * matrix.cols + col] = inputs[input_index][row][col];
            }
            offset += matrix.rows * matrix.cols;
        },
    };
}

fn restoreResult(comptime program: anytype, flat: [ir.outputSize(@TypeOf(program).scalar_type, program)]@TypeOf(program).scalar_type) Result(program) {
    const T = @TypeOf(program).scalar_type;
    return switch (program.result_shape) {
        .scalar => flat[0],
        .vector => |len| flat[0..len].*,
        .matrix => |matrix| blk: {
            var result: [matrix.rows][matrix.cols]T = undefined;
            for (0..matrix.rows) |row| {
                for (0..matrix.cols) |col| result[row][col] = flat[row * matrix.cols + col];
            }
            break :blk result;
        },
    };
}

pub fn Workspace(comptime program: anytype) type {
    const T = @TypeOf(program).scalar_type;
    return struct {
        inputs: [program.input_size]T = undefined,
        frame: Frame(program) = .{},
        outputs: [ir.outputSize(T, program)]T = undefined,
    };
}

pub fn evalWithWorkspace(comptime program: anytype, inputs: Inputs(program), workspace: *Workspace(program)) Result(program) {
    flattenInputs(program, inputs, &workspace.inputs);
    executeInto(@TypeOf(program).scalar_type, program, &workspace.inputs, &workspace.frame, &workspace.outputs);
    return restoreResult(program, workspace.outputs);
}

pub fn evalInto(comptime program: anytype, inputs: Inputs(program), workspace: *Workspace(program), result: *Result(program)) void {
    result.* = evalWithWorkspace(program, inputs, workspace);
}

pub fn eval(comptime program: anytype, inputs: Inputs(program)) Result(program) {
    if (comptime @sizeOf(Workspace(program)) > 1024 * 1024) {
        @compileError("typed VM workspace exceeds 1 MiB; use evalWithWorkspace");
    }
    var workspace: Workspace(program) = .{};
    return evalWithWorkspace(program, inputs, &workspace);
}

test "SIMD and scalar VMs agree for a 1024-bit dot product" {
    const dag = @import("dag.zig");
    const V = dag.Vector(f32, 64);
    const model = struct {
        fn call(lhs: *const V, rhs: *const V) dag.Scalar(f32) {
            return lhs.dot(rhs);
        }
    }.call;
    const graph = comptime dag.toDag(f32, model);
    const simd_program = comptime ir.lower(f32, graph, .{});
    const scalar_program = comptime ir.lower(f32, graph, .{ .tensor_backend = .scalar });
    var inputs: [128]f32 = undefined;
    for (0..64) |index| {
        inputs[index] = @floatFromInt(index + 1);
        inputs[64 + index] = 0.5;
    }
    try std.testing.expectEqual(evalFlat(f32, simd_program, &inputs), evalFlat(f32, scalar_program, &inputs));
}

test "1024-bit vector stack executes f16 f32 and f64" {
    const dag = @import("dag.zig");
    const Cases = struct {
        fn run(comptime T: type, comptime len: usize) !void {
            const V = dag.Vector(T, len);
            const model = struct {
                fn call(lhs: *const V, rhs: *const V) dag.Scalar(T) {
                    return lhs.dot(rhs);
                }
            }.call;
            const graph = comptime dag.toDag(T, model);
            const program = comptime ir.lower(T, graph, .{});
            const values: [len]T = @splat(1);
            try std.testing.expectEqual(@as(T, @floatFromInt(len)), eval(program, .{ values, values }));
        }
    };
    try Cases.run(f16, 64);
    try Cases.run(f32, 32);
    try Cases.run(f64, 16);
}

test "matrix transpose and outer stay in vector stack across padding" {
    const matrix_shape = ir.Shape{ .matrix = .{ .rows = 40, .cols = 34 } };
    const rows_shape = ir.Shape{ .vector = 40 };
    const cols_shape = ir.Shape{ .vector = 34 };
    const transpose_program = comptime blk: {
        var writer = ir.Writer(f32, 3, 1).init(1400, .simd, 1024);
        const matrix = writer.append(.{ .shape = matrix_shape, .kernel = .simd, .op = .{ .parameter = 0 } });
        const vector = writer.append(.{ .shape = rows_shape, .kernel = .simd, .op = .{ .parameter = 1360 } });
        writer.output(writer.transposeMatVec(matrix, vector));
        writer.program.input_shapes = &.{ matrix_shape, rows_shape };
        writer.program.result_shape = cols_shape;
        break :blk writer.program;
    };
    const row: [34]f32 = @splat(1);
    const matrix: [40][34]f32 = @splat(row);
    const vector: [40]f32 = @splat(2);
    try std.testing.expectEqual(@as([34]f32, @splat(80)), eval(transpose_program, .{ matrix, vector }));

    const outer_program = comptime blk: {
        var writer = ir.Writer(f32, 3, 1).init(74, .simd, 1024);
        const lhs = writer.append(.{ .shape = rows_shape, .kernel = .simd, .op = .{ .parameter = 0 } });
        const rhs = writer.append(.{ .shape = cols_shape, .kernel = .simd, .op = .{ .parameter = 40 } });
        writer.output(writer.outer(lhs, rhs));
        writer.program.input_shapes = &.{ rows_shape, cols_shape };
        writer.program.result_shape = matrix_shape;
        break :blk writer.program;
    };
    const expected_row: [34]f32 = @splat(6);
    try std.testing.expectEqual(@as([40][34]f32, @splat(expected_row)), eval(outer_program, .{ @as([40]f32, @splat(2)), @as([34]f32, @splat(3)) }));
}

test "caller-owned workspace uses separate scalar and vector stacks" {
    const dag = @import("dag.zig");
    const V = dag.Vector(f32, 4);
    const model = struct {
        fn call(x: *const V) V {
            return x.neg();
        }
    }.call;
    const graph = comptime dag.toDag(f32, model);
    const program = comptime ir.lower(f32, graph, .{});
    var workspace: Workspace(program) = .{};
    var output: Result(program) = undefined;
    evalInto(program, .{[4]f32{ 1, 2, 3, 4 }}, &workspace, &output);
    try std.testing.expectEqual([_]f32{ -1, -2, -3, -4 }, output);
    try std.testing.expectEqual(@as(usize, 0), program.scalar_stack_size);
    try std.testing.expect(program.vector_stack_size > 0);
}

test "padding lanes cannot contaminate reductions" {
    const shape = ir.Shape{ .vector = 3 };
    const program = comptime blk: {
        var writer = ir.Writer(f32, 7, 2).init(3, .simd, 1024);
        const input = writer.append(.{ .shape = shape, .kernel = .simd, .op = .{ .parameter = 0 } });
        const logged = writer.unary(input, .log);
        const zeros = writer.fill(shape, 0);
        writer.output(writer.reduceDot(logged, zeros));
        const divided = writer.binary(input, input, .div);
        const ones = writer.fill(shape, 1);
        writer.output(writer.reduceDot(divided, ones));
        writer.program.input_shapes = &.{shape};
        writer.program.result_shape = .{ .vector = 2 };
        break :blk writer.program;
    };
    try std.testing.expectEqual([_]f32{ 0, 3 }, eval(program, .{[3]f32{ 2, 3, 4 }}));
}
