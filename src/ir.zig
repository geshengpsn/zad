const std = @import("std");
const dag = @import("dag.zig");

pub const Shape = dag.Shape;
pub const UnaryOp = dag.UnaryOp;
pub const BinaryOp = dag.BinaryOp;

pub const TensorBackend = enum {
    simd,
    scalar,
};

pub const Kernel = enum {
    simd,
    scalar,
};

pub const CompileOptions = struct {
    tensor_backend: TensorBackend = .simd,
    vector_bits: usize = 1024,
    optimize: bool = true,
};

pub fn validateScalarType(comptime T: type) void {
    if (@typeInfo(T) != .float) @compileError("IR scalar type must be f16, f32, f64, f80, or f128");
}

pub fn simdLanes(comptime T: type, comptime vector_bits: usize) comptime_int {
    validateScalarType(T);
    if (vector_bits == 0 or vector_bits % @bitSizeOf(T) != 0) {
        @compileError("vector_bits must be a non-zero multiple of the scalar bit width");
    }
    return vector_bits / @bitSizeOf(T);
}

pub fn stackLanes(comptime T: type, comptime backend: TensorBackend, comptime vector_bits: usize) comptime_int {
    return if (backend == .simd) simdLanes(T, vector_bits) else 1;
}

pub fn Node(comptime T: type) type {
    return struct {
        shape: Shape,
        kernel: Kernel,
        op: union(enum) {
            parameter: usize,
            scalar_constant: T,
            tensor_constant: []const T,
            fill: T,
            basis: struct {
                index: usize,
                value: T,
            },
            unary: struct {
                input: usize,
                op: UnaryOp,
            },
            binary: struct {
                lhs: usize,
                rhs: usize,
                op: BinaryOp,
            },
            scale: struct {
                value: usize,
                scalar: usize,
            },
            reduce_dot: struct {
                lhs: usize,
                rhs: usize,
            },
            mat_vec: struct {
                matrix: usize,
                vector: usize,
            },
            transpose_mat_vec: struct {
                matrix: usize,
                vector: usize,
            },
            outer: struct {
                lhs: usize,
                rhs: usize,
            },
            extract: struct {
                input: usize,
                index: usize,
            },
        },
    };
}

pub fn Program(comptime T: type, comptime node_capacity: usize, comptime output_capacity: usize) type {
    validateScalarType(T);
    return struct {
        pub const scalar_type = T;
        pub const NodeType = Node(T);
        pub const node_cap = node_capacity;
        pub const output_cap = output_capacity;

        nodes: [node_capacity]Node(T) = undefined,
        scalar_offsets: [node_capacity]usize = undefined,
        vector_offsets: [node_capacity]usize = undefined,
        len: usize = 0,
        scalar_stack_size: usize = 0,
        vector_stack_size: usize = 0,
        outputs: [output_capacity]usize = undefined,
        output_len: usize = 0,
        input_size: usize = 0,
        input_shapes: []const Shape = &.{},
        result_shape: Shape = .scalar,
        tensor_backend: TensorBackend = .simd,
        vector_bits: usize = 1024,
    };
}

pub fn kernelFor(shape: Shape, backend: TensorBackend) Kernel {
    if (shape == .scalar or backend == .scalar) return .scalar;
    return .simd;
}

pub fn usesVectorStack(node: anytype) bool {
    return node.shape != .scalar and node.kernel == .simd;
}

pub fn chunksPerRow(comptime T: type, comptime shape: Shape, comptime vector_bits: usize) usize {
    const lanes = simdLanes(T, vector_bits);
    const columns = switch (shape) {
        .vector => |len| len,
        .matrix => |matrix| matrix.cols,
        .scalar => return 0,
    };
    return (columns + lanes - 1) / lanes;
}

pub fn vectorSlots(comptime T: type, comptime shape: Shape, comptime vector_bits: usize) usize {
    const chunks = chunksPerRow(T, shape, vector_bits);
    return switch (shape) {
        .scalar => 0,
        .vector => chunks,
        .matrix => |matrix| matrix.rows * chunks,
    };
}

pub fn Writer(comptime T: type, comptime node_capacity: usize, comptime output_capacity: usize) type {
    return struct {
        const Self = @This();

        program: Program(T, node_capacity, output_capacity),

        pub fn init(input_size: usize, backend: TensorBackend, vector_bits: usize) Self {
            if (backend == .simd) _ = simdLanes(T, vector_bits);
            return .{ .program = .{
                .input_size = input_size,
                .tensor_backend = backend,
                .vector_bits = vector_bits,
            } };
        }

        pub fn append(self: *Self, node: Node(T)) usize {
            if (self.program.len >= node_capacity) @compileError("IR writer node capacity exceeded");
            const index = self.program.len;
            self.program.nodes[index] = node;
            if (usesVectorStack(node)) {
                self.program.vector_offsets[index] = self.program.vector_stack_size;
                self.program.vector_stack_size += vectorSlots(T, node.shape, self.program.vector_bits);
            } else {
                self.program.scalar_offsets[index] = self.program.scalar_stack_size;
                self.program.scalar_stack_size += node.shape.size();
            }
            self.program.len += 1;
            return index;
        }

        pub fn output(self: *Self, node: usize) void {
            if (self.program.output_len >= output_capacity) @compileError("IR writer output capacity exceeded");
            self.program.outputs[self.program.output_len] = node;
            self.program.output_len += 1;
        }

        pub fn fill(self: *Self, shape: Shape, value: T) usize {
            return self.append(.{
                .shape = shape,
                .kernel = kernelFor(shape, self.program.tensor_backend),
                .op = .{ .fill = value },
            });
        }

        pub fn basis(self: *Self, shape: Shape, index: usize, value: T) usize {
            if (index >= shape.size()) @compileError("IR basis index out of bounds");
            return self.append(.{
                .shape = shape,
                .kernel = kernelFor(shape, self.program.tensor_backend),
                .op = .{ .basis = .{ .index = index, .value = value } },
            });
        }

        pub fn unary(self: *Self, input: usize, op: UnaryOp) usize {
            const shape = self.program.nodes[input].shape;
            return self.append(.{
                .shape = shape,
                .kernel = kernelFor(shape, self.program.tensor_backend),
                .op = .{ .unary = .{ .input = input, .op = op } },
            });
        }

        pub fn binary(self: *Self, lhs: usize, rhs: usize, op: BinaryOp) usize {
            const shape = self.program.nodes[lhs].shape;
            if (!shape.eql(self.program.nodes[rhs].shape)) @compileError("IR binary shape mismatch");
            return self.append(.{
                .shape = shape,
                .kernel = kernelFor(shape, self.program.tensor_backend),
                .op = .{ .binary = .{ .lhs = lhs, .rhs = rhs, .op = op } },
            });
        }

        pub fn scale(self: *Self, value: usize, scalar: usize) usize {
            const shape = self.program.nodes[value].shape;
            if (self.program.nodes[scalar].shape != .scalar) @compileError("IR scale requires a scalar operand");
            return self.append(.{
                .shape = shape,
                .kernel = kernelFor(shape, self.program.tensor_backend),
                .op = .{ .scale = .{ .value = value, .scalar = scalar } },
            });
        }

        pub fn reduceDot(self: *Self, lhs: usize, rhs: usize) usize {
            if (!self.program.nodes[lhs].shape.eql(self.program.nodes[rhs].shape)) @compileError("IR reduce_dot shape mismatch");
            return self.append(.{
                .shape = .scalar,
                .kernel = self.program.nodes[lhs].kernel,
                .op = .{ .reduce_dot = .{ .lhs = lhs, .rhs = rhs } },
            });
        }

        pub fn matVec(self: *Self, matrix: usize, vector: usize) usize {
            const matrix_shape = switch (self.program.nodes[matrix].shape) {
                .matrix => |shape| shape,
                else => @compileError("IR mat_vec lhs must be a matrix"),
            };
            const vector_len = switch (self.program.nodes[vector].shape) {
                .vector => |len| len,
                else => @compileError("IR mat_vec rhs must be a vector"),
            };
            if (matrix_shape.cols != vector_len) @compileError("IR mat_vec inner dimension mismatch");
            const shape = Shape{ .vector = matrix_shape.rows };
            return self.append(.{
                .shape = shape,
                .kernel = kernelFor(shape, self.program.tensor_backend),
                .op = .{ .mat_vec = .{ .matrix = matrix, .vector = vector } },
            });
        }

        pub fn transposeMatVec(self: *Self, matrix: usize, vector: usize) usize {
            const matrix_shape = switch (self.program.nodes[matrix].shape) {
                .matrix => |shape| shape,
                else => @compileError("IR transpose_mat_vec lhs must be a matrix"),
            };
            const vector_len = switch (self.program.nodes[vector].shape) {
                .vector => |len| len,
                else => @compileError("IR transpose_mat_vec rhs must be a vector"),
            };
            if (matrix_shape.rows != vector_len) @compileError("IR transpose_mat_vec inner dimension mismatch");
            const shape = Shape{ .vector = matrix_shape.cols };
            return self.append(.{
                .shape = shape,
                .kernel = kernelFor(shape, self.program.tensor_backend),
                .op = .{ .transpose_mat_vec = .{ .matrix = matrix, .vector = vector } },
            });
        }

        pub fn outer(self: *Self, lhs: usize, rhs: usize) usize {
            const rows = switch (self.program.nodes[lhs].shape) {
                .vector => |len| len,
                else => @compileError("IR outer lhs must be a vector"),
            };
            const cols = switch (self.program.nodes[rhs].shape) {
                .vector => |len| len,
                else => @compileError("IR outer rhs must be a vector"),
            };
            const shape = Shape{ .matrix = .{ .rows = rows, .cols = cols } };
            return self.append(.{
                .shape = shape,
                .kernel = kernelFor(shape, self.program.tensor_backend),
                .op = .{ .outer = .{ .lhs = lhs, .rhs = rhs } },
            });
        }

        pub fn extract(self: *Self, input: usize, index: usize) usize {
            if (index >= self.program.nodes[input].shape.size()) @compileError("IR extract index out of bounds");
            return self.append(.{
                .shape = .scalar,
                .kernel = .scalar,
                .op = .{ .extract = .{ .input = input, .index = index } },
            });
        }
    };
}

pub fn scalarOffset(comptime program: anytype, comptime node_index: usize) usize {
    return program.scalar_offsets[node_index];
}

pub fn vectorOffset(comptime program: anytype, comptime node_index: usize) usize {
    return program.vector_offsets[node_index];
}

pub fn outputSize(comptime T: type, comptime program: anytype) usize {
    _ = T;
    var size: usize = 0;
    for (program.outputs[0..program.output_len]) |output| size += program.nodes[output].shape.size();
    return size;
}

pub fn inputShapeSize(comptime program: anytype) usize {
    var size: usize = 0;
    for (program.input_shapes) |shape| size += shape.size();
    return size;
}

pub fn validate(comptime T: type, comptime program: anytype) void {
    validateScalarType(T);
    if (program.tensor_backend == .simd) _ = simdLanes(T, program.vector_bits);
    comptime var expected_scalar_offset: usize = 0;
    comptime var expected_vector_offset: usize = 0;
    inline for (0..program.len) |index| {
        const node = comptime program.nodes[index];
        if (comptime usesVectorStack(node)) {
            if (comptime program.vector_offsets[index] != expected_vector_offset) @compileError("IR vector stack offsets are not contiguous");
            expected_vector_offset += comptime vectorSlots(T, node.shape, program.vector_bits);
        } else {
            if (comptime program.scalar_offsets[index] != expected_scalar_offset) @compileError("IR scalar stack offsets are not contiguous");
            expected_scalar_offset += comptime node.shape.size();
        }
        switch (node.op) {
            .parameter => |offset| if (comptime offset + node.shape.size() > program.input_size) {
                @compileError(std.fmt.comptimePrint(
                    "IR parameter range {}..{} exceeds input size {}",
                    .{ offset, offset + node.shape.size(), program.input_size },
                ));
            },
            .scalar_constant => if (comptime node.shape != .scalar) @compileError("scalar constant must have scalar shape"),
            .tensor_constant => |values| if (comptime values.len != node.shape.size()) @compileError("IR tensor constant shape mismatch"),
            .fill => {},
            .basis => |basis| if (comptime basis.index >= node.shape.size()) @compileError("IR basis index out of bounds"),
            .unary => |unary| {
                if (comptime unary.input >= index) @compileError("IR unary input must reference an earlier node");
                if (comptime !node.shape.eql(program.nodes[unary.input].shape)) @compileError("IR unary shape mismatch");
            },
            .binary => |binary| {
                if (comptime binary.lhs >= index or binary.rhs >= index) @compileError("IR binary operands must reference earlier nodes");
                if (comptime !node.shape.eql(program.nodes[binary.lhs].shape) or !node.shape.eql(program.nodes[binary.rhs].shape)) @compileError("IR binary shape mismatch");
            },
            .scale => |scale| {
                if (comptime scale.value >= index or scale.scalar >= index) @compileError("IR scale operands must reference earlier nodes");
                if (comptime !node.shape.eql(program.nodes[scale.value].shape) or program.nodes[scale.scalar].shape != .scalar) @compileError("IR scale shape mismatch");
            },
            .reduce_dot => |dot| {
                if (comptime dot.lhs >= index or dot.rhs >= index) @compileError("IR reduce_dot operands must reference earlier nodes");
                if (comptime node.shape != .scalar or !program.nodes[dot.lhs].shape.eql(program.nodes[dot.rhs].shape)) @compileError("IR reduce_dot shape mismatch");
            },
            .mat_vec => |mat_vec| {
                if (comptime mat_vec.matrix >= index or mat_vec.vector >= index) @compileError("IR mat_vec operands must reference earlier nodes");
                const matrix = switch (program.nodes[mat_vec.matrix].shape) {
                    .matrix => |shape| shape,
                    else => @compileError("IR mat_vec lhs must be a matrix"),
                };
                const vector_len = switch (program.nodes[mat_vec.vector].shape) {
                    .vector => |len| len,
                    else => @compileError("IR mat_vec rhs must be a vector"),
                };
                if (comptime matrix.cols != vector_len or !node.shape.eql(.{ .vector = matrix.rows })) @compileError("IR mat_vec shape mismatch");
            },
            .transpose_mat_vec => |mat_vec| {
                if (comptime mat_vec.matrix >= index or mat_vec.vector >= index) @compileError("IR transpose_mat_vec operands must reference earlier nodes");
                const matrix = switch (program.nodes[mat_vec.matrix].shape) {
                    .matrix => |shape| shape,
                    else => @compileError("IR transpose_mat_vec lhs must be a matrix"),
                };
                const vector_len = switch (program.nodes[mat_vec.vector].shape) {
                    .vector => |len| len,
                    else => @compileError("IR transpose_mat_vec rhs must be a vector"),
                };
                if (comptime matrix.rows != vector_len or !node.shape.eql(.{ .vector = matrix.cols })) @compileError("IR transpose_mat_vec shape mismatch");
            },
            .outer => |outer| {
                if (comptime outer.lhs >= index or outer.rhs >= index) @compileError("IR outer operands must reference earlier nodes");
                const rows = switch (program.nodes[outer.lhs].shape) {
                    .vector => |len| len,
                    else => @compileError("IR outer lhs must be a vector"),
                };
                const cols = switch (program.nodes[outer.rhs].shape) {
                    .vector => |len| len,
                    else => @compileError("IR outer rhs must be a vector"),
                };
                if (comptime !node.shape.eql(.{ .matrix = .{ .rows = rows, .cols = cols } })) @compileError("IR outer shape mismatch");
            },
            .extract => |extract| {
                if (comptime extract.input >= index) @compileError("IR extract input must reference an earlier node");
                if (comptime extract.index >= program.nodes[extract.input].shape.size()) @compileError("IR extract index out of bounds");
                if (comptime node.shape != .scalar) @compileError("IR extract must produce a scalar");
            },
        }
        const expected_kernel = comptime switch (node.op) {
            .reduce_dot => |dot| program.nodes[dot.lhs].kernel,
            else => kernelFor(node.shape, program.tensor_backend),
        };
        if (comptime node.kernel != expected_kernel) @compileError("IR node kernel does not match backend and operation");
    }
    inline for (program.outputs[0..program.output_len]) |output| {
        if (comptime output >= program.len) @compileError("IR output references an invalid node");
    }
    if (comptime expected_scalar_offset != program.scalar_stack_size) @compileError("IR scalar stack size does not match node shapes");
    if (comptime expected_vector_offset != program.vector_stack_size) @compileError("IR vector stack size does not match node shapes");
    if (comptime inputShapeSize(program) != program.input_size) @compileError("IR input shapes do not match input size");
    if (comptime outputSize(T, program) != program.result_shape.size()) @compileError("IR result shape does not match flattened outputs");
}

pub fn lower(comptime T: type, comptime graph: anytype, comptime options: CompileOptions) Program(T, graph.nodes.len, 1) {
    validateScalarType(T);
    var writer = Writer(T, graph.nodes.len, 1).init(graph.input_size, options.tensor_backend, options.vector_bits);
    inline for (graph.nodes) |node| {
        _ = writer.append(.{
            .shape = node.shape,
            .kernel = switch (node.op) {
                .dot => |dot| writer.program.nodes[dot.lhs].kernel,
                else => kernelFor(node.shape, options.tensor_backend),
            },
            .op = switch (node.op) {
                .parameter => |offset| .{ .parameter = offset },
                .scalar_constant => |value| .{ .scalar_constant = value },
                .tensor_constant => |values| .{ .tensor_constant = values },
                .unary => |unary| .{ .unary = .{ .input = unary.input, .op = unary.op } },
                .binary => |binary| .{ .binary = .{ .lhs = binary.lhs, .rhs = binary.rhs, .op = binary.op } },
                .scale => |scale| .{ .scale = .{ .value = scale.value, .scalar = scale.scalar } },
                .dot => |dot| .{ .reduce_dot = .{ .lhs = dot.lhs, .rhs = dot.rhs } },
                .mat_vec => |mat_vec| .{ .mat_vec = .{ .matrix = mat_vec.matrix, .vector = mat_vec.vector } },
            },
        });
    }
    writer.output(graph.output_node);
    writer.program.input_shapes = &graph.input_shapes;
    writer.program.result_shape = graph.output_shape;
    const program = writer.program;
    validate(T, program);
    return program;
}

test "lower preserves tensor kernels and defaults to 1024-bit SIMD" {
    const S = dag.Scalar(f32);
    const V = dag.Vector(f32, 4);
    const model = struct {
        fn call(x: *const V, scale: *const S) V {
            return x.mul(scale);
        }
    }.call;
    const graph = comptime dag.toDag(f32, model);
    const program = comptime lower(f32, graph, .{});
    try std.testing.expectEqual(@as(usize, 32), simdLanes(f32, program.vector_bits));
    try std.testing.expectEqual(Kernel.simd, program.nodes[2].kernel);

    const scalar_program = comptime lower(f32, graph, .{ .tensor_backend = .scalar });
    try std.testing.expectEqual(Kernel.scalar, scalar_program.nodes[2].kernel);
}
