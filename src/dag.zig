const std = @import("std");

pub const UnaryOp = enum { neg, abs, exp, log, sqrt, sin, cos, tan };
pub const BinaryOp = enum { add, sub, mul, div };

pub const Shape = union(enum) {
    scalar,
    vector: usize,
    matrix: struct { rows: usize, cols: usize },

    pub fn size(self: Shape) usize {
        return switch (self) {
            .scalar => 1,
            .vector => |len| len,
            .matrix => |matrix| matrix.rows * matrix.cols,
        };
    }

    pub fn eql(lhs: Shape, rhs: Shape) bool {
        return switch (lhs) {
            .scalar => rhs == .scalar,
            .vector => |lhs_len| switch (rhs) {
                .vector => |rhs_len| lhs_len == rhs_len,
                else => false,
            },
            .matrix => |lhs_matrix| switch (rhs) {
                .matrix => |rhs_matrix| lhs_matrix.rows == rhs_matrix.rows and lhs_matrix.cols == rhs_matrix.cols,
                else => false,
            },
        };
    }
};

pub fn Node(comptime T: type) type {
    return struct {
        shape: Shape,
        op: union(enum) {
            parameter: usize,
            scalar_constant: T,
            tensor_constant: []const T,
            unary: struct { input: usize, op: UnaryOp },
            binary: struct { lhs: usize, rhs: usize, op: BinaryOp },
            scale: struct { value: usize, scalar: usize },
            dot: struct { lhs: usize, rhs: usize },
            mat_vec: struct { matrix: usize, vector: usize },
        },
    };
}

fn Context(comptime T: type) type {
    return struct {
        nodes: []Node(T),
        len: usize = 0,

        fn append(self: *@This(), node: Node(T)) usize {
            const index = self.len;
            if (index < self.nodes.len) self.nodes[index] = node;
            self.len += 1;
            return index;
        }
    };
}

fn sameContext(comptime T: type, lhs: *Context(T), rhs: *Context(T)) void {
    if (lhs != rhs) @compileError("cannot combine values from different DAG contexts");
}

fn applyUnary(comptime T: type, op: UnaryOp, value: T) T {
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

fn applyBinary(comptime T: type, op: BinaryOp, lhs: T, rhs: T) T {
    return switch (op) {
        .add => lhs + rhs,
        .sub => lhs - rhs,
        .mul => lhs * rhs,
        .div => lhs / rhs,
    };
}

pub fn Scalar(comptime T: type) type {
    return struct {
        const Self = @This();

        pub const dag_value = true;
        pub const scalar_type = T;
        pub const shape: Shape = .scalar;

        context: ?*Context(T) = null,
        node: usize = 0,
        literal: ?T = null,

        pub fn parameter(context: *Context(T), offset: usize) Self {
            return .{ .context = context, .node = context.append(.{ .shape = shape, .op = .{ .parameter = offset } }) };
        }

        pub fn c(value: T) Self {
            return .{ .literal = value };
        }

        pub fn resolve(self: *const Self, context: *Context(T)) usize {
            if (self.context) |existing| {
                sameContext(T, existing, context);
                return self.node;
            }
            return context.append(.{ .shape = shape, .op = .{ .scalar_constant = self.literal.? } });
        }

        fn fromNode(context: *Context(T), node: usize) Self {
            return .{ .context = context, .node = node };
        }

        fn unary(value: *const Self, op: UnaryOp) Self {
            if (value.literal) |literal| return c(applyUnary(T, op, literal));
            const context = value.context.?;
            return fromNode(context, context.append(.{ .shape = shape, .op = .{ .unary = .{ .input = value.node, .op = op } } }));
        }

        fn binary(lhs: *const Self, rhs: *const Self, op: BinaryOp) Self {
            if (lhs.literal) |lhs_value| if (rhs.literal) |rhs_value| return c(applyBinary(T, op, lhs_value, rhs_value));
            const context = lhs.context orelse rhs.context.?;
            return fromNode(context, context.append(.{
                .shape = shape,
                .op = .{ .binary = .{ .lhs = lhs.resolve(context), .rhs = rhs.resolve(context), .op = op } },
            }));
        }

        pub fn neg(value: *const Self) Self {
            return value.unary(.neg);
        }
        pub fn abs(value: *const Self) Self {
            return value.unary(.abs);
        }
        pub fn exp(value: *const Self) Self {
            return value.unary(.exp);
        }
        pub fn log(value: *const Self) Self {
            return value.unary(.log);
        }
        pub fn sqrt(value: *const Self) Self {
            return value.unary(.sqrt);
        }
        pub fn sin(value: *const Self) Self {
            return value.unary(.sin);
        }
        pub fn cos(value: *const Self) Self {
            return value.unary(.cos);
        }
        pub fn tan(value: *const Self) Self {
            return value.unary(.tan);
        }
        pub fn add(lhs: *const Self, rhs: *const Self) Self {
            return lhs.binary(rhs, .add);
        }
        pub fn sub(lhs: *const Self, rhs: *const Self) Self {
            return lhs.binary(rhs, .sub);
        }
        pub fn mul(lhs: *const Self, rhs: *const Self) Self {
            return lhs.binary(rhs, .mul);
        }
        pub fn div(lhs: *const Self, rhs: *const Self) Self {
            return lhs.binary(rhs, .div);
        }
    };
}

pub fn Vector(comptime T: type, comptime len: usize) type {
    return struct {
        const Self = @This();

        pub const dag_value = true;
        pub const scalar_type = T;
        pub const shape: Shape = .{ .vector = len };
        pub const length = len;

        context: ?*Context(T) = null,
        node: usize = 0,
        literal: ?[len]T = null,

        pub fn parameter(context: *Context(T), offset: usize) Self {
            return .{ .context = context, .node = context.append(.{ .shape = shape, .op = .{ .parameter = offset } }) };
        }

        pub fn c(values: [len]T) Self {
            return .{ .literal = values };
        }

        pub fn resolve(self: *const Self, context: *Context(T)) usize {
            if (self.context) |existing| {
                sameContext(T, existing, context);
                return self.node;
            }
            return context.append(.{ .shape = shape, .op = .{ .tensor_constant = &self.literal.? } });
        }

        fn fromNode(context: *Context(T), node: usize) Self {
            return .{ .context = context, .node = node };
        }

        fn binary(lhs: *const Self, rhs: *const Self, op: BinaryOp) Self {
            if (lhs.literal) |lhs_values| if (rhs.literal) |rhs_values| {
                var values: [len]T = undefined;
                for (0..len) |index| values[index] = applyBinary(T, op, lhs_values[index], rhs_values[index]);
                return c(values);
            };
            const context = lhs.context orelse rhs.context.?;
            return fromNode(context, context.append(.{
                .shape = shape,
                .op = .{ .binary = .{ .lhs = lhs.resolve(context), .rhs = rhs.resolve(context), .op = op } },
            }));
        }

        pub fn neg(value: *const Self) Self {
            if (value.literal) |literal| {
                var result: [len]T = undefined;
                for (literal, 0..) |element, index| result[index] = -element;
                return c(result);
            }
            const context = value.context.?;
            return fromNode(context, context.append(.{ .shape = shape, .op = .{ .unary = .{ .input = value.node, .op = .neg } } }));
        }

        pub fn add(lhs: *const Self, rhs: *const Self) Self {
            return lhs.binary(rhs, .add);
        }
        pub fn sub(lhs: *const Self, rhs: *const Self) Self {
            return lhs.binary(rhs, .sub);
        }
        pub fn hadamard(lhs: *const Self, rhs: *const Self) Self {
            return lhs.binary(rhs, .mul);
        }
        pub fn div(lhs: *const Self, rhs: *const Self) Self {
            return lhs.binary(rhs, .div);
        }

        pub fn mul(value: *const Self, scalar: *const Scalar(T)) Self {
            if (value.literal) |elements| if (scalar.literal) |factor| {
                var result: [len]T = undefined;
                for (elements, 0..) |element, index| result[index] = element * factor;
                return c(result);
            };
            const context = value.context orelse scalar.context.?;
            return fromNode(context, context.append(.{
                .shape = shape,
                .op = .{ .scale = .{ .value = value.resolve(context), .scalar = scalar.resolve(context) } },
            }));
        }

        pub fn dot(lhs: *const Self, rhs: *const Self) Scalar(T) {
            if (len == 0) @compileError("dot product requires a non-empty vector");
            if (lhs.literal) |lhs_values| if (rhs.literal) |rhs_values| {
                var result: T = 0;
                for (0..len) |index| result += lhs_values[index] * rhs_values[index];
                return Scalar(T).c(result);
            };
            const context = lhs.context orelse rhs.context.?;
            return Scalar(T).fromNode(context, context.append(.{
                .shape = .scalar,
                .op = .{ .dot = .{ .lhs = lhs.resolve(context), .rhs = rhs.resolve(context) } },
            }));
        }
    };
}

pub fn Matrix(comptime T: type, comptime rows: usize, comptime cols: usize) type {
    return struct {
        const Self = @This();

        pub const dag_value = true;
        pub const scalar_type = T;
        pub const shape: Shape = .{ .matrix = .{ .rows = rows, .cols = cols } };
        pub const row_count = rows;
        pub const col_count = cols;

        context: ?*Context(T) = null,
        node: usize = 0,
        literal: ?[rows][cols]T = null,

        pub fn parameter(context: *Context(T), offset: usize) Self {
            return .{ .context = context, .node = context.append(.{ .shape = shape, .op = .{ .parameter = offset } }) };
        }

        pub fn c(values: [rows][cols]T) Self {
            return .{ .literal = values };
        }

        pub fn resolve(self: *const Self, context: *Context(T)) usize {
            if (self.context) |existing| {
                sameContext(T, existing, context);
                return self.node;
            }
            const flat: *const [rows * cols]T = @ptrCast(&self.literal.?);
            return context.append(.{ .shape = shape, .op = .{ .tensor_constant = flat } });
        }

        fn fromNode(context: *Context(T), node: usize) Self {
            return .{ .context = context, .node = node };
        }

        fn binary(lhs: *const Self, rhs: *const Self, op: BinaryOp) Self {
            if (lhs.literal) |lhs_values| if (rhs.literal) |rhs_values| {
                var values: [rows][cols]T = undefined;
                for (0..rows) |row| {
                    for (0..cols) |col| values[row][col] = applyBinary(T, op, lhs_values[row][col], rhs_values[row][col]);
                }
                return c(values);
            };
            const context = lhs.context orelse rhs.context.?;
            return fromNode(context, context.append(.{
                .shape = shape,
                .op = .{ .binary = .{ .lhs = lhs.resolve(context), .rhs = rhs.resolve(context), .op = op } },
            }));
        }

        pub fn add(lhs: *const Self, rhs: *const Self) Self {
            return lhs.binary(rhs, .add);
        }
        pub fn sub(lhs: *const Self, rhs: *const Self) Self {
            return lhs.binary(rhs, .sub);
        }

        pub fn mul(value: *const Self, scalar: *const Scalar(T)) Self {
            if (value.literal) |elements| if (scalar.literal) |factor| {
                var result: [rows][cols]T = undefined;
                for (0..rows) |row| {
                    for (0..cols) |col| result[row][col] = elements[row][col] * factor;
                }
                return c(result);
            };
            const context = value.context orelse scalar.context.?;
            return fromNode(context, context.append(.{
                .shape = shape,
                .op = .{ .scale = .{ .value = value.resolve(context), .scalar = scalar.resolve(context) } },
            }));
        }

        pub fn matMul(matrix: *const Self, vector: *const Vector(T, cols)) Vector(T, rows) {
            if (cols == 0) @compileError("matrix-vector multiplication requires a non-empty inner dimension");
            if (matrix.literal) |matrix_values| if (vector.literal) |vector_values| {
                var result: [rows]T = @splat(0);
                for (0..rows) |row| {
                    for (0..cols) |col| result[row] += matrix_values[row][col] * vector_values[col];
                }
                return Vector(T, rows).c(result);
            };
            const context = matrix.context orelse vector.context.?;
            return Vector(T, rows).fromNode(context, context.append(.{
                .shape = .{ .vector = rows },
                .op = .{ .mat_vec = .{ .matrix = matrix.resolve(context), .vector = vector.resolve(context) } },
            }));
        }
    };
}

fn valueType(comptime T: type, comptime Value: type) type {
    if (@typeInfo(Value) != .@"struct" or !@hasDecl(Value, "dag_value") or !@hasDecl(Value, "scalar_type") or !@hasDecl(Value, "shape")) {
        @compileError("DAG values must be Scalar, Vector, or Matrix");
    }
    if (Value.scalar_type != T) @compileError("all DAG values must use the same floating-point type");
    return Value;
}

fn inputValueType(comptime T: type, comptime Param: type) type {
    const pointer = switch (@typeInfo(Param)) {
        .pointer => |pointer| pointer,
        else => @compileError("DAG function parameters must be *const Scalar, Vector, or Matrix"),
    };
    if (pointer.size != .one or !pointer.is_const) @compileError("DAG function parameters must be *const pointers");
    return valueType(T, pointer.child);
}

fn functionInfo(comptime T: type, comptime function: anytype) std.builtin.Type.Fn {
    const info = @typeInfo(@TypeOf(function));
    if (info != .@"fn" or info.@"fn".is_var_args) @compileError("to_dag expects a non-variadic function");
    inline for (info.@"fn".params) |param| _ = inputValueType(T, param.type orelse @compileError("DAG parameters cannot be anytype"));
    _ = valueType(T, info.@"fn".return_type orelse @compileError("DAG function requires an explicit return type"));
    return info.@"fn";
}

fn Storage(comptime T: type, comptime function: anytype) type {
    const info = functionInfo(T, function);
    var types: [info.params.len]type = undefined;
    inline for (info.params, 0..) |param, index| types[index] = inputValueType(T, param.type.?);
    return @Tuple(&types);
}

fn runFunction(comptime T: type, comptime function: anytype, context: *Context(T), storage: *Storage(T, function)) functionInfo(T, function).return_type.? {
    const info = functionInfo(T, function);
    var args: std.meta.ArgsTuple(@TypeOf(function)) = undefined;
    var offset: usize = 0;
    inline for (info.params, 0..) |param, index| {
        const Input = inputValueType(T, param.type.?);
        storage[index] = Input.parameter(context, offset);
        args[index] = &storage[index];
        offset += Input.shape.size();
    }
    return @call(.auto, function, args);
}

fn graphNodeCount(comptime T: type, comptime function: anytype) usize {
    @setEvalBranchQuota(1_000_000);
    var empty: [0]Node(T) = .{};
    var context = Context(T){ .nodes = &empty };
    var storage: Storage(T, function) = undefined;
    const result = runFunction(T, function, &context, &storage);
    _ = result.resolve(&context);
    return context.len;
}

pub fn Graph(comptime T: type, comptime node_count: usize, comptime argument_count: usize) type {
    return struct {
        pub const scalar_type = T;
        pub const NodeType = Node(T);

        nodes: [node_count]Node(T),
        input_shapes: [argument_count]Shape,
        input_size: usize,
        output_node: usize,
        output_shape: Shape,
    };
}

fn graphType(comptime T: type, comptime function: anytype) type {
    return Graph(T, graphNodeCount(T, function), functionInfo(T, function).params.len);
}

pub fn toDag(comptime T: type, comptime function: anytype) graphType(T, function) {
    return comptime blk: {
        @setEvalBranchQuota(1_000_000);
        const info = functionInfo(T, function);
        const count = graphNodeCount(T, function);
        var nodes: [count]Node(T) = undefined;
        var context = Context(T){ .nodes = &nodes };
        var storage: Storage(T, function) = undefined;
        const result = runFunction(T, function, &context, &storage);
        const output_node = result.resolve(&context);
        if (context.len != count) @compileError("DAG function changed between count and build passes");

        var input_shapes: [info.params.len]Shape = undefined;
        var input_size: usize = 0;
        for (info.params, 0..) |param, index| {
            const Input = inputValueType(T, param.type.?);
            input_shapes[index] = Input.shape;
            input_size += Input.shape.size();
        }
        break :blk .{
            .nodes = nodes,
            .input_shapes = input_shapes,
            .input_size = input_size,
            .output_node = output_node,
            .output_shape = @TypeOf(result).shape,
        };
    };
}

test "typed DAG preserves tensor operations" {
    const S = Scalar(f32);
    const V = Vector(f32, 4);
    const M = Matrix(f32, 2, 4);
    const model = struct {
        fn call(matrix: *const M, vector: *const V, scale: *const S) Vector(f32, 2) {
            const product = matrix.matMul(vector);
            return product.mul(scale);
        }
    }.call;
    const graph = toDag(f32, model);
    try std.testing.expectEqual(@as(usize, 13), graph.input_size);
    try std.testing.expect(graph.output_shape.eql(.{ .vector = 2 }));
    try std.testing.expectEqual(@as(usize, 5), graph.nodes.len);
    try std.testing.expect(graph.nodes[3].op == .mat_vec);
    try std.testing.expect(graph.nodes[4].op == .scale);
}

test "shared DAG construction remains linear" {
    const S = Scalar(f32);
    const model = struct {
        fn call(x: *const S) S {
            var value = x.add(x);
            for (1..64) |_| value = value.add(&value);
            return value;
        }
    }.call;
    const graph = toDag(f32, model);
    try std.testing.expectEqual(@as(usize, 65), graph.nodes.len);
}
