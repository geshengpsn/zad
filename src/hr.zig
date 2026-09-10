const std = @import("std");
const ir = @import("ir.zig");
const simplify = @import("simplify.zig");

const ValueKind = enum {
    scalar,
    vector,
    matrix,
};

fn Context(comptime T: type) type {
    return struct {
        builder: simplify.Builder(T),

        fn init(codes: []ir.IRCode(T), simplify_enabled: bool) @This() {
            return .{ .builder = simplify.Builder(T).init(codes, simplify_enabled) };
        }

        fn append(self: *@This(), comptime code: ir.IRCode(T)) usize {
            return self.builder.append(code);
        }

        fn len(self: @This()) usize {
            return self.builder.len;
        }
    };
}

fn validateValueType(comptime T: type, comptime Value: type) void {
    if (@typeInfo(Value) != .@"struct" or
        !@hasDecl(Value, "hr_value") or
        !@hasDecl(Value, "scalar_type") or
        !@hasDecl(Value, "value_kind"))
    {
        @compileError("expected an HR Scalar, Vector, or Matrix value");
    }
    if (Value.scalar_type != T) @compileError("cannot mix HR floating-point types");
}

fn selectContext(comptime T: type, lhs: anytype, rhs: anytype) *Context(T) {
    const context = if (lhs.context) |value|
        value
    else if (rhs.context) |value|
        value
    else
        unreachable;
    if (lhs.context) |lhs_context| if (lhs_context != context) @panic("cannot combine values from different HR contexts");
    if (rhs.context) |rhs_context| if (rhs_context != context) @panic("cannot combine values from different HR contexts");
    return context;
}

fn constantSlice(comptime T: type, comptime len: usize, comptime values: [len]T) []const T {
    return &struct {
        const data = values;
    }.data;
}

fn atan2(comptime T: type, y: T, x: T) T {
    return switch (T) {
        f16 => @floatCast(std.math.atan2(@as(f32, y), @as(f32, x))),
        f32, f64 => std.math.atan2(y, x),
        else => unreachable,
    };
}

fn applyUnary(comptime T: type, op: ir.Op1Code, value: T) T {
    return switch (op) {
        .neg => -value,
        .sqrt => @sqrt(value),
        .exp => @exp(value),
        .log => @log(value),
        .sin => @sin(value),
        .cos => @cos(value),
        .abs => @abs(value),
        .sum, .get => unreachable,
    };
}

fn applyBinary(comptime T: type, op: ir.Op2Code, lhs: T, rhs: T) T {
    return switch (op) {
        .add => lhs + rhs,
        .sub => lhs - rhs,
        .mul => lhs * rhs,
        .div => lhs / rhs,
        .atan2 => atan2(T, lhs, rhs),
        .set => unreachable,
    };
}

fn BinaryResult(comptime T: type, comptime Other: type) type {
    validateValueType(T, Other);
    return switch (Other.value_kind) {
        .scalar => Scalar(T),
        .vector => Other,
        .matrix => @compileError("Scalar and Matrix binary operations are not supported"),
    };
}

pub fn Scalar(comptime T: type) type {
    return struct {
        const Self = @This();

        pub const hr_value = true;
        pub const scalar_type = T;
        pub const value_kind = ValueKind.scalar;

        constant_value: ?T = null,
        context: ?*Context(T) = null,
        index: usize = 0,

        pub fn init(value: T) Self {
            return .{ .constant_value = value };
        }

        fn parameter(context: *Context(T), input_index: usize) Self {
            return .{
                .context = context,
                .index = context.append(.{ .scalar_input_index = input_index }),
            };
        }

        fn fromIndex(context: *Context(T), index: usize) Self {
            return .{ .context = context, .index = index };
        }

        fn resolve(self: Self, context: *Context(T)) usize {
            if (self.context) |existing| {
                if (existing != context) @panic("cannot use a Scalar from another HR context");
                return self.index;
            }
            return context.append(.{ .scalar_constant = self.constant_value.? });
        }

        fn unary(self: Self, op: ir.Op1Code) Self {
            if (self.constant_value) |value| return init(applyUnary(T, op, value));
            const context = self.context.?;
            return fromIndex(context, context.append(.{
                .Op1 = .{ .a = self.index, .op = op, .len = 1 },
            }));
        }

        fn binary(self: Self, other: anytype, op: ir.Op2Code) BinaryResult(T, @TypeOf(other)) {
            const Other = @TypeOf(other);
            const Result = BinaryResult(T, Other);
            if (comptime Other.value_kind == .scalar) {
                if (self.constant_value) |lhs| if (other.constant_value) |rhs| return Result.init(applyBinary(T, op, lhs, rhs));
                const context = selectContext(T, self, other);
                return Result.fromIndex(context, context.append(.{
                    .Op2 = .{
                        .lhs = self.resolve(context),
                        .rhs = other.resolve(context),
                        .op = op,
                        .len = 1,
                    },
                }));
            }

            if (self.constant_value) |lhs| if (other.constant_values) |rhs| {
                var values: [Other.length]T = undefined;
                for (0..Other.length) |index| values[index] = applyBinary(T, op, lhs, rhs[index]);
                return Result.init(values);
            };
            const context = selectContext(T, self, other);
            return Result.fromIndex(context, context.append(.{
                .Op2 = .{
                    .lhs = self.resolve(context),
                    .rhs = other.resolve(context),
                    .op = op,
                    .len = Other.length,
                },
            }));
        }

        pub fn add(self: Self, other: anytype) BinaryResult(T, @TypeOf(other)) {
            return self.binary(other, .add);
        }
        pub fn sub(self: Self, other: anytype) BinaryResult(T, @TypeOf(other)) {
            return self.binary(other, .sub);
        }
        pub fn mul(self: Self, other: anytype) BinaryResult(T, @TypeOf(other)) {
            return self.binary(other, .mul);
        }
        pub fn div(self: Self, other: anytype) BinaryResult(T, @TypeOf(other)) {
            return self.binary(other, .div);
        }
        pub fn atan2(self: Self, other: anytype) BinaryResult(T, @TypeOf(other)) {
            return self.binary(other, .atan2);
        }
        pub fn neg(self: Self) Self {
            return self.unary(.neg);
        }
        pub fn sqrt(self: Self) Self {
            return self.unary(.sqrt);
        }
        pub fn exp(self: Self) Self {
            return self.unary(.exp);
        }
        pub fn log(self: Self) Self {
            return self.unary(.log);
        }
        pub fn sin(self: Self) Self {
            return self.unary(.sin);
        }
        pub fn cos(self: Self) Self {
            return self.unary(.cos);
        }
        pub fn abs(self: Self) Self {
            return self.unary(.abs);
        }

        pub fn mulAdd(self: Self, b: Self, c: Self) Self {
            if (self.constant_value) |a_value| if (b.constant_value) |b_value| if (c.constant_value) |c_value| {
                return init(@mulAdd(T, a_value, b_value, c_value));
            };
            const context = if (self.context) |value|
                value
            else if (b.context) |value|
                value
            else
                c.context.?;
            if (self.context) |existing| if (existing != context) @panic("cannot mix HR contexts");
            if (b.context) |existing| if (existing != context) @panic("cannot mix HR contexts");
            if (c.context) |existing| if (existing != context) @panic("cannot mix HR contexts");
            return fromIndex(context, context.append(.{
                .muladd = .{
                    .a = self.resolve(context),
                    .b = b.resolve(context),
                    .c = c.resolve(context),
                    .len = 1,
                },
            }));
        }
    };
}

pub fn Vector(comptime len: usize, comptime T: type) type {
    if (len < 2) @compileError("Vector length must be at least 2; use Scalar for one value");
    return struct {
        const Self = @This();

        pub const hr_value = true;
        pub const scalar_type = T;
        pub const value_kind = ValueKind.vector;
        pub const length = len;

        constant_values: ?[len]T = null,
        context: ?*Context(T) = null,
        index: usize = 0,

        pub fn init(values: [len]T) Self {
            return .{ .constant_values = values };
        }

        fn parameter(context: *Context(T), input_index: usize) Self {
            return .{
                .context = context,
                .index = context.append(.{ .vec_input = .{ .input_index = input_index, .len = len } }),
            };
        }

        fn fromIndex(context: *Context(T), index: usize) Self {
            return .{ .context = context, .index = index };
        }

        fn resolve(self: Self, context: *Context(T)) usize {
            if (self.context) |existing| {
                if (existing != context) @panic("cannot use a Vector from another HR context");
                return self.index;
            }
            return context.append(.{ .vec_constant = constantSlice(T, len, self.constant_values.?) });
        }

        fn unary(self: Self, op: ir.Op1Code) Self {
            if (self.constant_values) |input| {
                var values: [len]T = undefined;
                for (input, 0..) |value, index| values[index] = applyUnary(T, op, value);
                return init(values);
            }
            const context = self.context.?;
            return fromIndex(context, context.append(.{ .Op1 = .{ .a = self.index, .op = op, .len = len } }));
        }

        fn binary(self: Self, other: anytype, op: ir.Op2Code) Self {
            const Other = @TypeOf(other);
            validateValueType(T, Other);
            if (comptime Other.value_kind == .matrix) @compileError("Vector and Matrix binary operations are not supported");
            if (comptime Other.value_kind == .vector and Other.length != len) @compileError("Vector lengths must match");

            if (self.constant_values) |lhs| {
                if (comptime Other.value_kind == .scalar) {
                    if (other.constant_value) |rhs| {
                        var values: [len]T = undefined;
                        for (0..len) |index| values[index] = applyBinary(T, op, lhs[index], rhs);
                        return init(values);
                    }
                } else if (other.constant_values) |rhs| {
                    var values: [len]T = undefined;
                    for (0..len) |index| values[index] = applyBinary(T, op, lhs[index], rhs[index]);
                    return init(values);
                }
            }

            const context = selectContext(T, self, other);
            return fromIndex(context, context.append(.{
                .Op2 = .{
                    .lhs = self.resolve(context),
                    .rhs = other.resolve(context),
                    .op = op,
                    .len = len,
                },
            }));
        }

        pub fn add(self: Self, other: anytype) Self {
            return self.binary(other, .add);
        }
        pub fn sub(self: Self, other: anytype) Self {
            return self.binary(other, .sub);
        }
        pub fn mul(self: Self, other: anytype) Self {
            return self.binary(other, .mul);
        }
        pub fn div(self: Self, other: anytype) Self {
            return self.binary(other, .div);
        }
        pub fn atan2(self: Self, other: anytype) Self {
            return self.binary(other, .atan2);
        }
        pub fn neg(self: Self) Self {
            return self.unary(.neg);
        }
        pub fn sqrt(self: Self) Self {
            return self.unary(.sqrt);
        }
        pub fn exp(self: Self) Self {
            return self.unary(.exp);
        }
        pub fn log(self: Self) Self {
            return self.unary(.log);
        }
        pub fn sin(self: Self) Self {
            return self.unary(.sin);
        }
        pub fn cos(self: Self) Self {
            return self.unary(.cos);
        }
        pub fn abs(self: Self) Self {
            return self.unary(.abs);
        }

        pub fn sum(self: Self) Scalar(T) {
            if (self.constant_values) |values| return Scalar(T).init(@reduce(.Add, @as(@Vector(len, T), values)));
            const context = self.context.?;
            return Scalar(T).fromIndex(context, context.append(.{ .Op1 = .{ .a = self.index, .op = .sum, .len = 1 } }));
        }

        pub fn dot(self: Self, other: Self) Scalar(T) {
            return self.mul(other).sum();
        }

        pub fn get(self: Self, comptime element_index: usize) Scalar(T) {
            if (element_index >= len) @compileError("Vector get index is out of bounds");
            if (self.constant_values) |values| return Scalar(T).init(values[element_index]);
            const context = self.context.?;
            return Scalar(T).fromIndex(context, context.append(.{
                .Op1 = .{ .a = self.index, .op = .{ .get = element_index }, .len = 1 },
            }));
        }

        pub fn set(self: Self, comptime element_index: usize, value: Scalar(T)) Self {
            if (element_index >= len) @compileError("Vector set index is out of bounds");
            if (self.constant_values) |elements| if (value.constant_value) |replacement| {
                var result = elements;
                result[element_index] = replacement;
                return init(result);
            };
            const context = selectContext(T, self, value);
            return fromIndex(context, context.append(.{
                .Op2 = .{
                    .lhs = self.resolve(context),
                    .rhs = value.resolve(context),
                    .op = .{ .set = element_index },
                    .len = len,
                },
            }));
        }

        pub fn mulAdd(self: Self, b: anytype, c: anytype) Self {
            const B = @TypeOf(b);
            const C = @TypeOf(c);
            validateValueType(T, B);
            validateValueType(T, C);
            if (comptime B.value_kind == .matrix or C.value_kind == .matrix) @compileError("Vector mulAdd does not accept Matrix operands");
            if (comptime B.value_kind == .vector and B.length != len) @compileError("Vector mulAdd lengths must match");
            if (comptime C.value_kind == .vector and C.length != len) @compileError("Vector mulAdd lengths must match");

            if (self.constant_values) |a_values| {
                const b_is_constant = if (comptime B.value_kind == .scalar) b.constant_value != null else b.constant_values != null;
                const c_is_constant = if (comptime C.value_kind == .scalar) c.constant_value != null else c.constant_values != null;
                if (b_is_constant and c_is_constant) {
                    var result: [len]T = undefined;
                    for (0..len) |index| {
                        const b_value = if (comptime B.value_kind == .scalar) b.constant_value.? else b.constant_values.?[index];
                        const c_value = if (comptime C.value_kind == .scalar) c.constant_value.? else c.constant_values.?[index];
                        result[index] = @mulAdd(T, a_values[index], b_value, c_value);
                    }
                    return init(result);
                }
            }

            const context = if (self.context) |value|
                value
            else if (b.context) |value|
                value
            else if (c.context) |value|
                value
            else
                unreachable;
            if (self.context) |existing| if (existing != context) @panic("cannot mix HR contexts");
            if (b.context) |existing| if (existing != context) @panic("cannot mix HR contexts");
            if (c.context) |existing| if (existing != context) @panic("cannot mix HR contexts");
            return fromIndex(context, context.append(.{
                .muladd = .{
                    .a = self.resolve(context),
                    .b = b.resolve(context),
                    .c = c.resolve(context),
                    .len = len,
                },
            }));
        }
    };
}

pub fn Matrix(comptime rows: usize, comptime cols: usize, comptime T: type) type {
    if (rows < 2 or cols < 2) @compileError("Matrix rows and columns must both be at least 2");
    return struct {
        const Self = @This();

        pub const hr_value = true;
        pub const scalar_type = T;
        pub const value_kind = ValueKind.matrix;
        pub const row_count = rows;
        pub const col_count = cols;

        row_values: [rows]Vector(cols, T),

        pub fn init(values: [rows][cols]T) Self {
            var result: Self = undefined;
            for (0..rows) |row| result.row_values[row] = Vector(cols, T).init(values[row]);
            return result;
        }

        pub fn add(self: Self, other: Self) Self {
            var result: Self = undefined;
            for (0..rows) |row| result.row_values[row] = self.row_values[row].add(other.row_values[row]);
            return result;
        }

        pub fn sub(self: Self, other: Self) Self {
            var result: Self = undefined;
            for (0..rows) |row| result.row_values[row] = self.row_values[row].sub(other.row_values[row]);
            return result;
        }

        pub fn scale(self: Self, scalar: Scalar(T)) Self {
            var result: Self = undefined;
            for (0..rows) |row| result.row_values[row] = self.row_values[row].mul(scalar);
            return result;
        }

        pub fn mul(self: Self, vector: Vector(cols, T)) Vector(rows, T) {
            var result = Vector(rows, T).init(@splat(0));
            for (0..rows) |row| result = result.set(row, self.row_values[row].dot(vector));
            return result;
        }
    };
}

fn irValueLen(comptime T: type, code: ir.IRCode(T)) usize {
    return switch (code) {
        .scalar_constant, .scalar_input_index => 1,
        .vec_constant => |values| values.len,
        .vec_input => |input| input.len,
        .Op1 => |op| op.len,
        .Op2 => |op| op.len,
        .muladd => |op| op.len,
        .output => 0,
    };
}

fn outputCount(comptime T: type, comptime codes: []const ir.IRCode(T)) usize {
    var count: usize = 0;
    for (codes) |code| if (code == .output) {
        count += 1;
    };
    return count;
}

fn outputInstruction(comptime T: type, comptime codes: []const ir.IRCode(T), comptime output_index: usize) usize {
    var current: usize = 0;
    for (codes) |code| switch (code) {
        .output => |instruction| {
            if (current == output_index) return instruction;
            current += 1;
        },
        else => {},
    };
    @compileError("IR output index is out of bounds");
}

fn HRValueType(comptime T: type, comptime len: usize) type {
    return if (len == 1) Scalar(T) else Vector(len, T);
}

pub fn InlineResultType(comptime T: type, comptime codes: []const ir.IRCode(T)) type {
    const count = outputCount(T, codes);
    if (count == 0) @compileError("inlined IR must contain at least one output");
    if (count == 1) {
        const instruction = outputInstruction(T, codes, 0);
        return HRValueType(T, irValueLen(T, codes[instruction]));
    }

    var types: [count]type = undefined;
    inline for (0..count) |index| {
        const instruction = outputInstruction(T, codes, index);
        types[index] = HRValueType(T, irValueLen(T, codes[instruction]));
    }
    return @Tuple(&types);
}

fn argumentsContext(comptime T: type, arguments: anytype) ?*Context(T) {
    var result: ?*Context(T) = null;
    inline for (arguments) |argument| {
        validateBoundaryValue(T, @TypeOf(argument), "inlined IR argument");
        if (argument.context) |context| {
            if (result) |existing| {
                if (existing != context) @panic("cannot inline IR with arguments from different HR contexts");
            } else {
                result = context;
            }
        }
    }
    return result;
}

fn constantInputs(comptime T: type, comptime codes: []const ir.IRCode(T), arguments: anytype) ir.InputType(T, codes) {
    var result: ir.InputType(T, codes) = undefined;
    inline for (codes) |code| switch (code) {
        .scalar_input_index => |input_index| {
            const argument = arguments[input_index];
            result[input_index] = argument.constant_value orelse unreachable;
        },
        .vec_input => |input| {
            const argument = arguments[input.input_index];
            const values = argument.constant_values orelse unreachable;
            result[input.input_index] = values;
        },
        else => {},
    };
    return result;
}

fn constantValueFromOutput(
    comptime T: type,
    comptime codes: []const ir.IRCode(T),
    comptime output_index: usize,
    value: anytype,
) HRValueType(T, irValueLen(T, codes[outputInstruction(T, codes, output_index)])) {
    const len = irValueLen(T, codes[outputInstruction(T, codes, output_index)]);
    if (len == 1) return Scalar(T).init(value);
    const array: [len]T = value;
    return Vector(len, T).init(array);
}

fn constantInlineResult(comptime T: type, comptime codes: []const ir.IRCode(T), arguments: anytype) InlineResultType(T, codes) {
    const values = ir.evalIRCode(T, codes, constantInputs(T, codes, arguments));
    const count = outputCount(T, codes);
    if (comptime count == 1) return constantValueFromOutput(T, codes, 0, values[0]);

    var result: InlineResultType(T, codes) = undefined;
    inline for (0..count) |output_index| {
        result[output_index] = constantValueFromOutput(T, codes, output_index, values[output_index]);
    }
    return result;
}

fn remapIRCode(comptime T: type, code: ir.IRCode(T), map: []const usize) ir.IRCode(T) {
    return switch (code) {
        .Op1 => |op| .{ .Op1 = .{ .a = map[op.a], .op = op.op, .len = op.len } },
        .Op2 => |op| .{ .Op2 = .{ .lhs = map[op.lhs], .rhs = map[op.rhs], .op = op.op, .len = op.len } },
        .muladd => |op| .{ .muladd = .{ .a = map[op.a], .b = map[op.b], .c = map[op.c], .len = op.len } },
        .output, .scalar_input_index, .vec_input => unreachable,
        else => code,
    };
}

fn valueFromIRIndex(comptime T: type, comptime codes: []const ir.IRCode(T), comptime instruction: usize, context: *Context(T), map: []const usize) HRValueType(T, irValueLen(T, codes[instruction])) {
    const Value = HRValueType(T, irValueLen(T, codes[instruction]));
    return Value.fromIndex(context, map[instruction]);
}

pub fn inlineIR(
    comptime T: type,
    comptime codes: []const ir.IRCode(T),
    arguments: anytype,
) InlineResultType(T, codes) {
    const context = argumentsContext(T, arguments) orelse return constantInlineResult(T, codes, arguments);
    var map: [codes.len]usize = undefined;

    inline for (codes, 0..) |code, instruction_index| {
        map[instruction_index] = switch (code) {
            .scalar_input_index => |input_index| blk: {
                if (input_index >= arguments.len) @compileError("inlined scalar input index is out of bounds");
                const argument = arguments[input_index];
                if (comptime @TypeOf(argument).value_kind != .scalar) @compileError("inlined scalar input type mismatch");
                break :blk argument.resolve(context);
            },
            .vec_input => |input| blk: {
                if (input.input_index >= arguments.len) @compileError("inlined Vector input index is out of bounds");
                const argument = arguments[input.input_index];
                if (comptime @TypeOf(argument).value_kind != .vector or @TypeOf(argument).length != input.len) {
                    @compileError("inlined Vector input type mismatch");
                }
                break :blk argument.resolve(context);
            },
            .output => continue,
            else => context.append(remapIRCode(T, code, &map)),
        };
    }

    const count = outputCount(T, codes);
    if (comptime count == 1) {
        const instruction = outputInstruction(T, codes, 0);
        return valueFromIRIndex(T, codes, instruction, context, &map);
    }

    var result: InlineResultType(T, codes) = undefined;
    inline for (0..count) |output_index| {
        const instruction = outputInstruction(T, codes, output_index);
        result[output_index] = valueFromIRIndex(T, codes, instruction, context, &map);
    }
    return result;
}

fn functionInfo(comptime function: anytype) std.builtin.Type.Fn {
    const info = @typeInfo(@TypeOf(function));
    if (info != .@"fn" or info.@"fn".is_var_args) @compileError("expected a non-variadic HR function");
    return info.@"fn";
}

fn isHRValue(comptime Value: type) bool {
    return @typeInfo(Value) == .@"struct" and @hasDecl(Value, "hr_value");
}

fn validateBoundaryValue(comptime T: type, comptime Value: type, comptime label: []const u8) void {
    if (!isHRValue(Value)) @compileError(label ++ " must be a Scalar or Vector");
    if (Value.scalar_type != T) @compileError("all HR function inputs and outputs must use the same floating-point type");
    if (Value.value_kind == .matrix) @compileError(label ++ " cannot be a Matrix");
}

fn firstOutputType(comptime Return: type) type {
    if (isHRValue(Return)) return Return;
    const info = @typeInfo(Return);
    if (info != .@"struct" or !info.@"struct".is_tuple) {
        @compileError("HR function output must be a Scalar, Vector, or a Tuple of them");
    }
    if (info.@"struct".fields.len == 0) @compileError("HR function output Tuple cannot be empty");
    return info.@"struct".fields[0].type;
}

fn validateReturnType(comptime T: type, comptime Return: type) void {
    if (isHRValue(Return)) {
        validateBoundaryValue(T, Return, "HR function output");
        return;
    }
    const info = @typeInfo(Return);
    if (info != .@"struct" or !info.@"struct".is_tuple or info.@"struct".fields.len == 0) {
        @compileError("HR function output must be a non-empty Tuple of Scalar and Vector values");
    }
    inline for (info.@"struct".fields) |field| {
        validateBoundaryValue(T, field.type, "HR function output Tuple field");
    }
}

fn functionScalarType(comptime function: anytype) type {
    const info = functionInfo(function);
    const Return = info.return_type orelse @compileError("HR function requires an explicit return type");
    const T = if (info.params.len > 0) blk: {
        const Param = info.params[0].type orelse @compileError("HR function parameters cannot be anytype");
        if (!isHRValue(Param)) @compileError("HR function input must be a Scalar or Vector");
        break :blk Param.scalar_type;
    } else blk: {
        const Output = firstOutputType(Return);
        if (!isHRValue(Output)) @compileError("HR function output must contain Scalar or Vector values");
        break :blk Output.scalar_type;
    };
    inline for (info.params) |param| {
        validateBoundaryValue(T, param.type orelse @compileError("HR function parameters cannot be anytype"), "HR function input");
    }
    validateReturnType(T, Return);
    return T;
}

fn FunctionStorage(comptime function: anytype) type {
    const info = functionInfo(function);
    const T = functionScalarType(function);
    var types: [info.params.len]type = undefined;
    inline for (info.params, 0..) |param, index| {
        const Param = param.type orelse @compileError("HR function parameters cannot be anytype");
        validateBoundaryValue(T, Param, "HR function input");
        types[index] = Param;
    }
    return @Tuple(&types);
}

fn runFunction(comptime function: anytype, context: anytype, storage: *FunctionStorage(function)) functionInfo(function).return_type.? {
    var args: std.meta.ArgsTuple(@TypeOf(function)) = undefined;
    inline for (functionInfo(function).params, 0..) |param, index| {
        const Param = param.type.?;
        storage[index] = Param.parameter(context, index);
        args[index] = storage[index];
    }
    return @call(.auto, function, args);
}

fn appendOutputs(comptime T: type, context: *Context(T), result: anytype) void {
    const Return = @TypeOf(result);
    if (comptime isHRValue(Return)) {
        _ = context.append(.{ .output = result.resolve(context) });
        return;
    }
    inline for (@typeInfo(Return).@"struct".fields) |field| {
        _ = context.append(.{ .output = @field(result, field.name).resolve(context) });
    }
}

fn rawIRCodeLen(comptime function: anytype) usize {
    @setEvalBranchQuota(1_000_000);
    const T = functionScalarType(function);
    var empty: [0]ir.IRCode(T) = .{};
    var context = Context(T).init(&empty, false);
    var storage: FunctionStorage(function) = undefined;
    const result = runFunction(function, &context, &storage);
    appendOutputs(T, &context, result);
    return context.len();
}

fn FunctionIR(comptime function: anytype) type {
    return struct {
        const T = functionScalarType(function);
        // Materialize once; type queries and callers share the same IR.
        const generated = blk: {
            @setEvalBranchQuota(1_000_000);
            var buffer: [rawIRCodeLen(function)]ir.IRCode(T) = undefined;
            var context = Context(T).init(&buffer, true);
            var storage: FunctionStorage(function) = undefined;
            const result = runFunction(function, &context, &storage);
            appendOutputs(T, &context, result);
            break :blk buffer[0..context.len()].*;
        };
        const codes = simplify.simplifyIR(T, &generated);
    };
}

pub fn toIRCode(comptime function: anytype) @TypeOf(FunctionIR(function).codes) {
    return FunctionIR(function).codes;
}

test "Scalar and Vector HR lower to executable IR" {
    const S = Scalar(f32);
    const V = Vector(3, f32);
    const model = struct {
        fn call(scale: S, input: V) V {
            const bias = V.init(.{ 1, 2, 3 });
            return scale.mul(input).add(bias).sin();
        }
    }.call;
    const codes = comptime toIRCode(model);
    const result = ir.evalIRCode(f32, &codes, .{ @as(f32, 2), @Vector(3, f32){ 1, 2, 3 } });
    const expected = @sin(@Vector(3, f32){ 3, 6, 9 });
    try std.testing.expectEqual(expected, result[0]);
}

test "Vector get set and mulAdd lower to IR" {
    const S = Scalar(f64);
    const V = Vector(3, f64);
    const model = struct {
        fn call(input: V, replacement: S) V {
            const updated = input.set(1, replacement);
            return updated.mulAdd(S.init(2), input.get(0));
        }
    }.call;
    const codes = comptime toIRCode(model);
    const result = ir.evalIRCode(f64, &codes, .{ @Vector(3, f64){ 1, 2, 3 }, 9 });
    try std.testing.expectEqual(@Vector(3, f64){ 3, 19, 7 }, result[0]);

    const folded = comptime V.init(.{ 1, 2, 3 }).mulAdd(S.init(2), V.init(.{ 4, 5, 6 }));
    try std.testing.expectEqual([_]f64{ 6, 9, 12 }, folded.constant_values.?);
}

test "Matrix vector quadratic function lowers without Matrix IR values" {
    const S = Scalar(f64);
    const V = Vector(2, f64);
    const M = Matrix(2, 2, f64);
    const model = struct {
        fn call(x: V) S {
            const q = M.init(.{
                .{ 1, 0 },
                .{ 0, 2 },
            });
            const p = V.init(.{ 1, 0.5 });
            return q.mul(x).dot(x).add(p.dot(x));
        }
    }.call;
    const codes = comptime toIRCode(model);
    const result = ir.evalIRCode(f64, &codes, .{@Vector(2, f64){ 1, 2 }});
    try std.testing.expectEqual(@as(f64, 11), result[0]);
}

test "multiple Scalar and Vector inputs can return a mixed Tuple" {
    const S = Scalar(f32);
    const V = Vector(3, f32);
    const Outputs = @Tuple(&.{ S, V, S });
    const model = struct {
        fn call(scale: S, input: V, offset: S) Outputs {
            const scaled = scale.mul(input);
            return .{
                scaled.sum(),
                scaled,
                offset.add(scale),
            };
        }
    }.call;

    const codes = comptime toIRCode(model);
    const Output = ir.OutputType(f32, &codes);
    const ExpectedOutput = @Tuple(&.{ f32, @Vector(3, f32), f32 });
    if (Output != ExpectedOutput) @compileError("mixed HR output Tuple lowered to an unexpected IR output type");

    const result = ir.evalIRCode(f32, &codes, .{
        @as(f32, 2),
        @Vector(3, f32){ 1, 2, 3 },
        @as(f32, 5),
    });
    try std.testing.expectEqual(@as(f32, 12), result[0]);
    try std.testing.expectEqual(@Vector(3, f32){ 2, 4, 6 }, result[1]);
    try std.testing.expectEqual(@as(f32, 7), result[2]);
}

test "HR simplifies identities CSE and dead values during IR generation" {
    const S = Scalar(f64);
    const Outputs = @Tuple(&.{ S, S });
    const model = struct {
        fn call(x: S, unused: S) Outputs {
            _ = unused;
            const zero = S.init(0);
            const one = S.init(1);
            const first = x.add(zero).mul(one).neg().neg();
            const square_a = first.mul(first);
            const square_b = first.mul(first);
            return .{ square_a, square_b };
        }
    }.call;

    const codes = comptime toIRCode(model);
    const expected = [_]ir.IRCode(f64){
        .{ .scalar_input_index = 0 },
        .{ .scalar_input_index = 1 },
        .{ .Op2 = .{ .lhs = 0, .rhs = 0, .op = .mul, .len = 1 } },
        .{ .output = 2 },
        .{ .output = 2 },
    };
    try std.testing.expectEqualSlices(ir.IRCode(f64), &expected, &codes);

    const result = ir.evalIRCode(f64, &codes, .{ 3, 100 });
    try std.testing.expectEqual(@as(f64, 9), result[0]);
    try std.testing.expectEqual(@as(f64, 9), result[1]);
}

test "HR simplifies get after set and redundant set" {
    const S = Scalar(f32);
    const V = Vector(3, f32);
    const Outputs = @Tuple(&.{ S, V });
    const model = struct {
        fn call(vector: V, replacement: S) Outputs {
            const updated = vector.set(1, replacement);
            const restored = vector.set(2, vector.get(2));
            return .{ updated.get(1), restored };
        }
    }.call;

    const codes = comptime toIRCode(model);
    const result = ir.evalIRCode(f32, &codes, .{ @Vector(3, f32){ 1, 2, 3 }, @as(f32, 9) });
    try std.testing.expectEqual(@as(f32, 9), result[0]);
    try std.testing.expectEqual(@Vector(3, f32){ 1, 2, 3 }, result[1]);
}

test "HR supports f16 f32 and f64" {
    inline for (.{ f16, f32, f64 }) |T| {
        const S = Scalar(T);
        const model = struct {
            fn call(x: S) S {
                return x.mulAdd(x, S.init(1));
            }
        }.call;
        const codes = comptime toIRCode(model);
        const result = ir.evalIRCode(T, &codes, .{@as(T, 2)});
        try std.testing.expectEqual(@as(T, 5), result[0]);
    }
}
