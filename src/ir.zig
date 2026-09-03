const std = @import("std");

const Op1Code = enum {
    neg,
    sqrt,
    exp,
    log,
    sin,
    cos,
    abs,
    sum,
};

const Op2Code = enum {
    add,
    sub,
    mul,
    div,
    atan2,
};

const IRCode = union(enum) {
    scalar_constant: f64,
    scalar_input_index: usize,
    vec_constant: []const f64,
    vec_input: struct {
        input_index: usize,
        len: usize,
    },
    Op1: struct {
        a: usize,
        op: Op1Code,
        len: usize,
    },
    Op2: struct {
        lhs: usize,
        rhs: usize,
        op: Op2Code,
        len: usize,
    },
    output: usize,
};

fn InputType(comptime ir: []const IRCode) type {
    var input_count: usize = 0;
    for (ir) |code| {
        const input_index = switch (code) {
            .scalar_input_index => |index| index,
            .vec_input => |input| input.input_index,
            else => continue,
        };
        input_count = @max(input_count, input_index + 1);
    }

    var field_types: [input_count]type = undefined;
    var initialized = [_]bool{false} ** input_count;
    for (ir) |code| {
        const input = switch (code) {
            .scalar_input_index => |index| .{ .index = index, .type = f64 },
            .vec_input => |value| .{ .index = value.input_index, .type = @Vector(value.len, f64) },
            else => continue,
        };
        if (initialized[input.index]) @compileError("duplicate IR input index");
        field_types[input.index] = input.type;
        initialized[input.index] = true;
    }
    for (initialized) |is_initialized| {
        if (!is_initialized) @compileError("IR input indices must be contiguous");
    }

    return @Tuple(&field_types);
}

test "InputType" {
    const test_ir = [_]IRCode{
        IRCode{ .scalar_input_index = 0 },
        IRCode{ .vec_input = .{ .input_index = 1, .len = 3 } },
        IRCode{ .scalar_input_index = 2 },
        IRCode{ .vec_input = .{ .input_index = 3, .len = 3 } },
        IRCode{ .Op2 = .{ .lhs = 0, .rhs = 1, .op = .mul, .len = 3 } },
        IRCode{ .Op2 = .{ .lhs = 2, .rhs = 3, .op = .mul, .len = 3 } },
        IRCode{ .Op2 = .{ .lhs = 4, .rhs = 5, .op = .mul, .len = 3 } },
        IRCode{ .Op1 = .{ .a = 6, .op = .sum, .len = 1 } },
        IRCode{ .output = 7 },
        IRCode{ .output = 5 },
    };
    const ty = InputType(&test_ir);
    const expected = @Tuple(&.{ f64, @Vector(3, f64), f64, @Vector(3, f64) });
    if (ty != expected) @compileError("InputType returned an unexpected tuple type");
}

fn OutputType(comptime ir: []const IRCode) type {
    var output_count: usize = 0;
    for (ir) |code| {
        if (code == .output) output_count += 1;
    }

    var field_types: [output_count]type = undefined;
    var output_index: usize = 0;
    for (ir, 0..) |code, instruction_index| {
        const node_index = switch (code) {
            .output => |index| index,
            else => continue,
        };
        if (node_index >= instruction_index) {
            @compileError("IR output must reference an earlier instruction");
        }

        const len = switch (ir[node_index]) {
            .scalar_constant, .scalar_input_index => 1,
            .vec_constant => |value| value.len,
            .vec_input => |value| value.len,
            .Op1 => |op| op.len,
            .Op2 => |op| op.len,
            .output => @compileError("IR output cannot reference another output"),
        };
        if (len == 0) @compileError("IR output length must be greater than zero");

        field_types[output_index] = if (len == 1) f64 else @Vector(len, f64);
        output_index += 1;
    }

    return @Tuple(&field_types);
}

test "OutputType" {
    const test_ir = [_]IRCode{
        IRCode{ .scalar_input_index = 0 },
        IRCode{ .vec_input = .{ .input_index = 1, .len = 3 } },
        IRCode{ .scalar_input_index = 2 },
        IRCode{ .vec_input = .{ .input_index = 3, .len = 3 } },
        IRCode{ .Op2 = .{ .lhs = 0, .rhs = 1, .op = .mul, .len = 3 } },
        IRCode{ .Op2 = .{ .lhs = 2, .rhs = 3, .op = .mul, .len = 3 } },
        IRCode{ .Op2 = .{ .lhs = 4, .rhs = 5, .op = .mul, .len = 3 } },
        IRCode{ .Op1 = .{ .a = 6, .op = .sum, .len = 1 } },
        IRCode{ .output = 7 },
        IRCode{ .output = 5 },
    };
    const ty = OutputType(&test_ir);
    const expected = @Tuple(&.{ f64, @Vector(3, f64) });
    if (ty != expected) @compileError("OutputType returned an unexpected tuple type");
}

fn Workspace(comptime ir: []const IRCode) type {
    var value_count: usize = 0;
    for (ir) |code| {
        if (code != .output) value_count += 1;
    }

    var field_types: [value_count]type = undefined;
    var field_index: usize = 0;
    for (ir) |code| {
        const value_type = switch (code) {
            .scalar_constant, .scalar_input_index => f64,
            .vec_constant => |value| blk: {
                if (value.len == 0) @compileError("IR vector length must be greater than zero");
                break :blk @Vector(value.len, f64);
            },
            .vec_input => |value| blk: {
                if (value.len == 0) @compileError("IR vector length must be greater than zero");
                break :blk @Vector(value.len, f64);
            },
            .Op1 => |op| blk: {
                if (op.len == 0) @compileError("IR operation result length must be greater than zero");
                break :blk if (op.len == 1) f64 else @Vector(op.len, f64);
            },
            .Op2 => |op| blk: {
                if (op.len == 0) @compileError("IR operation result length must be greater than zero");
                break :blk if (op.len == 1) f64 else @Vector(op.len, f64);
            },
            .output => continue,
        };
        field_types[field_index] = value_type;
        field_index += 1;
    }

    return @Tuple(&field_types);
}

test "Workspace" {
    const test_ir = [_]IRCode{
        IRCode{ .scalar_input_index = 0 },
        IRCode{ .vec_input = .{ .input_index = 1, .len = 3 } },
        IRCode{ .scalar_input_index = 2 },
        IRCode{ .vec_input = .{ .input_index = 3, .len = 3 } },
        IRCode{ .Op2 = .{ .lhs = 0, .rhs = 1, .op = .mul, .len = 3 } },
        IRCode{ .Op2 = .{ .lhs = 2, .rhs = 3, .op = .mul, .len = 3 } },
        IRCode{ .Op2 = .{ .lhs = 4, .rhs = 5, .op = .mul, .len = 3 } },
        IRCode{ .Op1 = .{ .a = 6, .op = .sum, .len = 1 } },
        IRCode{ .output = 7 },
        IRCode{ .output = 5 },
    };
    const ty = Workspace(&test_ir);
    const expected = @Tuple(&.{
        f64,
        @Vector(3, f64),
        f64,
        @Vector(3, f64),
        @Vector(3, f64),
        @Vector(3, f64),
        @Vector(3, f64),
        f64,
    });
    if (ty != expected) @compileError("workspace returned an unexpected tuple type");
}

fn workspaceIndex(comptime ir: []const IRCode, comptime instruction_index: usize) usize {
    if (instruction_index >= ir.len) @compileError("IR instruction index is out of bounds");
    if (ir[instruction_index] == .output) @compileError("output instructions do not have workspace values");

    var index: usize = 0;
    for (ir[0..instruction_index]) |code| {
        if (code != .output) index += 1;
    }
    return index;
}

fn evalOp1(comptime Result: type, comptime op: Op1Code, operand: anytype) Result {
    return switch (op) {
        .neg => -operand,
        .sqrt => @sqrt(operand),
        .sin => @sin(operand),
        .cos => @cos(operand),
        .log => @log(operand),
        .exp => @exp(operand),
        .abs => @abs(operand),
        .sum => if (@TypeOf(operand) == f64) operand else @reduce(.Add, operand),
    };
}

fn evalOp2(comptime Result: type, comptime op: Op2Code, lhs: anytype, rhs: anytype) Result {
    if (Result == f64) {
        if (@TypeOf(lhs) != f64 or @TypeOf(rhs) != f64) {
            @compileError("scalar operation requires scalar operands");
        }
        return switch (op) {
            .add => lhs + rhs,
            .sub => lhs - rhs,
            .mul => lhs * rhs,
            .div => lhs / rhs,
            .atan2 => std.math.atan2(lhs, rhs),
        };
    }

    const vector_lhs: Result = if (@TypeOf(lhs) == f64)
        @splat(lhs)
    else if (@TypeOf(lhs) == Result)
        lhs
    else
        @compileError("lhs type does not match operation result");
    const vector_rhs: Result = if (@TypeOf(rhs) == f64)
        @splat(rhs)
    else if (@TypeOf(rhs) == Result)
        rhs
    else
        @compileError("rhs type does not match operation result");
    return switch (op) {
        .add => vector_lhs + vector_rhs,
        .sub => vector_lhs - vector_rhs,
        .mul => vector_lhs * vector_rhs,
        .div => vector_lhs / vector_rhs,
        .atan2 => blk: {
            var result: Result = undefined;
            inline for (0..@typeInfo(Result).vector.len) |index| {
                result[index] = std.math.atan2(vector_lhs[index], vector_rhs[index]);
            }
            break :blk result;
        },
    };
}

fn evalIRCode(comptime ir: []const IRCode, input: InputType(ir)) OutputType(ir) {
    var workspace: Workspace(ir) = undefined;
    var result: OutputType(ir) = undefined;
    comptime var output_index: usize = 0;

    inline for (ir, 0..) |code, instruction_index| {
        switch (code) {
            .scalar_constant => |value| {
                workspace[comptime workspaceIndex(ir, instruction_index)] = value;
            },
            .scalar_input_index => |index| {
                workspace[comptime workspaceIndex(ir, instruction_index)] = input[comptime index];
            },
            .vec_constant => |value| {
                const destination = comptime workspaceIndex(ir, instruction_index);
                const array: [value.len]f64 = value[0..value.len].*;
                const vector: @Vector(value.len, f64) = array;
                workspace[destination] = vector;
            },
            .vec_input => |value| {
                workspace[comptime workspaceIndex(ir, instruction_index)] = input[comptime value.input_index];
            },
            .Op1 => |op| {
                if (op.a >= instruction_index) @compileError("Op1 operand must reference an earlier instruction");
                const destination = comptime workspaceIndex(ir, instruction_index);
                const operand = comptime workspaceIndex(ir, op.a);
                workspace[destination] = evalOp1(@TypeOf(workspace[destination]), op.op, workspace[operand]);
            },
            .Op2 => |op| {
                if (op.lhs >= instruction_index or op.rhs >= instruction_index) {
                    @compileError("Op2 operands must reference earlier instructions");
                }
                const destination = comptime workspaceIndex(ir, instruction_index);
                const lhs = comptime workspaceIndex(ir, op.lhs);
                const rhs = comptime workspaceIndex(ir, op.rhs);
                workspace[destination] = evalOp2(
                    @TypeOf(workspace[destination]),
                    op.op,
                    workspace[lhs],
                    workspace[rhs],
                );
            },
            .output => |index| {
                if (index >= instruction_index) @compileError("output must reference an earlier instruction");
                result[output_index] = workspace[comptime workspaceIndex(ir, index)];
                output_index += 1;
            },
        }
    }
    return result;
}

test "evalIRCode" {
    const test_ir = [_]IRCode{
        IRCode{ .scalar_input_index = 0 },
        IRCode{ .vec_input = .{ .input_index = 1, .len = 3 } },
        IRCode{ .scalar_input_index = 2 },
        IRCode{ .vec_input = .{ .input_index = 3, .len = 3 } },
        IRCode{ .Op2 = .{ .lhs = 0, .rhs = 1, .op = .mul, .len = 3 } },
        IRCode{ .Op2 = .{ .lhs = 2, .rhs = 3, .op = .mul, .len = 3 } },
        IRCode{ .Op2 = .{ .lhs = 4, .rhs = 5, .op = .mul, .len = 3 } },
        IRCode{ .Op1 = .{ .a = 6, .op = .sum, .len = 1 } },
        IRCode{ .output = 7 },
        IRCode{ .output = 5 },
    };
    const result = evalIRCode(&test_ir, .{
        0.1,
        @Vector(3, f64){ 0.1, 0.2, 0.3 },
        0.2,
        @Vector(3, f64){ 0.1, 0.2, 0.3 },
    });
    try std.testing.expectApproxEqAbs(@as(f64, 0.0028), result[0], 1e-12);
    const expected = @Vector(3, f64){ 0.02, 0.04, 0.06 };
    inline for (0..3) |index| {
        try std.testing.expectApproxEqAbs(expected[index], result[1][index], 1e-12);
    }
}

test "evalIRCode builds vec_constant from a constant slice" {
    const test_ir = [_]IRCode{
        IRCode{ .vec_constant = &.{ 1.0, 2.0, 3.0 } },
        IRCode{ .output = 0 },
    };

    const result = evalIRCode(&test_ir, .{});
    try std.testing.expectEqual(@Vector(3, f64){ 1.0, 2.0, 3.0 }, result[0]);
}

test "evalIRCode supports every scalar operation" {
    const test_ir = [_]IRCode{
        .{ .scalar_input_index = 0 },
        .{ .scalar_input_index = 1 },
        .{ .Op1 = .{ .a = 0, .op = .neg, .len = 1 } },
        .{ .Op1 = .{ .a = 0, .op = .sqrt, .len = 1 } },
        .{ .Op1 = .{ .a = 1, .op = .exp, .len = 1 } },
        .{ .Op1 = .{ .a = 0, .op = .log, .len = 1 } },
        .{ .Op1 = .{ .a = 1, .op = .sin, .len = 1 } },
        .{ .Op1 = .{ .a = 1, .op = .cos, .len = 1 } },
        .{ .Op1 = .{ .a = 2, .op = .abs, .len = 1 } },
        .{ .Op1 = .{ .a = 0, .op = .sum, .len = 1 } },
        .{ .Op2 = .{ .lhs = 0, .rhs = 1, .op = .add, .len = 1 } },
        .{ .Op2 = .{ .lhs = 0, .rhs = 1, .op = .sub, .len = 1 } },
        .{ .Op2 = .{ .lhs = 0, .rhs = 1, .op = .mul, .len = 1 } },
        .{ .Op2 = .{ .lhs = 0, .rhs = 1, .op = .div, .len = 1 } },
        .{ .Op2 = .{ .lhs = 0, .rhs = 1, .op = .atan2, .len = 1 } },
        .{ .output = 2 },
        .{ .output = 3 },
        .{ .output = 4 },
        .{ .output = 5 },
        .{ .output = 6 },
        .{ .output = 7 },
        .{ .output = 8 },
        .{ .output = 9 },
        .{ .output = 10 },
        .{ .output = 11 },
        .{ .output = 12 },
        .{ .output = 13 },
        .{ .output = 14 },
    };
    const result = evalIRCode(&test_ir, .{ 4.0, 2.0 });
    const expected = .{
        -4.0,
        2.0,
        @exp(2.0),
        @log(4.0),
        @sin(2.0),
        @cos(2.0),
        4.0,
        4.0,
        6.0,
        2.0,
        8.0,
        2.0,
        std.math.atan2(@as(f64, 4.0), @as(f64, 2.0)),
    };
    inline for (expected, 0..) |value, index| {
        try std.testing.expectApproxEqAbs(value, result[index], 1e-12);
    }
}

test "evalIRCode supports vector math and atan2" {
    const test_ir = [_]IRCode{
        .{ .vec_input = .{ .input_index = 0, .len = 3 } },
        .{ .vec_input = .{ .input_index = 1, .len = 3 } },
        .{ .Op1 = .{ .a = 0, .op = .neg, .len = 3 } },
        .{ .Op1 = .{ .a = 0, .op = .sqrt, .len = 3 } },
        .{ .Op1 = .{ .a = 1, .op = .exp, .len = 3 } },
        .{ .Op1 = .{ .a = 0, .op = .log, .len = 3 } },
        .{ .Op1 = .{ .a = 0, .op = .sin, .len = 3 } },
        .{ .Op1 = .{ .a = 0, .op = .cos, .len = 3 } },
        .{ .Op1 = .{ .a = 2, .op = .abs, .len = 3 } },
        .{ .Op2 = .{ .lhs = 0, .rhs = 1, .op = .atan2, .len = 3 } },
        .{ .Op2 = .{ .lhs = 0, .rhs = 1, .op = .div, .len = 3 } },
        .{ .Op1 = .{ .a = 0, .op = .sum, .len = 1 } },
        .{ .output = 2 },
        .{ .output = 3 },
        .{ .output = 4 },
        .{ .output = 5 },
        .{ .output = 6 },
        .{ .output = 7 },
        .{ .output = 8 },
        .{ .output = 9 },
        .{ .output = 10 },
        .{ .output = 11 },
    };
    const lhs = @Vector(3, f64){ 1.0, 4.0, 9.0 };
    const rhs = @Vector(3, f64){ 2.0, 2.0, 2.0 };
    const result = evalIRCode(&test_ir, .{ lhs, rhs });
    const expected_vectors = .{
        -lhs,
        @sqrt(lhs),
        @exp(rhs),
        @log(lhs),
        @sin(lhs),
        @cos(lhs),
        @abs(-lhs),
        @Vector(3, f64){
            std.math.atan2(lhs[0], rhs[0]),
            std.math.atan2(lhs[1], rhs[1]),
            std.math.atan2(lhs[2], rhs[2]),
        },
        lhs / rhs,
    };
    inline for (expected_vectors, 0..) |expected, output_index| {
        inline for (0..3) |lane| {
            try std.testing.expectApproxEqAbs(expected[lane], result[output_index][lane], 1e-12);
        }
    }
    try std.testing.expectEqual(@as(f64, 14.0), result[9]);
}

fn randomNext(state: *u64) u64 {
    state.* = state.* *% 6364136223846793005 +% 1442695040888963407;
    return state.*;
}

fn RandomIRCase(comptime vec_len: usize, comptime operation_count: usize) type {
    return struct {
        ir: [2 + vec_len + 1 + operation_count + 2]IRCode,
        scalar_input: f64,
        vector_input: @Vector(vec_len, f64),
        scalar_output: f64,
        vector_output: @Vector(vec_len, f64),
    };
}

fn generateRandomIR(
    comptime vec_len: usize,
    comptime operation_count: usize,
    comptime seed: u64,
) RandomIRCase(vec_len, operation_count) {
    if (vec_len == 0) @compileError("random IR vector length must be greater than zero");

    const V = @Vector(vec_len, f64);
    const Value = union(enum) {
        scalar: f64,
        vector: V,
    };
    const value_count = 2 + vec_len + 1 + operation_count;

    var state = seed;
    var ir: [value_count + 2]IRCode = undefined;
    var values: [value_count]Value = undefined;
    var scalar_indices: [value_count]usize = undefined;
    var vector_indices: [value_count]usize = undefined;
    var scalar_count: usize = 0;
    var vector_count: usize = 0;
    var len: usize = 0;

    const scalar_input = 0.125;
    var vector_input: V = undefined;
    inline for (0..vec_len) |lane| {
        vector_input[lane] = @as(f64, @floatFromInt(lane + 1)) * 0.1;
    }

    ir[len] = .{ .scalar_input_index = 0 };
    values[len] = .{ .scalar = scalar_input };
    scalar_indices[scalar_count] = len;
    scalar_count += 1;
    len += 1;

    ir[len] = .{ .vec_input = .{ .input_index = 1, .len = vec_len } };
    values[len] = .{ .vector = vector_input };
    vector_indices[vector_count] = len;
    vector_count += 1;
    len += 1;

    const constant_start = len;
    inline for (0..vec_len) |_| {
        const value = @as(f64, @floatFromInt(randomNext(&state) % 9 + 1)) * 0.1;
        ir[len] = .{ .scalar_constant = value };
        values[len] = .{ .scalar = value };
        scalar_indices[scalar_count] = len;
        scalar_count += 1;
        len += 1;
    }

    const constant_vector = blk: {
        var result: [vec_len]f64 = undefined;
        inline for (0..vec_len) |lane| {
            result[lane] = values[constant_start + lane].scalar;
        }
        break :blk result;
    };
    ir[len] = .{ .vec_constant = &constant_vector };
    values[len] = .{ .vector = constant_vector };
    vector_indices[vector_count] = len;
    vector_count += 1;
    len += 1;

    inline for (0..operation_count) |_| {
        const operation = randomNext(&state) % 10;
        switch (operation) {
            0, 1 => {
                const lhs = scalar_indices[randomNext(&state) % scalar_count];
                const rhs = scalar_indices[randomNext(&state) % scalar_count];
                const op: Op2Code = if (operation == 0) .add else .mul;
                ir[len] = .{ .Op2 = .{ .lhs = lhs, .rhs = rhs, .op = op, .len = 1 } };
                values[len] = .{ .scalar = switch (op) {
                    .add => values[lhs].scalar + values[rhs].scalar,
                    .mul => values[lhs].scalar * values[rhs].scalar,
                    else => unreachable,
                } };
                scalar_indices[scalar_count] = len;
                scalar_count += 1;
            },
            2, 3, 4 => {
                const lhs = vector_indices[randomNext(&state) % vector_count];
                const rhs = vector_indices[randomNext(&state) % vector_count];
                const op: Op2Code = switch (operation) {
                    2 => .add,
                    3 => .sub,
                    4 => .mul,
                    else => unreachable,
                };
                ir[len] = .{ .Op2 = .{ .lhs = lhs, .rhs = rhs, .op = op, .len = vec_len } };
                values[len] = .{ .vector = switch (op) {
                    .add => values[lhs].vector + values[rhs].vector,
                    .sub => values[lhs].vector - values[rhs].vector,
                    .mul => values[lhs].vector * values[rhs].vector,
                    else => unreachable,
                } };
                vector_indices[vector_count] = len;
                vector_count += 1;
            },
            5 => {
                const scalar = scalar_indices[randomNext(&state) % scalar_count];
                const vector = vector_indices[randomNext(&state) % vector_count];
                const scalar_first = randomNext(&state) & 1 == 0;
                ir[len] = .{ .Op2 = .{
                    .lhs = if (scalar_first) scalar else vector,
                    .rhs = if (scalar_first) vector else scalar,
                    .op = .mul,
                    .len = vec_len,
                } };
                values[len] = .{ .vector = @as(V, @splat(values[scalar].scalar)) * values[vector].vector };
                vector_indices[vector_count] = len;
                vector_count += 1;
            },
            6 => {
                const lhs = scalar_indices[randomNext(&state) % scalar_count];
                const rhs = scalar_indices[randomNext(&state) % scalar_count];
                ir[len] = .{ .Op2 = .{ .lhs = lhs, .rhs = rhs, .op = .atan2, .len = 1 } };
                values[len] = .{ .scalar = std.math.atan2(values[lhs].scalar, values[rhs].scalar) };
                scalar_indices[scalar_count] = len;
                scalar_count += 1;
            },
            7, 8 => {
                const source = vector_indices[randomNext(&state) % vector_count];
                const op: Op1Code = if (operation == 7) .sin else .cos;
                ir[len] = .{ .Op1 = .{ .a = source, .op = op, .len = vec_len } };
                values[len] = .{ .vector = if (op == .sin) @sin(values[source].vector) else @cos(values[source].vector) };
                vector_indices[vector_count] = len;
                vector_count += 1;
            },
            9 => {
                const source = vector_indices[randomNext(&state) % vector_count];
                ir[len] = .{ .Op1 = .{ .a = source, .op = .sum, .len = 1 } };
                values[len] = .{ .scalar = @reduce(.Add, values[source].vector) };
                scalar_indices[scalar_count] = len;
                scalar_count += 1;
            },
            else => unreachable,
        }
        len += 1;
    }

    const scalar_output = scalar_indices[randomNext(&state) % scalar_count];
    const vector_output = vector_indices[randomNext(&state) % vector_count];
    ir[len] = .{ .output = scalar_output };
    len += 1;
    ir[len] = .{ .output = vector_output };

    return .{
        .ir = ir,
        .scalar_input = scalar_input,
        .vector_input = vector_input,
        .scalar_output = values[scalar_output].scalar,
        .vector_output = values[vector_output].vector,
    };
}

test "random legal IR combinations match progressively generated references" {
    @setEvalBranchQuota(1_000_000);
    inline for ([_]usize{ 2, 3, 4, 8 }) |vec_len| {
        inline for (0..16) |case_index| {
            const generated = comptime generateRandomIR(
                vec_len,
                12,
                0x9e3779b97f4a7c15 +% case_index *% 0x100000001b3,
            );
            const result = evalIRCode(&generated.ir, .{
                generated.scalar_input,
                generated.vector_input,
            });
            try std.testing.expectApproxEqAbs(generated.scalar_output, result[0], 1e-10);
            inline for (0..vec_len) |lane| {
                try std.testing.expectApproxEqAbs(generated.vector_output[lane], result[1][lane], 1e-10);
            }
        }
    }
}
