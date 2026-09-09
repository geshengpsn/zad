const std = @import("std");

pub const Op1Code = union(enum) {
    neg,
    sqrt,
    exp,
    log,
    sin,
    cos,
    abs,
    sum,
    get: usize,
};

pub const Op2Code = union(enum) {
    add,
    sub,
    mul,
    div,
    atan2,
    set: usize,
};

fn validateFloatType(comptime T: type) void {
    switch (T) {
        f16, f32, f64 => {},
        else => @compileError("IRCode only supports f16, f32, and f64"),
    }
}

pub fn IRCode(comptime T: type) type {
    validateFloatType(T);
    return union(enum) {
        scalar_constant: T,
        scalar_input_index: usize,
        vec_constant: []const T,
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
        muladd: struct {
            a: usize,
            b: usize,
            c: usize,
            len: usize,
        },
        output: usize,
    };
}

pub fn InputType(comptime T: type, comptime ir: []const IRCode(T)) type {
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
            .scalar_input_index => |index| .{ .index = index, .type = T },
            .vec_input => |value| .{ .index = value.input_index, .type = @Vector(value.len, T) },
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
    const test_ir = [_]IRCode(f64){
        IRCode(f64){ .scalar_input_index = 0 },
        IRCode(f64){ .vec_input = .{ .input_index = 1, .len = 3 } },
        IRCode(f64){ .scalar_input_index = 2 },
        IRCode(f64){ .vec_input = .{ .input_index = 3, .len = 3 } },
        IRCode(f64){ .Op2 = .{ .lhs = 0, .rhs = 1, .op = .mul, .len = 3 } },
        IRCode(f64){ .Op2 = .{ .lhs = 2, .rhs = 3, .op = .mul, .len = 3 } },
        IRCode(f64){ .Op2 = .{ .lhs = 4, .rhs = 5, .op = .mul, .len = 3 } },
        IRCode(f64){ .Op1 = .{ .a = 6, .op = .sum, .len = 1 } },
        IRCode(f64){ .output = 7 },
        IRCode(f64){ .output = 5 },
    };
    const ty = InputType(f64, &test_ir);
    const expected = @Tuple(&.{ f64, @Vector(3, f64), f64, @Vector(3, f64) });
    if (ty != expected) @compileError("InputType returned an unexpected tuple type");
}

pub fn OutputType(comptime T: type, comptime ir: []const IRCode(T)) type {
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
            .muladd => |op| op.len,
            .output => @compileError("IR output cannot reference another output"),
        };
        if (len == 0) @compileError("IR output length must be greater than zero");

        field_types[output_index] = if (len == 1) T else @Vector(len, T);
        output_index += 1;
    }

    return @Tuple(&field_types);
}

test "OutputType" {
    const test_ir = [_]IRCode(f64){
        .{ .scalar_input_index = 0 },
        .{ .vec_input = .{ .input_index = 1, .len = 3 } },
        .{ .scalar_input_index = 2 },
        .{ .vec_input = .{ .input_index = 3, .len = 3 } },
        .{ .Op2 = .{ .lhs = 0, .rhs = 1, .op = .mul, .len = 3 } },
        .{ .Op2 = .{ .lhs = 2, .rhs = 3, .op = .mul, .len = 3 } },
        .{ .Op2 = .{ .lhs = 4, .rhs = 5, .op = .mul, .len = 3 } },
        .{ .Op1 = .{ .a = 6, .op = .sum, .len = 1 } },
        .{ .output = 7 },
        .{ .output = 5 },
    };
    const ty = OutputType(f64, &test_ir);
    const expected = @Tuple(&.{ f64, @Vector(3, f64) });
    if (ty != expected) @compileError("OutputType returned an unexpected tuple type");
}

pub fn Workspace(comptime T: type, comptime ir: []const IRCode(T)) type {
    var value_count: usize = 0;
    for (ir) |code| {
        if (code != .output) value_count += 1;
    }

    var field_types: [value_count]type = undefined;
    var field_index: usize = 0;
    for (ir) |code| {
        const value_type = switch (code) {
            .scalar_constant, .scalar_input_index => T,
            .vec_constant => |value| blk: {
                if (value.len == 0) @compileError("IR vector length must be greater than zero");
                break :blk @Vector(value.len, T);
            },
            .vec_input => |value| blk: {
                if (value.len == 0) @compileError("IR vector length must be greater than zero");
                break :blk @Vector(value.len, T);
            },
            .Op1 => |op| blk: {
                if (op.len == 0) @compileError("IR operation result length must be greater than zero");
                break :blk if (op.len == 1) T else @Vector(op.len, T);
            },
            .Op2 => |op| blk: {
                if (op.len == 0) @compileError("IR operation result length must be greater than zero");
                break :blk if (op.len == 1) T else @Vector(op.len, T);
            },
            .muladd => |op| blk: {
                if (op.len == 0) @compileError("IR operation result length must be greater than zero");
                break :blk if (op.len == 1) T else @Vector(op.len, T);
            },
            .output => continue,
        };
        field_types[field_index] = value_type;
        field_index += 1;
    }

    return @Tuple(&field_types);
}

test "Workspace" {
    const test_ir = [_]IRCode(f64){
        .{ .scalar_input_index = 0 },
        .{ .vec_input = .{ .input_index = 1, .len = 3 } },
        .{ .scalar_input_index = 2 },
        .{ .vec_input = .{ .input_index = 3, .len = 3 } },
        .{ .Op2 = .{ .lhs = 0, .rhs = 1, .op = .mul, .len = 3 } },
        .{ .Op2 = .{ .lhs = 2, .rhs = 3, .op = .mul, .len = 3 } },
        .{ .Op2 = .{ .lhs = 4, .rhs = 5, .op = .mul, .len = 3 } },
        .{ .Op1 = .{ .a = 6, .op = .sum, .len = 1 } },
        .{ .output = 7 },
        .{ .output = 5 },
    };
    const ty = Workspace(f64, &test_ir);
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

fn workspaceIndex(comptime T: type, comptime ir: []const IRCode(T), comptime instruction_index: usize) usize {
    if (instruction_index >= ir.len) @compileError("IR instruction index is out of bounds");
    if (ir[instruction_index] == .output) @compileError("output instructions do not have workspace values");

    var index: usize = 0;
    for (ir[0..instruction_index]) |code| {
        if (code != .output) index += 1;
    }
    return index;
}

fn evalOp1(comptime T: type, comptime Result: type, comptime op: Op1Code, operand: anytype) Result {
    return switch (op) {
        .neg => -operand,
        .sqrt => @sqrt(operand),
        .sin => @sin(operand),
        .cos => @cos(operand),
        .log => @log(operand),
        .exp => @exp(operand),
        .abs => @abs(operand),
        .sum => if (@TypeOf(operand) == T) operand else @reduce(.Add, operand),
        .get => |index| blk: {
            if (Result != T) @compileError("get must produce a scalar");
            const operand_info = @typeInfo(@TypeOf(operand));
            if (operand_info != .vector) @compileError("get requires a vector operand");
            if (index >= operand_info.vector.len) @compileError("get index is out of bounds");
            break :blk operand[comptime index];
        },
    };
}

fn atan2(comptime T: type, y: T, x: T) T {
    return switch (T) {
        f16 => @floatCast(std.math.atan2(@as(f32, y), @as(f32, x))),
        f32, f64 => std.math.atan2(y, x),
        else => unreachable,
    };
}

fn evalOp2(comptime T: type, comptime Result: type, comptime op: Op2Code, lhs: anytype, rhs: anytype) Result {
    switch (op) {
        .set => |index| {
            if (Result == T) @compileError("set must produce a vector");
            if (@TypeOf(lhs) != Result) @compileError("set lhs must match the result vector type");
            if (@TypeOf(rhs) != T) @compileError("set rhs must be a scalar");
            if (index >= @typeInfo(Result).vector.len) @compileError("set index is out of bounds");
            var result = lhs;
            result[comptime index] = rhs;
            return result;
        },
        else => {},
    }

    if (Result == T) {
        if (@TypeOf(lhs) != T or @TypeOf(rhs) != T) {
            @compileError("scalar operation requires scalar operands");
        }
        return switch (op) {
            .add => lhs + rhs,
            .sub => lhs - rhs,
            .mul => lhs * rhs,
            .div => lhs / rhs,
            .atan2 => atan2(T, lhs, rhs),
            .set => unreachable,
        };
    }

    const vector_lhs: Result = if (@TypeOf(lhs) == T)
        @splat(lhs)
    else if (@TypeOf(lhs) == Result)
        lhs
    else
        @compileError("lhs type does not match operation result");
    const vector_rhs: Result = if (@TypeOf(rhs) == T)
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
                result[index] = atan2(T, vector_lhs[index], vector_rhs[index]);
            }
            break :blk result;
        },
        .set => unreachable,
    };
}

fn evalMulAdd(comptime T: type, comptime Result: type, a: anytype, b: anytype, c: anytype) Result {
    if (Result == T) {
        if (@TypeOf(a) != T or @TypeOf(b) != T or @TypeOf(c) != T) {
            @compileError("scalar muladd requires scalar operands");
        }
        return @mulAdd(T, a, b, c);
    }

    const vector_a: Result = if (@TypeOf(a) == T)
        @splat(a)
    else if (@TypeOf(a) == Result)
        a
    else
        @compileError("muladd operand a does not match result type");
    const vector_b: Result = if (@TypeOf(b) == T)
        @splat(b)
    else if (@TypeOf(b) == Result)
        b
    else
        @compileError("muladd operand b does not match result type");
    const vector_c: Result = if (@TypeOf(c) == T)
        @splat(c)
    else if (@TypeOf(c) == Result)
        c
    else
        @compileError("muladd operand c does not match result type");
    return @mulAdd(Result, vector_a, vector_b, vector_c);
}

pub fn evalIRCode(comptime T: type, comptime ir: []const IRCode(T), input: InputType(T, ir)) OutputType(T, ir) {
    // Unrolling large IRs and resolving workspace indices exceeds the default quota.
    @setEvalBranchQuota(10_000_000);
    var workspace: Workspace(T, ir) = undefined;
    var result: OutputType(T, ir) = undefined;
    comptime var output_index: usize = 0;

    inline for (ir, 0..) |code, instruction_index| {
        switch (code) {
            .scalar_constant => |value| {
                workspace[comptime workspaceIndex(T, ir, instruction_index)] = value;
            },
            .scalar_input_index => |index| {
                workspace[comptime workspaceIndex(T, ir, instruction_index)] = input[comptime index];
            },
            .vec_constant => |value| {
                const destination = comptime workspaceIndex(T, ir, instruction_index);
                const array: [value.len]T = value[0..value.len].*;
                const vector: @Vector(value.len, T) = array;
                workspace[destination] = vector;
            },
            .vec_input => |value| {
                workspace[comptime workspaceIndex(T, ir, instruction_index)] = input[comptime value.input_index];
            },
            .Op1 => |op| {
                if (op.a >= instruction_index) @compileError("Op1 operand must reference an earlier instruction");
                const destination = comptime workspaceIndex(T, ir, instruction_index);
                const operand = comptime workspaceIndex(T, ir, op.a);
                workspace[destination] = evalOp1(T, @TypeOf(workspace[destination]), op.op, workspace[operand]);
            },
            .Op2 => |op| {
                if (op.lhs >= instruction_index or op.rhs >= instruction_index) {
                    @compileError("Op2 operands must reference earlier instructions");
                }
                const destination = comptime workspaceIndex(T, ir, instruction_index);
                const lhs = comptime workspaceIndex(T, ir, op.lhs);
                const rhs = comptime workspaceIndex(T, ir, op.rhs);
                workspace[destination] = evalOp2(
                    T,
                    @TypeOf(workspace[destination]),
                    op.op,
                    workspace[lhs],
                    workspace[rhs],
                );
            },
            .muladd => |op| {
                if (op.a >= instruction_index or op.b >= instruction_index or op.c >= instruction_index) {
                    @compileError("muladd operands must reference earlier instructions");
                }
                const destination = comptime workspaceIndex(T, ir, instruction_index);
                const a = comptime workspaceIndex(T, ir, op.a);
                const b = comptime workspaceIndex(T, ir, op.b);
                const c = comptime workspaceIndex(T, ir, op.c);
                workspace[destination] = evalMulAdd(
                    T,
                    @TypeOf(workspace[destination]),
                    workspace[a],
                    workspace[b],
                    workspace[c],
                );
            },
            .output => |index| {
                if (index >= instruction_index) @compileError("output must reference an earlier instruction");
                result[output_index] = workspace[comptime workspaceIndex(T, ir, index)];
                output_index += 1;
            },
        }
    }
    return result;
}

test "evalIRCode" {
    const test_ir = [_]IRCode(f64){
        .{ .scalar_input_index = 0 },
        .{ .vec_input = .{ .input_index = 1, .len = 3 } },
        .{ .scalar_input_index = 2 },
        .{ .vec_input = .{ .input_index = 3, .len = 3 } },
        .{ .Op2 = .{ .lhs = 0, .rhs = 1, .op = .mul, .len = 3 } },
        .{ .Op2 = .{ .lhs = 2, .rhs = 3, .op = .mul, .len = 3 } },
        .{ .Op2 = .{ .lhs = 4, .rhs = 5, .op = .mul, .len = 3 } },
        .{ .Op1 = .{ .a = 6, .op = .sum, .len = 1 } },
        .{ .output = 7 },
        .{ .output = 5 },
    };
    const result = evalIRCode(f64, &test_ir, .{
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

test "evalIRCode handles long instruction chains without a caller branch quota" {
    const operation_count = 128;
    const test_ir = comptime blk: {
        var codes: [operation_count + 3]IRCode(f64) = undefined;
        codes[0] = .{ .scalar_constant = 1 };
        codes[1] = .{ .scalar_input_index = 0 };
        for (2..codes.len - 1) |index| {
            codes[index] = .{ .Op2 = .{ .lhs = index - 1, .rhs = 0, .op = .add, .len = 1 } };
        }
        codes[codes.len - 1] = .{ .output = codes.len - 2 };
        break :blk codes;
    };

    const result = evalIRCode(f64, &test_ir, .{3.0});
    try std.testing.expectEqual(@as(f64, 131), result[0]);
}

test "evalIRCode builds vec_constant from a constant slice" {
    const test_ir = [_]IRCode(f64){
        .{ .vec_constant = &.{ 1.0, 2.0, 3.0 } },
        .{ .output = 0 },
    };

    const result = evalIRCode(f64, &test_ir, .{});
    try std.testing.expectEqual(@Vector(3, f64){ 1.0, 2.0, 3.0 }, result[0]);
}

test "evalIRCode supports every scalar operation" {
    const test_ir = [_]IRCode(f64){
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
    const result = evalIRCode(f64, &test_ir, .{ 4.0, 2.0 });
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
    const test_ir = [_]IRCode(f64){
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
    const result = evalIRCode(f64, &test_ir, .{ lhs, rhs });
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

fn testSelectedFloatType(comptime T: type, comptime tolerance: T) !void {
    const constants = [_]T{ 0.5, 1.0, 1.5 };
    const test_ir = [_]IRCode(T){
        .{ .scalar_input_index = 0 },
        .{ .vec_input = .{ .input_index = 1, .len = 3 } },
        .{ .vec_constant = &constants },
        .{ .Op2 = .{ .lhs = 0, .rhs = 1, .op = .mul, .len = 3 } },
        .{ .Op2 = .{ .lhs = 3, .rhs = 2, .op = .add, .len = 3 } },
        .{ .Op1 = .{ .a = 4, .op = .sum, .len = 1 } },
        .{ .output = 5 },
        .{ .output = 4 },
    };
    const input_vector = @Vector(3, T){ 1.0, 2.0, 3.0 };
    const result = evalIRCode(T, &test_ir, .{ @as(T, 2.0), input_vector });
    try std.testing.expectApproxEqAbs(@as(T, 15.0), result[0], tolerance);
    const expected = @Vector(3, T){ 2.5, 5.0, 7.5 };
    inline for (0..3) |lane| {
        try std.testing.expectApproxEqAbs(expected[lane], result[1][lane], tolerance);
    }
}

test "IRCode supports selectable f16 f32 and f64 types" {
    try testSelectedFloatType(f16, 1e-2);
    try testSelectedFloatType(f32, 1e-6);
    try testSelectedFloatType(f64, 1e-12);
}

fn testAtan2Type(comptime T: type, comptime tolerance: T) !void {
    const test_ir = [_]IRCode(T){
        .{ .scalar_input_index = 0 },
        .{ .scalar_input_index = 1 },
        .{ .Op2 = .{ .lhs = 0, .rhs = 1, .op = .atan2, .len = 1 } },
        .{ .output = 2 },
    };
    const result = evalIRCode(T, &test_ir, .{ @as(T, 2), @as(T, 3) });
    try std.testing.expectApproxEqAbs(atan2(T, @as(T, 2), @as(T, 3)), result[0], tolerance);
}

test "atan2 supports every selectable IR float type" {
    try testAtan2Type(f16, 1e-2);
    try testAtan2Type(f32, 1e-6);
    try testAtan2Type(f64, 1e-12);
}

fn testMulAddType(comptime T: type, comptime tolerance: T) !void {
    const V = @Vector(3, T);
    const test_ir = [_]IRCode(T){
        .{ .scalar_input_index = 0 },
        .{ .vec_input = .{ .input_index = 1, .len = 3 } },
        .{ .vec_input = .{ .input_index = 2, .len = 3 } },
        .{ .muladd = .{ .a = 0, .b = 1, .c = 2, .len = 3 } },
        .{ .scalar_constant = 2 },
        .{ .scalar_constant = 3 },
        .{ .scalar_constant = 4 },
        .{ .muladd = .{ .a = 4, .b = 5, .c = 6, .len = 1 } },
        .{ .output = 3 },
        .{ .output = 7 },
    };
    const b = V{ 1, 2, 3 };
    const c = V{ 4, 5, 6 };
    const result = evalIRCode(T, &test_ir, .{ @as(T, 2), b, c });
    const expected_vector = @mulAdd(V, @as(V, @splat(@as(T, 2))), b, c);
    inline for (0..3) |lane| {
        try std.testing.expectApproxEqAbs(expected_vector[lane], result[0][lane], tolerance);
    }
    try std.testing.expectApproxEqAbs(@as(T, 10), result[1], tolerance);
}

test "muladd supports scalar vector broadcasting for every float type" {
    try testMulAddType(f16, 1e-2);
    try testMulAddType(f32, 1e-6);
    try testMulAddType(f64, 1e-12);
}

fn testGetSetType(comptime T: type) !void {
    const V = @Vector(3, T);
    const test_ir = [_]IRCode(T){
        .{ .vec_input = .{ .input_index = 0, .len = 3 } },
        .{ .scalar_input_index = 1 },
        .{ .Op1 = .{ .a = 0, .op = .{ .get = 1 }, .len = 1 } },
        .{ .Op2 = .{ .lhs = 0, .rhs = 1, .op = .{ .set = 2 }, .len = 3 } },
        .{ .output = 2 },
        .{ .output = 3 },
        .{ .output = 0 },
    };
    const source = V{ 1, 2, 3 };
    const result = evalIRCode(T, &test_ir, .{ source, @as(T, 9) });
    try std.testing.expectEqual(@as(T, 2), result[0]);
    try std.testing.expectEqual(V{ 1, 2, 9 }, result[1]);
    try std.testing.expectEqual(source, result[2]);
}

test "get and set support every float type and preserve the source vector" {
    try testGetSetType(f16);
    try testGetSetType(f32);
    try testGetSetType(f64);
}

fn randomNext(state: *u64) u64 {
    state.* = state.* *% 6364136223846793005 +% 1442695040888963407;
    return state.*;
}

fn RandomIRCase(comptime T: type, comptime vec_len: usize, comptime operation_count: usize) type {
    return struct {
        ir: [2 + vec_len + 1 + operation_count + 2]IRCode(T),
        scalar_input: T,
        vector_input: @Vector(vec_len, T),
        scalar_output: T,
        vector_output: @Vector(vec_len, T),
    };
}

fn generateRandomIR(
    comptime T: type,
    comptime vec_len: usize,
    comptime operation_count: usize,
    comptime seed: u64,
) RandomIRCase(T, vec_len, operation_count) {
    validateFloatType(T);
    if (vec_len == 0) @compileError("random IR vector length must be greater than zero");

    const V = @Vector(vec_len, T);
    const Value = union(enum) {
        scalar: T,
        vector: V,
    };
    const value_count = 2 + vec_len + 1 + operation_count;

    var state = seed;
    var ir: [value_count + 2]IRCode(T) = undefined;
    var values: [value_count]Value = undefined;
    var scalar_indices: [value_count]usize = undefined;
    var vector_indices: [value_count]usize = undefined;
    var scalar_count: usize = 0;
    var vector_count: usize = 0;
    var len: usize = 0;

    const scalar_input: T = 0.125;
    var vector_input: V = undefined;
    inline for (0..vec_len) |lane| {
        vector_input[lane] = @as(T, @floatFromInt(lane + 1)) * @as(T, 0.1);
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
        const value = @as(T, @floatFromInt(randomNext(&state) % 9 + 1)) * @as(T, 0.1);
        ir[len] = .{ .scalar_constant = value };
        values[len] = .{ .scalar = value };
        scalar_indices[scalar_count] = len;
        scalar_count += 1;
        len += 1;
    }

    const constant_vector = blk: {
        var result: [vec_len]T = undefined;
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
        const operation = randomNext(&state) % 13;
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
                values[len] = .{ .scalar = atan2(T, values[lhs].scalar, values[rhs].scalar) };
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
            10 => {
                const a = scalar_indices[randomNext(&state) % scalar_count];
                const b = scalar_indices[randomNext(&state) % scalar_count];
                const c = scalar_indices[randomNext(&state) % scalar_count];
                ir[len] = .{ .muladd = .{ .a = a, .b = b, .c = c, .len = 1 } };
                values[len] = .{ .scalar = @mulAdd(T, values[a].scalar, values[b].scalar, values[c].scalar) };
                scalar_indices[scalar_count] = len;
                scalar_count += 1;
            },
            11 => {
                const source = vector_indices[randomNext(&state) % vector_count];
                const index = randomNext(&state) % vec_len;
                ir[len] = .{ .Op1 = .{ .a = source, .op = .{ .get = index }, .len = 1 } };
                values[len] = .{ .scalar = values[source].vector[index] };
                scalar_indices[scalar_count] = len;
                scalar_count += 1;
            },
            12 => {
                const vector = vector_indices[randomNext(&state) % vector_count];
                const scalar = scalar_indices[randomNext(&state) % scalar_count];
                const index = randomNext(&state) % vec_len;
                ir[len] = .{ .Op2 = .{ .lhs = vector, .rhs = scalar, .op = .{ .set = index }, .len = vec_len } };
                var result = values[vector].vector;
                result[index] = values[scalar].scalar;
                values[len] = .{ .vector = result };
                vector_indices[vector_count] = len;
                vector_count += 1;
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
                f64,
                vec_len,
                12,
                0x9e3779b97f4a7c15 +% case_index *% 0x100000001b3,
            );
            const result = evalIRCode(f64, &generated.ir, .{
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
