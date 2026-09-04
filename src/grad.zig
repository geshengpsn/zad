const ir = @import("ir.zig");
const simplify = @import("simplify.zig");

fn codeLen(comptime T: type, code: ir.IRCode(T)) usize {
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

fn staticVector(comptime T: type, comptime len: usize, comptime value: T) []const T {
    const values: [len]T = @splat(value);
    return &struct {
        const data = values;
    }.data;
}

fn inputInstruction(comptime T: type, comptime source: []const ir.IRCode(T), comptime input_index: usize) usize {
    var result: ?usize = null;
    for (source, 0..) |code, instruction_index| switch (code) {
        .scalar_input_index => |index| if (index == input_index) {
            if (result != null) @compileError("duplicate selected IR input index");
            result = instruction_index;
        },
        .vec_input => |input| if (input.input_index == input_index) {
            if (result != null) @compileError("duplicate selected IR input index");
            result = instruction_index;
        },
        else => {},
    };
    return result orelse @compileError("selected IR input index does not exist");
}

fn outputInstruction(comptime T: type, comptime source: []const ir.IRCode(T), comptime output_index: usize) usize {
    var current: usize = 0;
    for (source) |code| switch (code) {
        .output => |instruction| {
            if (current == output_index) return instruction;
            current += 1;
        },
        else => {},
    };
    @compileError("selected IR output index does not exist");
}

fn remapCode(comptime T: type, code: ir.IRCode(T), map: []const usize) ir.IRCode(T) {
    return switch (code) {
        .Op1 => |op| .{ .Op1 = .{ .a = map[op.a], .op = op.op, .len = op.len } },
        .Op2 => |op| .{ .Op2 = .{ .lhs = map[op.lhs], .rhs = map[op.rhs], .op = op.op, .len = op.len } },
        .muladd => |op| .{ .muladd = .{ .a = map[op.a], .b = map[op.b], .c = map[op.c], .len = op.len } },
        .output => unreachable,
        else => code,
    };
}

fn appendConstant(comptime T: type, builder: anytype, comptime value: T, comptime len: usize) usize {
    return if (len == 1)
        builder.append(.{ .scalar_constant = value })
    else
        builder.append(.{ .vec_constant = staticVector(T, len, value) });
}

fn appendUnary(builder: anytype, input: usize, op: ir.Op1Code, len: usize) usize {
    return builder.append(.{ .Op1 = .{ .a = input, .op = op, .len = len } });
}

fn appendBinary(builder: anytype, lhs: usize, rhs: usize, op: ir.Op2Code, len: usize) usize {
    return builder.append(.{ .Op2 = .{ .lhs = lhs, .rhs = rhs, .op = op, .len = len } });
}

fn projectAdjoint(builder: anytype, contribution: usize, source_len: usize, target_len: usize) usize {
    if (source_len == target_len) return contribution;
    if (target_len == 1 and source_len > 1) return appendUnary(builder, contribution, .sum, 1);
    @compileError("cannot project adjoint to target shape");
}

fn addAdjoint(
    builder: anytype,
    adjoints: []?usize,
    target: usize,
    contribution: usize,
    source_len: usize,
    target_len: usize,
) void {
    const projected = projectAdjoint(builder, contribution, source_len, target_len);
    adjoints[target] = if (adjoints[target]) |existing|
        appendBinary(builder, existing, projected, .add, target_len)
    else
        projected;
}

fn reverseInstruction(
    comptime T: type,
    comptime source: []const ir.IRCode(T),
    builder: anytype,
    map: []const usize,
    adjoints: []?usize,
    instruction_index: usize,
) void {
    const adjoint = adjoints[instruction_index] orelse return;
    const code = source[instruction_index];
    const result_len = codeLen(T, code);
    switch (code) {
        .scalar_constant, .scalar_input_index, .vec_constant, .vec_input => {},
        .Op1 => |operation| switch (operation.op) {
            .neg => addAdjoint(builder, adjoints, operation.a, appendUnary(builder, adjoint, .neg, result_len), result_len, codeLen(T, source[operation.a])),
            .sqrt => {
                const two = appendConstant(T, builder, 2, result_len);
                const denominator = appendBinary(builder, two, map[instruction_index], .mul, result_len);
                addAdjoint(builder, adjoints, operation.a, appendBinary(builder, adjoint, denominator, .div, result_len), result_len, codeLen(T, source[operation.a]));
            },
            .exp => addAdjoint(builder, adjoints, operation.a, appendBinary(builder, adjoint, map[instruction_index], .mul, result_len), result_len, codeLen(T, source[operation.a])),
            .log => addAdjoint(builder, adjoints, operation.a, appendBinary(builder, adjoint, map[operation.a], .div, result_len), result_len, codeLen(T, source[operation.a])),
            .sin => {
                const cosine = appendUnary(builder, map[operation.a], .cos, result_len);
                addAdjoint(builder, adjoints, operation.a, appendBinary(builder, adjoint, cosine, .mul, result_len), result_len, codeLen(T, source[operation.a]));
            },
            .cos => {
                const sine = appendUnary(builder, map[operation.a], .sin, result_len);
                const contribution = appendUnary(builder, appendBinary(builder, adjoint, sine, .mul, result_len), .neg, result_len);
                addAdjoint(builder, adjoints, operation.a, contribution, result_len, codeLen(T, source[operation.a]));
            },
            .abs => {
                const local = appendBinary(builder, map[operation.a], map[instruction_index], .div, result_len);
                addAdjoint(builder, adjoints, operation.a, appendBinary(builder, adjoint, local, .mul, result_len), result_len, codeLen(T, source[operation.a]));
            },
            .sum => {
                const input_len = codeLen(T, source[operation.a]);
                const ones = appendConstant(T, builder, 1, input_len);
                addAdjoint(builder, adjoints, operation.a, appendBinary(builder, adjoint, ones, .mul, input_len), input_len, input_len);
            },
            .get => |element_index| {
                const input_len = codeLen(T, source[operation.a]);
                const zeros = appendConstant(T, builder, 0, input_len);
                const contribution = appendBinary(builder, zeros, adjoint, .{ .set = element_index }, input_len);
                addAdjoint(builder, adjoints, operation.a, contribution, input_len, input_len);
            },
        },
        .Op2 => |operation| {
            const lhs_len = codeLen(T, source[operation.lhs]);
            const rhs_len = codeLen(T, source[operation.rhs]);
            switch (operation.op) {
                .add => {
                    addAdjoint(builder, adjoints, operation.lhs, adjoint, result_len, lhs_len);
                    addAdjoint(builder, adjoints, operation.rhs, adjoint, result_len, rhs_len);
                },
                .sub => {
                    addAdjoint(builder, adjoints, operation.lhs, adjoint, result_len, lhs_len);
                    addAdjoint(builder, adjoints, operation.rhs, appendUnary(builder, adjoint, .neg, result_len), result_len, rhs_len);
                },
                .mul => {
                    addAdjoint(builder, adjoints, operation.lhs, appendBinary(builder, adjoint, map[operation.rhs], .mul, result_len), result_len, lhs_len);
                    addAdjoint(builder, adjoints, operation.rhs, appendBinary(builder, adjoint, map[operation.lhs], .mul, result_len), result_len, rhs_len);
                },
                .div => {
                    addAdjoint(builder, adjoints, operation.lhs, appendBinary(builder, adjoint, map[operation.rhs], .div, result_len), result_len, lhs_len);
                    const denominator = appendBinary(builder, map[operation.rhs], map[operation.rhs], .mul, result_len);
                    const numerator = appendBinary(builder, adjoint, map[operation.lhs], .mul, result_len);
                    const negative = appendUnary(builder, appendBinary(builder, numerator, denominator, .div, result_len), .neg, result_len);
                    addAdjoint(builder, adjoints, operation.rhs, negative, result_len, rhs_len);
                },
                .atan2 => {
                    const lhs_squared = appendBinary(builder, map[operation.lhs], map[operation.lhs], .mul, result_len);
                    const rhs_squared = appendBinary(builder, map[operation.rhs], map[operation.rhs], .mul, result_len);
                    const denominator = appendBinary(builder, lhs_squared, rhs_squared, .add, result_len);
                    const lhs_numerator = appendBinary(builder, adjoint, map[operation.rhs], .mul, result_len);
                    addAdjoint(builder, adjoints, operation.lhs, appendBinary(builder, lhs_numerator, denominator, .div, result_len), result_len, lhs_len);
                    const rhs_numerator = appendBinary(builder, adjoint, map[operation.lhs], .mul, result_len);
                    const rhs_negative = appendUnary(builder, appendBinary(builder, rhs_numerator, denominator, .div, result_len), .neg, result_len);
                    addAdjoint(builder, adjoints, operation.rhs, rhs_negative, result_len, rhs_len);
                },
                .set => |element_index| {
                    const zero = appendConstant(T, builder, 0, 1);
                    const lhs_contribution = appendBinary(builder, adjoint, zero, .{ .set = element_index }, result_len);
                    addAdjoint(builder, adjoints, operation.lhs, lhs_contribution, result_len, lhs_len);
                    const rhs_contribution = appendUnary(builder, adjoint, .{ .get = element_index }, 1);
                    addAdjoint(builder, adjoints, operation.rhs, rhs_contribution, 1, rhs_len);
                },
            }
        },
        .muladd => |operation| {
            const a_len = codeLen(T, source[operation.a]);
            const b_len = codeLen(T, source[operation.b]);
            const c_len = codeLen(T, source[operation.c]);
            addAdjoint(builder, adjoints, operation.a, appendBinary(builder, adjoint, map[operation.b], .mul, result_len), result_len, a_len);
            addAdjoint(builder, adjoints, operation.b, appendBinary(builder, adjoint, map[operation.a], .mul, result_len), result_len, b_len);
            addAdjoint(builder, adjoints, operation.c, adjoint, result_len, c_len);
        },
        .output => {},
    }
}

fn gradientCapacity(comptime T: type, comptime source: []const ir.IRCode(T), comptime output_index: usize) usize {
    const output_len = codeLen(T, source[outputInstruction(T, source, output_index)]);
    return source.len + output_len * (source.len * 32 + 16) + output_len * 4 + 16;
}

fn buildGradient(
    comptime T: type,
    comptime source: []const ir.IRCode(T),
    comptime selected_input: usize,
    comptime selected_output: usize,
    storage: []ir.IRCode(T),
) usize {
    const input_instruction = inputInstruction(T, source, selected_input);
    const output_instruction = outputInstruction(T, source, selected_output);
    const input_len = codeLen(T, source[input_instruction]);
    const output_len = codeLen(T, source[output_instruction]);
    var builder = simplify.Builder(T).init(storage, true);
    var map: [source.len]usize = undefined;
    inline for (source, 0..) |code, index| {
        if (code == .output) continue;
        map[index] = builder.append(remapCode(T, code, &map));
    }

    var scalar_rows: [output_len]usize = undefined;
    inline for (0..output_len) |output_element| {
        var adjoints: [source.len]?usize = @splat(null);
        adjoints[output_instruction] = if (output_len == 1)
            appendConstant(T, &builder, 1, 1)
        else blk: {
            const zero = appendConstant(T, &builder, 0, output_len);
            const one = appendConstant(T, &builder, 1, 1);
            break :blk appendBinary(&builder, zero, one, .{ .set = output_element }, output_len);
        };

        inline for (0..source.len) |offset| {
            const index = source.len - 1 - offset;
            reverseInstruction(T, source, &builder, &map, &adjoints, index);
        }

        const derivative = adjoints[input_instruction] orelse appendConstant(T, &builder, 0, input_len);
        if (output_len == 1 or input_len > 1) {
            _ = builder.append(.{ .output = derivative });
        } else {
            scalar_rows[output_element] = derivative;
        }
    }

    if (output_len > 1 and input_len == 1) {
        var vector = appendConstant(T, &builder, 0, output_len);
        inline for (0..output_len) |element| {
            vector = appendBinary(&builder, vector, scalar_rows[element], .{ .set = element }, output_len);
        }
        _ = builder.append(.{ .output = vector });
    }
    return builder.len;
}

fn generatedGradientLen(
    comptime T: type,
    comptime source: []const ir.IRCode(T),
    comptime input_index: usize,
    comptime output_index: usize,
) usize {
    @setEvalBranchQuota(10_000_000);
    var storage: [gradientCapacity(T, source, output_index)]ir.IRCode(T) = undefined;
    return buildGradient(T, source, input_index, output_index, &storage);
}

fn generatedGradient(
    comptime T: type,
    comptime source: []const ir.IRCode(T),
    comptime input_index: usize,
    comptime output_index: usize,
) [generatedGradientLen(T, source, input_index, output_index)]ir.IRCode(T) {
    @setEvalBranchQuota(10_000_000);
    var storage: [gradientCapacity(T, source, output_index)]ir.IRCode(T) = undefined;
    const len = buildGradient(T, source, input_index, output_index, &storage);
    return storage[0..len].*;
}

fn gradientType(
    comptime T: type,
    comptime source: []const ir.IRCode(T),
    comptime input_index: usize,
    comptime output_index: usize,
) type {
    @setEvalBranchQuota(10_000_000);
    const generated = comptime generatedGradient(T, source, input_index, output_index);
    const result = comptime simplify.simplifyIR(T, &generated);
    return @TypeOf(result);
}

pub fn gradIR(
    comptime T: type,
    comptime source: []const ir.IRCode(T),
    comptime input_index: usize,
    comptime output_index: usize,
) gradientType(T, source, input_index, output_index) {
    @setEvalBranchQuota(10_000_000);
    const generated = comptime generatedGradient(T, source, input_index, output_index);
    return comptime simplify.simplifyIR(T, &generated);
}

test "scalar output differentiates with respect to selected Scalar input" {
    const source = [_]ir.IRCode(f64){
        .{ .scalar_input_index = 0 },
        .{ .scalar_input_index = 1 },
        .{ .Op2 = .{ .lhs = 0, .rhs = 1, .op = .mul, .len = 1 } },
        .{ .output = 2 },
    };
    const derivative = comptime gradIR(f64, &source, 0, 0);
    const result = ir.evalIRCode(f64, &derivative, .{ 2, 3 });
    try @import("std").testing.expectEqual(@as(f64, 3), result[0]);
}

test "scalar output differentiates with respect to selected Vector input" {
    const source = [_]ir.IRCode(f32){
        .{ .vec_input = .{ .input_index = 0, .len = 3 } },
        .{ .vec_input = .{ .input_index = 1, .len = 3 } },
        .{ .Op2 = .{ .lhs = 0, .rhs = 1, .op = .mul, .len = 3 } },
        .{ .Op1 = .{ .a = 2, .op = .sum, .len = 1 } },
        .{ .output = 3 },
    };
    const derivative = comptime gradIR(f32, &source, 0, 0);
    const result = ir.evalIRCode(f32, &derivative, .{ @Vector(3, f32){ 1, 2, 3 }, @Vector(3, f32){ 4, 5, 6 } });
    try @import("std").testing.expectEqual(@Vector(3, f32){ 4, 5, 6 }, result[0]);
}

test "Vector output and Scalar input produce one Vector derivative" {
    const source = [_]ir.IRCode(f64){
        .{ .vec_input = .{ .input_index = 0, .len = 3 } },
        .{ .scalar_input_index = 1 },
        .{ .Op2 = .{ .lhs = 0, .rhs = 1, .op = .mul, .len = 3 } },
        .{ .output = 2 },
    };
    const derivative = comptime gradIR(f64, &source, 1, 0);
    const result = ir.evalIRCode(f64, &derivative, .{ @Vector(3, f64){ 1, 2, 3 }, 4 });
    try @import("std").testing.expectEqual(@Vector(3, f64){ 1, 2, 3 }, result[0]);
}

test "Vector output and Vector input produce Jacobian row outputs" {
    const source = [_]ir.IRCode(f64){
        .{ .vec_input = .{ .input_index = 0, .len = 2 } },
        .{ .Op2 = .{ .lhs = 0, .rhs = 0, .op = .mul, .len = 2 } },
        .{ .output = 1 },
    };
    const derivative = comptime gradIR(f64, &source, 0, 0);
    const result = ir.evalIRCode(f64, &derivative, .{@Vector(2, f64){ 3, 4 }});
    try @import("std").testing.expectEqual(@Vector(2, f64){ 6, 0 }, result[0]);
    try @import("std").testing.expectEqual(@Vector(2, f64){ 0, 8 }, result[1]);
}

test "input and output indices select logical Tuple positions" {
    const source = [_]ir.IRCode(f64){
        .{ .scalar_input_index = 0 },
        .{ .scalar_input_index = 1 },
        .{ .Op2 = .{ .lhs = 0, .rhs = 1, .op = .add, .len = 1 } },
        .{ .Op2 = .{ .lhs = 0, .rhs = 1, .op = .mul, .len = 1 } },
        .{ .output = 2 },
        .{ .output = 3 },
    };
    const add_wrt_y = comptime gradIR(f64, &source, 1, 0);
    const mul_wrt_x = comptime gradIR(f64, &source, 0, 1);
    try @import("std").testing.expectEqual(@as(f64, 1), ir.evalIRCode(f64, &add_wrt_y, .{ 2, 3 })[0]);
    try @import("std").testing.expectEqual(@as(f64, 3), ir.evalIRCode(f64, &mul_wrt_x, .{ 2, 3 })[0]);
}

test "get and set propagate selected gradients" {
    const source = [_]ir.IRCode(f64){
        .{ .vec_input = .{ .input_index = 0, .len = 3 } },
        .{ .scalar_input_index = 1 },
        .{ .Op2 = .{ .lhs = 0, .rhs = 1, .op = .{ .set = 1 }, .len = 3 } },
        .{ .Op1 = .{ .a = 2, .op = .sum, .len = 1 } },
        .{ .output = 3 },
    };
    const wrt_vector = comptime gradIR(f64, &source, 0, 0);
    const wrt_scalar = comptime gradIR(f64, &source, 1, 0);
    const inputs = .{ @Vector(3, f64){ 1, 2, 3 }, @as(f64, 9) };
    try @import("std").testing.expectEqual(@Vector(3, f64){ 1, 0, 1 }, ir.evalIRCode(f64, &wrt_vector, inputs)[0]);
    try @import("std").testing.expectEqual(@as(f64, 1), ir.evalIRCode(f64, &wrt_scalar, inputs)[0]);
}

test "muladd and atan2 differentiation" {
    const source = [_]ir.IRCode(f64){
        .{ .scalar_input_index = 0 },
        .{ .scalar_input_index = 1 },
        .{ .scalar_input_index = 2 },
        .{ .muladd = .{ .a = 0, .b = 1, .c = 2, .len = 1 } },
        .{ .Op2 = .{ .lhs = 3, .rhs = 1, .op = .atan2, .len = 1 } },
        .{ .output = 4 },
    };
    const derivative = comptime gradIR(f64, &source, 0, 0);
    const actual = ir.evalIRCode(f64, &derivative, .{ 2, 3, 1 })[0];
    try @import("std").testing.expectApproxEqAbs(@as(f64, 9.0 / 58.0), actual, 1e-12);
}

test "generated derivative IR supports second differentiation" {
    const source = [_]ir.IRCode(f64){
        .{ .scalar_input_index = 0 },
        .{ .Op2 = .{ .lhs = 0, .rhs = 0, .op = .mul, .len = 1 } },
        .{ .Op2 = .{ .lhs = 1, .rhs = 0, .op = .mul, .len = 1 } },
        .{ .output = 2 },
    };
    const first = comptime gradIR(f64, &source, 0, 0);
    const second = comptime gradIR(f64, &first, 0, 0);
    try @import("std").testing.expectEqual(@as(f64, 12), ir.evalIRCode(f64, &second, .{2})[0]);
}

test "all differentiable unary IR operations generate correct derivatives" {
    const source = [_]ir.IRCode(f64){
        .{ .scalar_input_index = 0 },
        .{ .Op1 = .{ .a = 0, .op = .neg, .len = 1 } },
        .{ .Op1 = .{ .a = 0, .op = .sqrt, .len = 1 } },
        .{ .Op1 = .{ .a = 0, .op = .exp, .len = 1 } },
        .{ .Op1 = .{ .a = 0, .op = .log, .len = 1 } },
        .{ .Op1 = .{ .a = 0, .op = .sin, .len = 1 } },
        .{ .Op1 = .{ .a = 0, .op = .cos, .len = 1 } },
        .{ .Op1 = .{ .a = 0, .op = .abs, .len = 1 } },
        .{ .Op1 = .{ .a = 0, .op = .sum, .len = 1 } },
        .{ .output = 1 },
        .{ .output = 2 },
        .{ .output = 3 },
        .{ .output = 4 },
        .{ .output = 5 },
        .{ .output = 6 },
        .{ .output = 7 },
        .{ .output = 8 },
    };
    const x: f64 = 2;
    const expected = [_]f64{
        -1,
        1.0 / (2.0 * @sqrt(x)),
        @exp(x),
        1.0 / x,
        @cos(x),
        -@sin(x),
        1,
        1,
    };
    inline for (expected, 0..) |expected_value, output_index| {
        const derivative = comptime gradIR(f64, &source, 0, output_index);
        const actual = ir.evalIRCode(f64, &derivative, .{x})[0];
        try @import("std").testing.expectApproxEqAbs(expected_value, actual, 1e-12);
    }
}

test "broadcast adjoints reduce back to a selected Scalar input" {
    const source = [_]ir.IRCode(f64){
        .{ .vec_input = .{ .input_index = 0, .len = 3 } },
        .{ .scalar_input_index = 1 },
        .{ .Op2 = .{ .lhs = 0, .rhs = 1, .op = .mul, .len = 3 } },
        .{ .Op1 = .{ .a = 2, .op = .sum, .len = 1 } },
        .{ .output = 3 },
    };
    const derivative = comptime gradIR(f64, &source, 1, 0);
    const actual = ir.evalIRCode(f64, &derivative, .{ @Vector(3, f64){ 1, 2, 3 }, 4 })[0];
    try @import("std").testing.expectEqual(@as(f64, 6), actual);
}

test "standalone get produces a Vector basis gradient" {
    const source = [_]ir.IRCode(f32){
        .{ .vec_input = .{ .input_index = 0, .len = 3 } },
        .{ .Op1 = .{ .a = 0, .op = .{ .get = 1 }, .len = 1 } },
        .{ .output = 1 },
    };
    const derivative = comptime gradIR(f32, &source, 0, 0);
    const actual = ir.evalIRCode(f32, &derivative, .{@Vector(3, f32){ 4, 5, 6 }})[0];
    try @import("std").testing.expectEqual(@Vector(3, f32){ 0, 1, 0 }, actual);
}

test "IR differentiation supports f16 f32 and f64" {
    inline for (.{ f16, f32, f64 }) |T| {
        const source = [_]ir.IRCode(T){
            .{ .scalar_input_index = 0 },
            .{ .scalar_input_index = 1 },
            .{ .Op2 = .{ .lhs = 0, .rhs = 1, .op = .mul, .len = 1 } },
            .{ .output = 2 },
        };
        const derivative = comptime gradIR(T, &source, 0, 0);
        const actual = ir.evalIRCode(T, &derivative, .{ @as(T, 2), @as(T, 3) })[0];
        try @import("std").testing.expectEqual(@as(T, 3), actual);
    }
}
