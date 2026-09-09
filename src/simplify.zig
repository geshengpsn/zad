const std = @import("std");
const ir = @import("ir.zig");

fn staticVector(comptime T: type, comptime len: usize, comptime values: [len]T) []const T {
    return &struct {
        const data = values;
    }.data;
}

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

fn atan2(comptime T: type, y: T, x: T) T {
    return switch (T) {
        f16 => @floatCast(std.math.atan2(@as(f32, y), @as(f32, x))),
        f32, f64 => std.math.atan2(y, x),
        else => unreachable,
    };
}

fn unaryValue(comptime T: type, op: ir.Op1Code, value: T) T {
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

fn binaryValue(comptime T: type, op: ir.Op2Code, lhs: T, rhs: T) T {
    return switch (op) {
        .add => lhs + rhs,
        .sub => lhs - rhs,
        .mul => lhs * rhs,
        .div => lhs / rhs,
        .atan2 => atan2(T, lhs, rhs),
        .set => unreachable,
    };
}

fn isValue(comptime T: type, code: ir.IRCode(T), value: T) bool {
    return switch (code) {
        .scalar_constant => |constant| constant == value,
        .vec_constant => |constants| blk: {
            for (constants) |constant| if (constant != value) break :blk false;
            break :blk true;
        },
        else => false,
    };
}

fn sameCode(comptime T: type, lhs: ir.IRCode(T), rhs: ir.IRCode(T)) bool {
    if (std.meta.activeTag(lhs) != std.meta.activeTag(rhs)) return false;
    return switch (lhs) {
        .scalar_constant => |value| value == rhs.scalar_constant,
        .scalar_input_index => |index| index == rhs.scalar_input_index,
        .vec_constant => |values| std.mem.eql(T, values, rhs.vec_constant),
        .vec_input => |input| input.input_index == rhs.vec_input.input_index and input.len == rhs.vec_input.len,
        .Op1 => |op| op.a == rhs.Op1.a and std.meta.eql(op.op, rhs.Op1.op) and op.len == rhs.Op1.len,
        .Op2 => |op| op.lhs == rhs.Op2.lhs and op.rhs == rhs.Op2.rhs and std.meta.eql(op.op, rhs.Op2.op) and op.len == rhs.Op2.len,
        .muladd => |op| op.a == rhs.muladd.a and op.b == rhs.muladd.b and op.c == rhs.muladd.c and op.len == rhs.muladd.len,
        .output => false,
    };
}

pub fn Builder(comptime T: type) type {
    return struct {
        const Self = @This();

        codes: []ir.IRCode(T),
        len: usize = 0,
        enabled: bool = true,

        pub fn init(codes: []ir.IRCode(T), enabled: bool) Self {
            return .{ .codes = codes, .enabled = enabled };
        }

        fn appendRaw(self: *Self, comptime code: ir.IRCode(T)) usize {
            const index = self.len;
            if (index < self.codes.len) self.codes[index] = code;
            self.len += 1;
            return index;
        }

        fn findEquivalent(self: *Self, comptime code: ir.IRCode(T)) ?usize {
            // An equivalent operation must also follow all of its operands.
            const start = switch (code) {
                .Op1 => |op| op.a + 1,
                .Op2 => |op| @as(usize, @max(op.lhs, op.rhs)) + 1,
                .muladd => |op| @as(usize, @max(op.a, op.b, op.c)) + 1,
                else => 0,
            };
            for (self.codes[start..self.len], start..) |existing, index| {
                if (sameCode(T, existing, code)) return index;
            }
            return null;
        }

        fn aliasForResult(self: *Self, index: usize, result_len: usize) ?usize {
            return if (codeLen(T, self.codes[index]) == result_len) index else null;
        }

        fn zeroForResult(self: *Self, index: usize, result_len: usize) ?usize {
            if (!isValue(T, self.codes[index], 0)) return null;
            if (self.aliasForResult(index, result_len)) |alias| return alias;
            if (result_len > 1 and self.codes[index] == .scalar_constant) {
                const values: [result_len]T = @splat(0);
                return self.append(.{ .vec_constant = staticVector(T, result_len, values) });
            }
            return null;
        }

        fn appendConstant(self: *Self, comptime value: T, comptime result_len: usize) usize {
            if (result_len == 1) return self.append(.{ .scalar_constant = value });
            const values: [result_len]T = @splat(value);
            return self.append(.{ .vec_constant = staticVector(T, result_len, values) });
        }

        fn appendUnary(self: *Self, comptime operation: @FieldType(ir.IRCode(T), "Op1")) usize {
            if (operation.a >= self.len) @compileError("Op1 operand must reference an earlier instruction");
            const operand = self.codes[operation.a];
            const operand_len = codeLen(T, operand);
            switch (operation.op) {
                .sum => {
                    if (operation.len != 1) @compileError("sum must produce a scalar");
                    if (operand_len == 1) return operation.a;
                    if (operand == .vec_constant) {
                        var result: T = 0;
                        for (operand.vec_constant) |value| result += value;
                        return self.append(.{ .scalar_constant = result });
                    }
                },
                .get => |element_index| {
                    if (operation.len != 1) @compileError("get must produce a scalar");
                    if (element_index >= operand_len) @compileError("get index is out of bounds");
                    if (operand == .vec_constant) return self.append(.{ .scalar_constant = operand.vec_constant[element_index] });
                    if (operand == .Op2) switch (operand.Op2.op) {
                        .set => |set_index| {
                            if (set_index == element_index) return operand.Op2.rhs;
                            return self.append(.{ .Op1 = .{ .a = operand.Op2.lhs, .op = operation.op, .len = 1 } });
                        },
                        else => {},
                    };
                },
                .neg => if (operand == .Op1 and operand.Op1.op == .neg) return operand.Op1.a,
                .abs => if (operand == .Op1 and operand.Op1.op == .abs) return operation.a,
                .cos => if (operand == .Op1 and operand.Op1.op == .neg) {
                    return self.append(.{ .Op1 = .{ .a = operand.Op1.a, .op = .cos, .len = operation.len } });
                },
                .log => if (operand == .Op1 and operand.Op1.op == .exp) return operand.Op1.a,
                else => {},
            }

            if (operation.op != .sum and operation.op != .get and operation.len != operand_len) {
                @compileError("Op1 result length must match its operand");
            }
            switch (operand) {
                .scalar_constant => |value| return self.append(.{ .scalar_constant = unaryValue(T, operation.op, value) }),
                .vec_constant => |values| if (operation.op != .sum and operation.op != .get) {
                    var result: [values.len]T = undefined;
                    for (values, 0..) |value, index| result[index] = unaryValue(T, operation.op, value);
                    return self.append(.{ .vec_constant = staticVector(T, values.len, result) });
                },
                else => {},
            }
            return self.appendDeduplicated(.{ .Op1 = operation });
        }

        fn appendBinary(self: *Self, comptime original: @FieldType(ir.IRCode(T), "Op2")) usize {
            if (original.lhs >= self.len or original.rhs >= self.len) @compileError("Op2 operands must reference earlier instructions");
            var operation = original;
            switch (operation.op) {
                .add, .mul => if (operation.lhs > operation.rhs) {
                    const temporary = operation.lhs;
                    operation.lhs = operation.rhs;
                    operation.rhs = temporary;
                },
                else => {},
            }
            const lhs = self.codes[operation.lhs];
            const rhs = self.codes[operation.rhs];
            const lhs_len = codeLen(T, lhs);
            const rhs_len = codeLen(T, rhs);
            if (operation.op != .set) {
                if ((lhs_len != 1 and lhs_len != operation.len) or (rhs_len != 1 and rhs_len != operation.len)) {
                    @compileError("Op2 operand length does not match its result");
                }
                if (operation.len == 1 and (lhs_len != 1 or rhs_len != 1)) {
                    @compileError("scalar Op2 cannot consume a Vector operand");
                }
            }

            switch (operation.op) {
                .set => |element_index| {
                    if (codeLen(T, lhs) != operation.len or codeLen(T, rhs) != 1) @compileError("set operand lengths are invalid");
                    if (element_index >= operation.len) @compileError("set index is out of bounds");
                    if (lhs == .vec_constant and rhs == .scalar_constant) {
                        var result: [operation.len]T = lhs.vec_constant[0..operation.len].*;
                        result[element_index] = rhs.scalar_constant;
                        return self.append(.{ .vec_constant = staticVector(T, operation.len, result) });
                    }
                    if (rhs == .Op1) switch (rhs.Op1.op) {
                        .get => |get_index| if (get_index == element_index and rhs.Op1.a == operation.lhs) return operation.lhs,
                        else => {},
                    };
                    if (lhs == .Op2) switch (lhs.Op2.op) {
                        .set => |set_index| if (set_index == element_index) {
                            operation.lhs = lhs.Op2.lhs;
                        },
                        else => {},
                    };
                    return self.appendDeduplicated(.{ .Op2 = operation });
                },
                .add => {
                    if (isValue(T, lhs, 0)) if (self.aliasForResult(operation.rhs, operation.len)) |alias| return alias;
                    if (isValue(T, rhs, 0)) if (self.aliasForResult(operation.lhs, operation.len)) |alias| return alias;
                },
                .sub => {
                    if (operation.lhs == operation.rhs) return self.appendConstant(0, operation.len);
                    if (isValue(T, rhs, 0)) if (self.aliasForResult(operation.lhs, operation.len)) |alias| return alias;
                },
                .mul => {
                    if (self.zeroForResult(operation.lhs, operation.len)) |zero| return zero;
                    if (self.zeroForResult(operation.rhs, operation.len)) |zero| return zero;
                    if (isValue(T, lhs, 1)) if (self.aliasForResult(operation.rhs, operation.len)) |alias| return alias;
                    if (isValue(T, rhs, 1)) if (self.aliasForResult(operation.lhs, operation.len)) |alias| return alias;
                    if (isValue(T, lhs, -1)) if (self.aliasForResult(operation.rhs, operation.len)) |alias| {
                        return self.append(.{ .Op1 = .{ .a = alias, .op = .neg, .len = operation.len } });
                    };
                    if (isValue(T, rhs, -1)) if (self.aliasForResult(operation.lhs, operation.len)) |alias| {
                        return self.append(.{ .Op1 = .{ .a = alias, .op = .neg, .len = operation.len } });
                    };
                },
                .div => if (isValue(T, rhs, 1)) if (self.aliasForResult(operation.lhs, operation.len)) |alias| return alias,
                .atan2 => {},
            }

            if (lhs == .scalar_constant and rhs == .scalar_constant and operation.len == 1) {
                return self.append(.{ .scalar_constant = binaryValue(T, operation.op, lhs.scalar_constant, rhs.scalar_constant) });
            }
            if (operation.len > 1 and (lhs == .scalar_constant or lhs == .vec_constant) and (rhs == .scalar_constant or rhs == .vec_constant)) {
                var result: [operation.len]T = undefined;
                for (0..operation.len) |index| {
                    const lhs_value = if (lhs == .scalar_constant) lhs.scalar_constant else lhs.vec_constant[index];
                    const rhs_value = if (rhs == .scalar_constant) rhs.scalar_constant else rhs.vec_constant[index];
                    result[index] = binaryValue(T, operation.op, lhs_value, rhs_value);
                }
                return self.append(.{ .vec_constant = staticVector(T, operation.len, result) });
            }
            return self.appendDeduplicated(.{ .Op2 = operation });
        }

        fn appendMulAdd(self: *Self, comptime operation: @FieldType(ir.IRCode(T), "muladd")) usize {
            if (operation.a >= self.len or operation.b >= self.len or operation.c >= self.len) @compileError("muladd operands must reference earlier instructions");
            const a = self.codes[operation.a];
            const b = self.codes[operation.b];
            const c = self.codes[operation.c];
            const a_len = codeLen(T, a);
            const b_len = codeLen(T, b);
            const c_len = codeLen(T, c);
            if ((a_len != 1 and a_len != operation.len) or
                (b_len != 1 and b_len != operation.len) or
                (c_len != 1 and c_len != operation.len))
            {
                @compileError("muladd operand length does not match its result");
            }
            if (operation.len == 1 and (a_len != 1 or b_len != 1 or c_len != 1)) {
                @compileError("scalar muladd cannot consume a Vector operand");
            }
            if (isValue(T, a, 0) or isValue(T, b, 0)) {
                if (self.aliasForResult(operation.c, operation.len)) |alias| return alias;
            }
            if (isValue(T, a, 1)) return self.append(.{ .Op2 = .{ .lhs = operation.b, .rhs = operation.c, .op = .add, .len = operation.len } });
            if (isValue(T, b, 1)) return self.append(.{ .Op2 = .{ .lhs = operation.a, .rhs = operation.c, .op = .add, .len = operation.len } });

            if (a == .scalar_constant and b == .scalar_constant and c == .scalar_constant and operation.len == 1) {
                return self.append(.{ .scalar_constant = @mulAdd(T, a.scalar_constant, b.scalar_constant, c.scalar_constant) });
            }
            if (operation.len > 1 and
                (a == .scalar_constant or a == .vec_constant) and
                (b == .scalar_constant or b == .vec_constant) and
                (c == .scalar_constant or c == .vec_constant))
            {
                var result: [operation.len]T = undefined;
                for (0..operation.len) |index| {
                    const a_value = if (a == .scalar_constant) a.scalar_constant else a.vec_constant[index];
                    const b_value = if (b == .scalar_constant) b.scalar_constant else b.vec_constant[index];
                    const c_value = if (c == .scalar_constant) c.scalar_constant else c.vec_constant[index];
                    result[index] = @mulAdd(T, a_value, b_value, c_value);
                }
                return self.append(.{ .vec_constant = staticVector(T, operation.len, result) });
            }
            return self.appendDeduplicated(.{ .muladd = operation });
        }

        fn appendDeduplicated(self: *Self, comptime code: ir.IRCode(T)) usize {
            if (self.findEquivalent(code)) |index| return index;
            return self.appendRaw(code);
        }

        pub fn append(self: *Self, comptime code: ir.IRCode(T)) usize {
            if (!self.enabled) return self.appendRaw(code);
            return switch (code) {
                .Op1 => |operation| self.appendUnary(operation),
                .Op2 => |operation| self.appendBinary(operation),
                .muladd => |operation| self.appendMulAdd(operation),
                .output => self.appendRaw(code),
                else => self.appendDeduplicated(code),
            };
        }
    };
}

fn rewriteCode(comptime T: type, code: ir.IRCode(T), map: []const usize) ir.IRCode(T) {
    return switch (code) {
        .Op1 => |op| .{ .Op1 = .{ .a = map[op.a], .op = op.op, .len = op.len } },
        .Op2 => |op| .{ .Op2 = .{ .lhs = map[op.lhs], .rhs = map[op.rhs], .op = op.op, .len = op.len } },
        .muladd => |op| .{ .muladd = .{ .a = map[op.a], .b = map[op.b], .c = map[op.c], .len = op.len } },
        .output => |index| .{ .output = map[index] },
        else => code,
    };
}

fn locallySimplified(comptime T: type, comptime source: []const ir.IRCode(T)) struct {
    codes: [source.len]ir.IRCode(T),
    len: usize,
} {
    @setEvalBranchQuota(10_000_000);
    var storage: [source.len]ir.IRCode(T) = undefined;
    var builder = Builder(T).init(&storage, true);
    var map: [source.len]usize = undefined;
    inline for (source, 0..) |code, index| {
        const rewritten = rewriteCode(T, code, &map);
        const result = builder.append(rewritten);
        if (code != .output) map[index] = result;
    }
    return .{ .codes = storage, .len = builder.len };
}

fn markActive(comptime T: type, codes: []const ir.IRCode(T), active: []bool, index: usize) void {
    if (active[index]) return;
    active[index] = true;
    switch (codes[index]) {
        .Op1 => |op| markActive(T, codes, active, op.a),
        .Op2 => |op| {
            markActive(T, codes, active, op.lhs);
            markActive(T, codes, active, op.rhs);
        },
        .muladd => |op| {
            markActive(T, codes, active, op.a);
            markActive(T, codes, active, op.b);
            markActive(T, codes, active, op.c);
        },
        .output => |output| markActive(T, codes, active, output),
        else => {},
    }
}

fn activeNodes(comptime T: type, comptime source: []const ir.IRCode(T)) struct {
    codes: [source.len]ir.IRCode(T),
    active: [source.len]bool,
    len: usize,
} {
    @setEvalBranchQuota(10_000_000);
    const local = locallySimplified(T, source);
    var active: [source.len]bool = @splat(false);
    for (local.codes[0..local.len], 0..) |code, index| {
        switch (code) {
            .scalar_input_index, .vec_input, .output => markActive(T, local.codes[0..local.len], &active, index),
            else => {},
        }
    }
    var len: usize = 0;
    for (active[0..local.len]) |is_active| if (is_active) {
        len += 1;
    };
    return .{ .codes = local.codes, .active = active, .len = len };
}

fn simplifiedLen(comptime T: type, comptime source: []const ir.IRCode(T)) usize {
    @setEvalBranchQuota(10_000_000);
    return activeNodes(T, source).len;
}

pub fn simplifyIR(comptime T: type, comptime source: []const ir.IRCode(T)) [simplifiedLen(T, source)]ir.IRCode(T) {
    return comptime blk: {
        @setEvalBranchQuota(1_000_000);
        const state = activeNodes(T, source);
        var result: [simplifiedLen(T, source)]ir.IRCode(T) = undefined;
        var map: [source.len]usize = undefined;
        var len: usize = 0;
        for (0..state.active.len) |index| {
            if (!state.active[index]) continue;
            const code = state.codes[index];
            result[len] = rewriteCode(T, code, &map);
            if (code != .output) map[index] = len;
            len += 1;
        }
        break :blk result;
    };
}

test "Builder simplifies while generating IR" {
    const result = comptime blk: {
        var storage: [16]ir.IRCode(f64) = undefined;
        var builder = Builder(f64).init(&storage, true);
        const x = builder.append(.{ .scalar_input_index = 0 });
        const zero = builder.append(.{ .scalar_constant = 0 });
        const one = builder.append(.{ .scalar_constant = 1 });
        const plus_zero = builder.append(.{ .Op2 = .{ .lhs = x, .rhs = zero, .op = .add, .len = 1 } });
        const times_one = builder.append(.{ .Op2 = .{ .lhs = plus_zero, .rhs = one, .op = .mul, .len = 1 } });
        const negative = builder.append(.{ .Op1 = .{ .a = times_one, .op = .neg, .len = 1 } });
        const restored = builder.append(.{ .Op1 = .{ .a = negative, .op = .neg, .len = 1 } });
        break :blk .{ .x = x, .restored = restored, .len = builder.len };
    };
    try std.testing.expectEqual(result.x, result.restored);
    try std.testing.expectEqual(@as(usize, 4), result.len);
}

test "Builder deduplicates operations immediately after their operands" {
    const result = comptime blk: {
        var storage: [8]ir.IRCode(f64) = undefined;
        var builder = Builder(f64).init(&storage, true);
        _ = builder.append(.{ .scalar_input_index = 0 });
        const operations = [_]ir.IRCode(f64){
            .{ .Op1 = .{ .a = 0, .op = .neg, .len = 1 } },
            .{ .Op2 = .{ .lhs = 0, .rhs = 1, .op = .sub, .len = 1 } },
            .{ .muladd = .{ .a = 0, .b = 1, .c = 2, .len = 1 } },
        };
        for (operations) |code| _ = builder.append(code);
        var indices: [operations.len]usize = undefined;
        for (operations, 0..) |code, index| indices[index] = builder.append(code);
        break :blk .{ .indices = indices, .len = builder.len };
    };
    try std.testing.expectEqual([3]usize{ 1, 2, 3 }, result.indices);
    try std.testing.expectEqual(@as(usize, 4), result.len);
}

test "Builder simplifies get through unrelated sets during creation" {
    inline for (.{ f16, f32, f64 }) |T| {
        const result = comptime blk: {
            var storage: [16]ir.IRCode(T) = undefined;
            var builder = Builder(T).init(&storage, true);
            const vector = builder.append(.{ .vec_input = .{ .input_index = 0, .len = 3 } });
            const scalar = builder.append(.{ .scalar_input_index = 1 });
            const original = builder.append(.{ .Op1 = .{ .a = vector, .op = .{ .get = 0 }, .len = 1 } });
            const first = builder.append(.{ .Op2 = .{ .lhs = vector, .rhs = scalar, .op = .{ .set = 1 }, .len = 3 } });
            const second = builder.append(.{ .Op2 = .{ .lhs = first, .rhs = scalar, .op = .{ .set = 2 }, .len = 3 } });
            const len_before = builder.len;
            const unchanged = builder.append(.{ .Op1 = .{ .a = second, .op = .{ .get = 0 }, .len = 1 } });
            const changed = builder.append(.{ .Op1 = .{ .a = second, .op = .{ .get = 1 }, .len = 1 } });
            break :blk .{
                .original = original,
                .scalar = scalar,
                .unchanged = unchanged,
                .changed = changed,
                .len_before = len_before,
                .len_after = builder.len,
            };
        };
        try std.testing.expectEqual(result.original, result.unchanged);
        try std.testing.expectEqual(result.scalar, result.changed);
        try std.testing.expectEqual(result.len_before, result.len_after);
    }
}

test "simplifyIR rewrites references and preserves evaluation" {
    const source = [_]ir.IRCode(f32){
        .{ .vec_input = .{ .input_index = 0, .len = 3 } },
        .{ .vec_constant = &.{ 0, 0, 0 } },
        .{ .Op2 = .{ .lhs = 0, .rhs = 1, .op = .add, .len = 3 } },
        .{ .Op1 = .{ .a = 2, .op = .neg, .len = 3 } },
        .{ .Op1 = .{ .a = 3, .op = .neg, .len = 3 } },
        .{ .output = 4 },
    };
    const simplified = comptime simplifyIR(f32, &source);
    try std.testing.expect(simplified.len < source.len);
    const input = @Vector(3, f32){ 1, 2, 3 };
    try std.testing.expectEqual(ir.evalIRCode(f32, &source, .{input}), ir.evalIRCode(f32, &simplified, .{input}));
}

fn testConstantFolding(comptime T: type) !void {
    const source = [_]ir.IRCode(T){
        .{ .scalar_constant = 4 },
        .{ .Op1 = .{ .a = 0, .op = .sqrt, .len = 1 } },
        .{ .scalar_constant = 3 },
        .{ .Op2 = .{ .lhs = 1, .rhs = 2, .op = .add, .len = 1 } },
        .{ .scalar_constant = 2 },
        .{ .scalar_constant = 1 },
        .{ .muladd = .{ .a = 3, .b = 4, .c = 5, .len = 1 } },
        .{ .output = 6 },
    };
    const result = comptime simplifyIR(T, &source);
    try std.testing.expectEqual(@as(usize, 2), result.len);
    try std.testing.expectEqual(@as(T, 11), ir.evalIRCode(T, &result, .{})[0]);
}

test "constant folding supports every IR float type" {
    try testConstantFolding(f16);
    try testConstantFolding(f32);
    try testConstantFolding(f64);
}

test "broadcast scalar zero simplifies to a vector zero without changing inputs" {
    const source = [_]ir.IRCode(f32){
        .{ .vec_input = .{ .input_index = 0, .len = 3 } },
        .{ .scalar_constant = 0 },
        .{ .Op2 = .{ .lhs = 0, .rhs = 1, .op = .mul, .len = 3 } },
        .{ .output = 2 },
    };
    const result = comptime simplifyIR(f32, &source);
    const Input = ir.InputType(f32, &result);
    if (Input != @Tuple(&.{@Vector(3, f32)})) @compileError("simplification changed the input signature");
    try std.testing.expectEqual(
        @Vector(3, f32){ 0, 0, 0 },
        ir.evalIRCode(f32, &result, .{@Vector(3, f32){ 1, 2, 3 }})[0],
    );
}

test "cos_neg log_exp self_sub and mul_neg_one simplify for vectors" {
    const source = [_]ir.IRCode(f64){
        .{ .vec_input = .{ .input_index = 0, .len = 3 } },
        .{ .Op1 = .{ .a = 0, .op = .neg, .len = 3 } },
        .{ .Op1 = .{ .a = 1, .op = .cos, .len = 3 } },
        .{ .Op1 = .{ .a = 0, .op = .exp, .len = 3 } },
        .{ .Op1 = .{ .a = 3, .op = .log, .len = 3 } },
        .{ .Op2 = .{ .lhs = 0, .rhs = 0, .op = .sub, .len = 3 } },
        .{ .scalar_constant = -1 },
        .{ .Op2 = .{ .lhs = 0, .rhs = 6, .op = .mul, .len = 3 } },
        .{ .output = 2 },
        .{ .output = 4 },
        .{ .output = 5 },
        .{ .output = 7 },
    };
    const simplified = comptime simplifyIR(f64, &source);
    try std.testing.expect(simplified.len < source.len);
    for (simplified) |code| switch (code) {
        .Op1 => |op| switch (op.op) {
            .exp, .log => return error.UnsimplifiedUnaryOperation,
            else => {},
        },
        .Op2 => |op| switch (op.op) {
            .sub, .mul => return error.UnsimplifiedBinaryOperation,
            else => {},
        },
        else => {},
    };

    const input = @Vector(3, f64){ 0.5, 1.0, 2.0 };
    const original_result = ir.evalIRCode(f64, &source, .{input});
    const simplified_result = ir.evalIRCode(f64, &simplified, .{input});
    inline for (0..4) |output_index| {
        inline for (0..3) |lane| {
            try std.testing.expectApproxEqAbs(original_result[output_index][lane], simplified_result[output_index][lane], 1e-12);
        }
    }
}

fn randomNext(state: *u64) u64 {
    state.* = state.* *% 6364136223846793005 +% 1442695040888963407;
    return state.*;
}

fn randomSimplifyIR(
    comptime T: type,
    comptime vec_len: usize,
    comptime rounds: usize,
    comptime seed: u64,
) [3 + rounds * 2 + 1]ir.IRCode(T) {
    const zeros: [vec_len]T = @splat(0);
    var result: [3 + rounds * 2 + 1]ir.IRCode(T) = undefined;
    result[0] = .{ .vec_input = .{ .input_index = 0, .len = vec_len } };
    result[1] = .{ .scalar_constant = 1 };
    result[2] = .{ .vec_constant = &zeros };
    var current: usize = 0;
    var state = seed;
    inline for (0..rounds) |round| {
        const first = 3 + round * 2;
        switch (randomNext(&state) % 4) {
            0 => {
                result[first] = .{ .Op2 = .{ .lhs = current, .rhs = current, .op = .mul, .len = vec_len } };
                result[first + 1] = .{ .Op2 = .{ .lhs = current, .rhs = current, .op = .mul, .len = vec_len } };
                current = first + 1;
            },
            1 => {
                result[first] = .{ .Op2 = .{ .lhs = current, .rhs = 2, .op = .add, .len = vec_len } };
                result[first + 1] = .{ .Op2 = .{ .lhs = first, .rhs = 1, .op = .mul, .len = vec_len } };
                current = first + 1;
            },
            2 => {
                result[first] = .{ .Op1 = .{ .a = current, .op = .neg, .len = vec_len } };
                result[first + 1] = .{ .Op1 = .{ .a = first, .op = .neg, .len = vec_len } };
                current = first + 1;
            },
            3 => {
                const element_index = randomNext(&state) % vec_len;
                result[first] = .{ .Op1 = .{ .a = current, .op = .{ .get = element_index }, .len = 1 } };
                result[first + 1] = .{ .Op2 = .{ .lhs = current, .rhs = first, .op = .{ .set = element_index }, .len = vec_len } };
                current = first + 1;
            },
            else => unreachable,
        }
    }
    result[result.len - 1] = .{ .output = current };
    return result;
}

test "random simplification combinations preserve vector results" {
    @setEvalBranchQuota(1_000_000);
    inline for ([_]usize{ 3, 8 }) |vec_len| {
        inline for (0..16) |case_index| {
            const source = comptime randomSimplifyIR(
                f64,
                vec_len,
                10,
                0x517cc1b727220a95 +% case_index *% 0x9e3779b97f4a7c15,
            );
            const simplified = comptime simplifyIR(f64, &source);
            var input: @Vector(vec_len, f64) = undefined;
            inline for (0..vec_len) |index| input[index] = @as(f64, @floatFromInt(index + 1)) * 0.01;
            try std.testing.expectEqual(
                ir.evalIRCode(f64, &source, .{input}),
                ir.evalIRCode(f64, &simplified, .{input}),
            );
            try std.testing.expect(simplified.len <= source.len);
        }
    }
}
