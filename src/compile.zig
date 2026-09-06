const std = @import("std");
const hr = @import("hr.zig");
const ir = @import("ir.zig");
const differentiation = @import("grad.zig");

const CartPoleScalar = hr.Scalar(f64);
const CartPoleVector = hr.Vector(2, f64);
const mass_pole = CartPoleScalar.init(1.0);
const mass_cart = CartPoleScalar.init(5.0);
const pole_length = CartPoleScalar.init(1.0);
const gravity = CartPoleScalar.init(9.81);

fn cartPolePotential(x: CartPoleVector, dx: CartPoleVector) CartPoleScalar {
    _ = dx;
    _ = mass_cart;
    const theta = x.get(1);
    return mass_pole.mul(gravity).mul(pole_length).mul(theta.cos());
}

pub const GradOptions = struct {
    input_index: usize = 0,
    output_index: usize = 0,
};

fn isGradientDefinition(comptime definition: anytype) bool {
    if (@TypeOf(definition) != type) return false;
    return @hasDecl(definition, "gradient_definition");
}

fn GradientBaseFunctionType(comptime Definition: type) type {
    if (!@hasDecl(Definition, "gradient_definition")) @compileError("expected a grad result type");
    return @TypeOf(Definition.base_function);
}

fn gradientBaseFunction(comptime Definition: type) GradientBaseFunctionType(Definition) {
    return Definition.base_function;
}

fn BaseFunctionType(comptime definition: anytype) type {
    return switch (@typeInfo(@TypeOf(definition))) {
        .@"fn" => @TypeOf(definition),
        .type => GradientBaseFunctionType(definition),
        else => @compileError("expected an HR function or grad result"),
    };
}

fn baseFunction(comptime definition: anytype) BaseFunctionType(definition) {
    return switch (@typeInfo(@TypeOf(definition))) {
        .@"fn" => definition,
        .type => gradientBaseFunction(definition),
        else => unreachable,
    };
}

fn functionInfo(comptime function: anytype) std.builtin.Type.Fn {
    const info = @typeInfo(@TypeOf(function));
    if (info != .@"fn" or info.@"fn".is_var_args) @compileError("compile expects a non-variadic HR function");
    return info.@"fn";
}

fn isHRValue(comptime Value: type) bool {
    return @typeInfo(Value) == .@"struct" and @hasDecl(Value, "hr_value");
}

fn RuntimeValue(comptime Value: type) type {
    if (!isHRValue(Value)) @compileError("compile expects Scalar or Vector function values");
    if (@hasDecl(Value, "row_count")) @compileError("Matrix function inputs and outputs are not supported");
    return if (@hasDecl(Value, "length"))
        @Vector(Value.length, Value.scalar_type)
    else
        Value.scalar_type;
}

fn scalarType(comptime definition: anytype) type {
    const function = baseFunction(definition);
    const info = functionInfo(function);
    if (info.params.len > 0) {
        const Param = info.params[0].type orelse @compileError("compile does not accept anytype parameters");
        _ = RuntimeValue(Param);
        return Param.scalar_type;
    }

    const Return = info.return_type orelse @compileError("compiled function requires an explicit return type");
    if (isHRValue(Return)) return Return.scalar_type;
    const return_info = @typeInfo(Return);
    if (return_info != .@"struct" or !return_info.@"struct".is_tuple or return_info.@"struct".fields.len == 0) {
        @compileError("compiled function must return a Scalar, Vector, or non-empty Tuple");
    }
    return return_info.@"struct".fields[0].type.scalar_type;
}

fn GradientDefinition(comptime source: anytype, comptime options: GradOptions) type {
    const function = baseFunction(source);
    const T = scalarType(source);
    const source_codes = comptime definitionCodes(source);
    const derivative_codes = comptime differentiation.gradIR(
        T,
        &source_codes,
        options.input_index,
        options.output_index,
    );
    return struct {
        pub const gradient_definition = true;
        pub const base_function = function;
        pub const scalar_type = T;
        pub const codes = derivative_codes;
    };
}

fn GradFunctionType(comptime source: anytype, comptime options: GradOptions) type {
    const info = functionInfo(baseFunction(source));
    const Definition = GradientDefinition(source, options);
    const R = hr.InlineResultType(Definition.scalar_type, &Definition.codes);
    if (info.params.len == 0) @compileError("grad requires at least one function parameter");
    if (info.params.len > 8) @compileError("grad currently supports at most 8 function parameters");
    return switch (info.params.len) {
        1 => fn (info.params[0].type.?) R,
        2 => fn (info.params[0].type.?, info.params[1].type.?) R,
        3 => fn (info.params[0].type.?, info.params[1].type.?, info.params[2].type.?) R,
        4 => fn (info.params[0].type.?, info.params[1].type.?, info.params[2].type.?, info.params[3].type.?) R,
        5 => fn (info.params[0].type.?, info.params[1].type.?, info.params[2].type.?, info.params[3].type.?, info.params[4].type.?) R,
        6 => fn (info.params[0].type.?, info.params[1].type.?, info.params[2].type.?, info.params[3].type.?, info.params[4].type.?, info.params[5].type.?) R,
        7 => fn (info.params[0].type.?, info.params[1].type.?, info.params[2].type.?, info.params[3].type.?, info.params[4].type.?, info.params[5].type.?, info.params[6].type.?) R,
        8 => fn (info.params[0].type.?, info.params[1].type.?, info.params[2].type.?, info.params[3].type.?, info.params[4].type.?, info.params[5].type.?, info.params[6].type.?, info.params[7].type.?) R,
        else => unreachable,
    };
}

pub fn grad(comptime source: anytype, comptime options: GradOptions) GradFunctionType(source, options) {
    const info = functionInfo(baseFunction(source));
    const Definition = GradientDefinition(source, options);
    return switch (info.params.len) {
        1 => struct {
            const derivative = Definition;
            fn call(a0: functionInfo(derivative.base_function).params[0].type.?) hr.InlineResultType(derivative.scalar_type, &derivative.codes) {
                return hr.inlineIR(derivative.scalar_type, &derivative.codes, .{a0});
            }
        }.call,
        2 => struct {
            const derivative = Definition;
            fn call(a0: functionInfo(derivative.base_function).params[0].type.?, a1: functionInfo(derivative.base_function).params[1].type.?) hr.InlineResultType(derivative.scalar_type, &derivative.codes) {
                return hr.inlineIR(derivative.scalar_type, &derivative.codes, .{ a0, a1 });
            }
        }.call,
        3 => struct {
            const derivative = Definition;
            fn call(a0: functionInfo(derivative.base_function).params[0].type.?, a1: functionInfo(derivative.base_function).params[1].type.?, a2: functionInfo(derivative.base_function).params[2].type.?) hr.InlineResultType(derivative.scalar_type, &derivative.codes) {
                return hr.inlineIR(derivative.scalar_type, &derivative.codes, .{ a0, a1, a2 });
            }
        }.call,
        4 => struct {
            const derivative = Definition;
            fn call(a0: functionInfo(derivative.base_function).params[0].type.?, a1: functionInfo(derivative.base_function).params[1].type.?, a2: functionInfo(derivative.base_function).params[2].type.?, a3: functionInfo(derivative.base_function).params[3].type.?) hr.InlineResultType(derivative.scalar_type, &derivative.codes) {
                return hr.inlineIR(derivative.scalar_type, &derivative.codes, .{ a0, a1, a2, a3 });
            }
        }.call,
        5 => struct {
            const derivative = Definition;
            fn call(a0: functionInfo(derivative.base_function).params[0].type.?, a1: functionInfo(derivative.base_function).params[1].type.?, a2: functionInfo(derivative.base_function).params[2].type.?, a3: functionInfo(derivative.base_function).params[3].type.?, a4: functionInfo(derivative.base_function).params[4].type.?) hr.InlineResultType(derivative.scalar_type, &derivative.codes) {
                return hr.inlineIR(derivative.scalar_type, &derivative.codes, .{ a0, a1, a2, a3, a4 });
            }
        }.call,
        6 => struct {
            const derivative = Definition;
            fn call(a0: functionInfo(derivative.base_function).params[0].type.?, a1: functionInfo(derivative.base_function).params[1].type.?, a2: functionInfo(derivative.base_function).params[2].type.?, a3: functionInfo(derivative.base_function).params[3].type.?, a4: functionInfo(derivative.base_function).params[4].type.?, a5: functionInfo(derivative.base_function).params[5].type.?) hr.InlineResultType(derivative.scalar_type, &derivative.codes) {
                return hr.inlineIR(derivative.scalar_type, &derivative.codes, .{ a0, a1, a2, a3, a4, a5 });
            }
        }.call,
        7 => struct {
            const derivative = Definition;
            fn call(a0: functionInfo(derivative.base_function).params[0].type.?, a1: functionInfo(derivative.base_function).params[1].type.?, a2: functionInfo(derivative.base_function).params[2].type.?, a3: functionInfo(derivative.base_function).params[3].type.?, a4: functionInfo(derivative.base_function).params[4].type.?, a5: functionInfo(derivative.base_function).params[5].type.?, a6: functionInfo(derivative.base_function).params[6].type.?) hr.InlineResultType(derivative.scalar_type, &derivative.codes) {
                return hr.inlineIR(derivative.scalar_type, &derivative.codes, .{ a0, a1, a2, a3, a4, a5, a6 });
            }
        }.call,
        8 => struct {
            const derivative = Definition;
            fn call(a0: functionInfo(derivative.base_function).params[0].type.?, a1: functionInfo(derivative.base_function).params[1].type.?, a2: functionInfo(derivative.base_function).params[2].type.?, a3: functionInfo(derivative.base_function).params[3].type.?, a4: functionInfo(derivative.base_function).params[4].type.?, a5: functionInfo(derivative.base_function).params[5].type.?, a6: functionInfo(derivative.base_function).params[6].type.?, a7: functionInfo(derivative.base_function).params[7].type.?) hr.InlineResultType(derivative.scalar_type, &derivative.codes) {
                return hr.inlineIR(derivative.scalar_type, &derivative.codes, .{ a0, a1, a2, a3, a4, a5, a6, a7 });
            }
        }.call,
        else => unreachable,
    };
}

fn RuntimeFunctionReturn(comptime function: anytype) type {
    const Return = functionInfo(function).return_type.?;
    if (isHRValue(Return)) return RuntimeValue(Return);
    const info = @typeInfo(Return);
    var types: [info.@"struct".fields.len]type = undefined;
    inline for (info.@"struct".fields, 0..) |field, index| types[index] = RuntimeValue(field.type);
    return @Tuple(&types);
}

fn RuntimeGradientReturn(comptime Definition: type) type {
    const T = Definition.scalar_type;
    const gradient_codes = Definition.codes;
    const Output = ir.OutputType(T, &gradient_codes);
    const fields = @typeInfo(Output).@"struct".fields;
    return if (fields.len == 1) fields[0].type else Output;
}

fn RuntimeReturn(comptime definition: anytype) type {
    return switch (@typeInfo(@TypeOf(definition))) {
        .@"fn" => RuntimeFunctionReturn(definition),
        .type => RuntimeGradientReturn(definition),
        else => @compileError("expected an HR function or grad result"),
    };
}

fn FunctionCodesType(comptime function: anytype) type {
    return @TypeOf(comptime hr.toIRCode(function));
}

fn functionCodes(comptime function: anytype) FunctionCodesType(function) {
    return comptime hr.toIRCode(function);
}

fn GradientCodesType(comptime Definition: type) type {
    return @TypeOf(Definition.codes);
}

fn gradientCodes(comptime Definition: type) GradientCodesType(Definition) {
    return Definition.codes;
}

fn CodesType(comptime definition: anytype) type {
    return switch (@typeInfo(@TypeOf(definition))) {
        .@"fn" => FunctionCodesType(definition),
        .type => GradientCodesType(definition),
        else => @compileError("expected an HR function or grad result"),
    };
}

fn definitionCodes(comptime definition: anytype) CodesType(definition) {
    return switch (@typeInfo(@TypeOf(definition))) {
        .@"fn" => functionCodes(definition),
        .type => gradientCodes(definition),
        else => unreachable,
    };
}

fn CompileType(comptime definition: anytype) type {
    const info = functionInfo(baseFunction(definition));
    const R = RuntimeReturn(definition);
    if (info.params.len > 8) @compileError("compile currently supports at most 8 function parameters");
    return switch (info.params.len) {
        0 => fn () R,
        1 => fn (RuntimeValue(info.params[0].type.?)) R,
        2 => fn (RuntimeValue(info.params[0].type.?), RuntimeValue(info.params[1].type.?)) R,
        3 => fn (RuntimeValue(info.params[0].type.?), RuntimeValue(info.params[1].type.?), RuntimeValue(info.params[2].type.?)) R,
        4 => fn (RuntimeValue(info.params[0].type.?), RuntimeValue(info.params[1].type.?), RuntimeValue(info.params[2].type.?), RuntimeValue(info.params[3].type.?)) R,
        5 => fn (RuntimeValue(info.params[0].type.?), RuntimeValue(info.params[1].type.?), RuntimeValue(info.params[2].type.?), RuntimeValue(info.params[3].type.?), RuntimeValue(info.params[4].type.?)) R,
        6 => fn (RuntimeValue(info.params[0].type.?), RuntimeValue(info.params[1].type.?), RuntimeValue(info.params[2].type.?), RuntimeValue(info.params[3].type.?), RuntimeValue(info.params[4].type.?), RuntimeValue(info.params[5].type.?)) R,
        7 => fn (RuntimeValue(info.params[0].type.?), RuntimeValue(info.params[1].type.?), RuntimeValue(info.params[2].type.?), RuntimeValue(info.params[3].type.?), RuntimeValue(info.params[4].type.?), RuntimeValue(info.params[5].type.?), RuntimeValue(info.params[6].type.?)) R,
        8 => fn (RuntimeValue(info.params[0].type.?), RuntimeValue(info.params[1].type.?), RuntimeValue(info.params[2].type.?), RuntimeValue(info.params[3].type.?), RuntimeValue(info.params[4].type.?), RuntimeValue(info.params[5].type.?), RuntimeValue(info.params[6].type.?), RuntimeValue(info.params[7].type.?)) R,
        else => unreachable,
    };
}

fn evaluateFunction(comptime function: anytype, comptime generated_codes: anytype, inputs: anytype) RuntimeFunctionReturn(function) {
    const outputs = ir.evalIRCode(scalarType(function), &generated_codes, inputs);
    return if (comptime isHRValue(functionInfo(function).return_type.?)) outputs[0] else outputs;
}

fn evaluateGradient(comptime Definition: type, comptime generated_codes: anytype, inputs: anytype) RuntimeGradientReturn(Definition) {
    const outputs = ir.evalIRCode(Definition.scalar_type, &generated_codes, inputs);
    return if (@typeInfo(@TypeOf(outputs)).@"struct".fields.len == 1) outputs[0] else outputs;
}

fn evaluate(comptime definition: anytype, comptime generated_codes: anytype, inputs: anytype) RuntimeReturn(definition) {
    return switch (@typeInfo(@TypeOf(definition))) {
        .@"fn" => evaluateFunction(definition, generated_codes, inputs),
        .type => evaluateGradient(definition, generated_codes, inputs),
        else => unreachable,
    };
}

pub fn compile(comptime source: anytype) CompileType(source) {
    const info = functionInfo(baseFunction(source));
    const generated_codes = comptime definitionCodes(source);
    return switch (info.params.len) {
        0 => struct {
            const definition = source;
            const codes = generated_codes;
            fn call() RuntimeReturn(definition) {
                return evaluate(definition, codes, .{});
            }
        }.call,
        1 => struct {
            const definition = source;
            const codes = generated_codes;
            fn call(a0: RuntimeValue(functionInfo(baseFunction(definition)).params[0].type.?)) RuntimeReturn(definition) {
                return evaluate(definition, codes, .{a0});
            }
        }.call,
        2 => struct {
            const definition = source;
            const codes = generated_codes;
            fn call(a0: RuntimeValue(functionInfo(baseFunction(definition)).params[0].type.?), a1: RuntimeValue(functionInfo(baseFunction(definition)).params[1].type.?)) RuntimeReturn(definition) {
                return evaluate(definition, codes, .{ a0, a1 });
            }
        }.call,
        3 => struct {
            const definition = source;
            const codes = generated_codes;
            fn call(a0: RuntimeValue(functionInfo(baseFunction(definition)).params[0].type.?), a1: RuntimeValue(functionInfo(baseFunction(definition)).params[1].type.?), a2: RuntimeValue(functionInfo(baseFunction(definition)).params[2].type.?)) RuntimeReturn(definition) {
                return evaluate(definition, codes, .{ a0, a1, a2 });
            }
        }.call,
        4 => struct {
            const definition = source;
            const codes = generated_codes;
            fn call(a0: RuntimeValue(functionInfo(baseFunction(definition)).params[0].type.?), a1: RuntimeValue(functionInfo(baseFunction(definition)).params[1].type.?), a2: RuntimeValue(functionInfo(baseFunction(definition)).params[2].type.?), a3: RuntimeValue(functionInfo(baseFunction(definition)).params[3].type.?)) RuntimeReturn(definition) {
                return evaluate(definition, codes, .{ a0, a1, a2, a3 });
            }
        }.call,
        5 => struct {
            const definition = source;
            const codes = generated_codes;
            fn call(a0: RuntimeValue(functionInfo(baseFunction(definition)).params[0].type.?), a1: RuntimeValue(functionInfo(baseFunction(definition)).params[1].type.?), a2: RuntimeValue(functionInfo(baseFunction(definition)).params[2].type.?), a3: RuntimeValue(functionInfo(baseFunction(definition)).params[3].type.?), a4: RuntimeValue(functionInfo(baseFunction(definition)).params[4].type.?)) RuntimeReturn(definition) {
                return evaluate(definition, codes, .{ a0, a1, a2, a3, a4 });
            }
        }.call,
        6 => struct {
            const definition = source;
            const codes = generated_codes;
            fn call(a0: RuntimeValue(functionInfo(baseFunction(definition)).params[0].type.?), a1: RuntimeValue(functionInfo(baseFunction(definition)).params[1].type.?), a2: RuntimeValue(functionInfo(baseFunction(definition)).params[2].type.?), a3: RuntimeValue(functionInfo(baseFunction(definition)).params[3].type.?), a4: RuntimeValue(functionInfo(baseFunction(definition)).params[4].type.?), a5: RuntimeValue(functionInfo(baseFunction(definition)).params[5].type.?)) RuntimeReturn(definition) {
                return evaluate(definition, codes, .{ a0, a1, a2, a3, a4, a5 });
            }
        }.call,
        7 => struct {
            const definition = source;
            const codes = generated_codes;
            fn call(a0: RuntimeValue(functionInfo(baseFunction(definition)).params[0].type.?), a1: RuntimeValue(functionInfo(baseFunction(definition)).params[1].type.?), a2: RuntimeValue(functionInfo(baseFunction(definition)).params[2].type.?), a3: RuntimeValue(functionInfo(baseFunction(definition)).params[3].type.?), a4: RuntimeValue(functionInfo(baseFunction(definition)).params[4].type.?), a5: RuntimeValue(functionInfo(baseFunction(definition)).params[5].type.?), a6: RuntimeValue(functionInfo(baseFunction(definition)).params[6].type.?)) RuntimeReturn(definition) {
                return evaluate(definition, codes, .{ a0, a1, a2, a3, a4, a5, a6 });
            }
        }.call,
        8 => struct {
            const definition = source;
            const codes = generated_codes;
            fn call(a0: RuntimeValue(functionInfo(baseFunction(definition)).params[0].type.?), a1: RuntimeValue(functionInfo(baseFunction(definition)).params[1].type.?), a2: RuntimeValue(functionInfo(baseFunction(definition)).params[2].type.?), a3: RuntimeValue(functionInfo(baseFunction(definition)).params[3].type.?), a4: RuntimeValue(functionInfo(baseFunction(definition)).params[4].type.?), a5: RuntimeValue(functionInfo(baseFunction(definition)).params[5].type.?), a6: RuntimeValue(functionInfo(baseFunction(definition)).params[6].type.?), a7: RuntimeValue(functionInfo(baseFunction(definition)).params[7].type.?)) RuntimeReturn(definition) {
                return evaluate(definition, codes, .{ a0, a1, a2, a3, a4, a5, a6, a7 });
            }
        }.call,
        else => unreachable,
    };
}

test "compile returns a callable function for one Vector input" {
    const S = hr.Scalar(f64);
    const V = hr.Vector(2, f64);
    const model = struct {
        fn call(x: V) S {
            return x.dot(x);
        }
    }.call;
    const function = compile(model);
    try std.testing.expectEqual(@as(f64, 5), function(@Vector(2, f64){ 1, 2 }));
}

test "compile supports multiple mixed inputs and Tuple outputs" {
    const S = hr.Scalar(f32);
    const V = hr.Vector(3, f32);
    const Outputs = @Tuple(&.{ S, V });
    const model = struct {
        fn call(scale: S, input: V, offset: S) Outputs {
            const vector = scale.mul(input);
            return .{ vector.sum().add(offset), vector };
        }
    }.call;
    const function = compile(model);
    const result = function(2, @Vector(3, f32){ 1, 2, 3 }, 5);
    try std.testing.expectEqual(@as(f32, 17), result[0]);
    try std.testing.expectEqual(@Vector(3, f32){ 2, 4, 6 }, result[1]);
}

test "compile supports constant functions and every float type" {
    inline for (.{ f16, f32, f64 }) |T| {
        const S = hr.Scalar(T);
        const model = struct {
            fn call() S {
                return S.init(3);
            }
        }.call;
        const function = compile(model);
        try std.testing.expectEqual(@as(T, 3), function());
    }
}

test "compile supports every direct-call arity from zero through eight" {
    const S = hr.Scalar(f64);
    const Models = struct {
        fn call2(a: S, b: S) S {
            return a.add(b);
        }
        fn call4(a: S, b: S, c: S, d: S) S {
            return a.add(b).add(c).add(d);
        }
        fn call5(a: S, b: S, c: S, d: S, e: S) S {
            return a.add(b).add(c).add(d).add(e);
        }
        fn call6(a: S, b: S, c: S, d: S, e: S, f: S) S {
            return a.add(b).add(c).add(d).add(e).add(f);
        }
        fn call7(a: S, b: S, c: S, d: S, e: S, f: S, g: S) S {
            return a.add(b).add(c).add(d).add(e).add(f).add(g);
        }
        fn call8(a: S, b: S, c: S, d: S, e: S, f: S, g: S, h: S) S {
            return a.add(b).add(c).add(d).add(e).add(f).add(g).add(h);
        }
    };
    try std.testing.expectEqual(@as(f64, 3), compile(Models.call2)(1, 2));
    try std.testing.expectEqual(@as(f64, 10), compile(Models.call4)(1, 2, 3, 4));
    try std.testing.expectEqual(@as(f64, 15), compile(Models.call5)(1, 2, 3, 4, 5));
    try std.testing.expectEqual(@as(f64, 21), compile(Models.call6)(1, 2, 3, 4, 5, 6));
    try std.testing.expectEqual(@as(f64, 28), compile(Models.call7)(1, 2, 3, 4, 5, 6, 7));
    try std.testing.expectEqual(@as(f64, 36), compile(Models.call8)(1, 2, 3, 4, 5, 6, 7, 8));
}

test "compile executes the quadratic function shape from example a" {
    const S = hr.Scalar(f64);
    const V = hr.Vector(2, f64);
    const M = hr.Matrix(2, 2, f64);
    const model = struct {
        fn call(x: V) S {
            const q = M.init(.{
                .{ 1, 0 },
                .{ 0, 2 },
            });
            return q.mul(x).dot(x);
        }
    }.call;
    const function = compile(model);
    try std.testing.expectEqual(@as(f64, 9), function(@Vector(2, f64){ 1, 2 }));
}

test "compile accepts nested grad products for gradient and Hessian" {
    const S = hr.Scalar(f64);
    const V = hr.Vector(2, f64);
    const M = hr.Matrix(2, 2, f64);
    const model = struct {
        fn call(x: V) S {
            const q = M.init(.{
                .{ 1, 0 },
                .{ 0, 2 },
            });
            return q.mul(x).dot(x);
        }
    }.call;

    const value_function = compile(model);
    const gradient_function = compile(grad(model, .{}));
    const hessian_function = compile(grad(grad(model, .{}), .{}));
    const input = @Vector(2, f64){ 1, 2 };
    try std.testing.expectEqual(@as(f64, 9), value_function(input));
    try std.testing.expectEqual(@Vector(2, f64){ 2, 8 }, gradient_function(input));
    const hessian = hessian_function(input);
    try std.testing.expectEqual(@Vector(2, f64){ 2, 0 }, hessian[0]);
    try std.testing.expectEqual(@Vector(2, f64){ 0, 4 }, hessian[1]);
}

test "grad options select logical input and output indices" {
    const S = hr.Scalar(f32);
    const V = hr.Vector(2, f32);
    const Outputs = @Tuple(&.{ S, S });
    const model = struct {
        fn call(x: V, scale: S) Outputs {
            const scaled = x.mul(scale);
            return .{ scaled.sum(), scale.mul(scale) };
        }
    }.call;

    const vector_gradient = compile(grad(model, .{ .input_index = 0, .output_index = 0 }));
    const scalar_gradient = compile(grad(model, .{ .input_index = 1, .output_index = 1 }));
    const input = @Vector(2, f32){ 3, 4 };
    try std.testing.expectEqual(@Vector(2, f32){ 2, 2 }, vector_gradient(input, 2));
    try std.testing.expectEqual(@as(f32, 4), scalar_gradient(input, 2));
}

test "file-level constants work in compiled functions and gradients" {
    const potential = compile(cartPolePotential);
    const gradient_x = compile(grad(cartPolePotential, .{ .input_index = 0 }));
    const gradient_dx = compile(grad(cartPolePotential, .{ .input_index = 1 }));
    const x = @Vector(2, f64){ 0, 0.5 };
    const dx = @Vector(2, f64){ 0, 0 };

    try std.testing.expectApproxEqAbs(9.81 * @cos(0.5), potential(x, dx), 1e-12);
    const actual_x = gradient_x(x, dx);
    try std.testing.expectApproxEqAbs(@as(f64, 0), actual_x[0], 1e-12);
    try std.testing.expectApproxEqAbs(-9.81 * @sin(0.5), actual_x[1], 1e-12);
    try std.testing.expectEqual(@Vector(2, f64){ 0, 0 }, gradient_dx(x, dx));
}
