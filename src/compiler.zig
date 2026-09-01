const dag = @import("dag.zig");
const ir = @import("ir.zig");
const ir_opt = @import("ir_opt.zig");
const ir_grad = @import("ir_grad.zig");

pub const CompileOptions = ir.CompileOptions;
pub const GradOptions = ir_grad.Options;
pub const GradMode = ir_grad.Mode;
pub const GradSelection = ir_grad.Selection;

fn compiledType(comptime T: type, comptime function: anytype, comptime options: CompileOptions) type {
    const graph = comptime dag.toDag(T, function);
    return @TypeOf(comptime ir.lower(T, graph, options));
}

pub fn compile(
    comptime T: type,
    comptime function: anytype,
    comptime options: CompileOptions,
) compiledType(T, function, options) {
    const graph = comptime dag.toDag(T, function);
    const lowered = comptime ir.lower(T, graph, options);
    return if (options.optimize) comptime ir_opt.optimize(T, lowered) else lowered;
}

fn gradientType(comptime T: type, comptime program: anytype, comptime options: GradOptions) type {
    return @TypeOf(comptime ir_grad.differentiate(T, program, options));
}

pub fn grad(
    comptime T: type,
    comptime program: anytype,
    comptime options: GradOptions,
) gradientType(T, program, options) {
    const differentiated = comptime ir_grad.differentiate(T, program, options);
    return if (options.optimize) comptime ir_opt.optimize(T, differentiated) else differentiated;
}

test "compiler supports homogeneous f16 f32 and f64 programs" {
    const vm = @import("vm.zig");

    const F16 = struct {
        const S = dag.Scalar(f16);
        fn call(x: *const S) S {
            return x.mul(x);
        }
    };
    const F32 = struct {
        const V = dag.Vector(f32, 40);
        fn call(x: *const V, y: *const V) dag.Scalar(f32) {
            return x.dot(y);
        }
    };
    const F64 = struct {
        const V = dag.Vector(f64, 2);
        const M = dag.Matrix(f64, 2, 2);
        fn call(matrix: *const M, x: *const V) V {
            return matrix.matMul(x);
        }
    };
    const F80 = struct {
        const S = dag.Scalar(f80);
        fn call(x: *const S) S {
            return x.mul(x);
        }
    };

    const f16_program = comptime compile(f16, F16.call, .{});
    try @import("std").testing.expectEqual(@as(f16, 9), vm.eval(f16_program, .{@as(f16, 3)}));

    const f32_program = comptime compile(f32, F32.call, .{});
    const f32_vector: [40]f32 = @splat(1);
    try @import("std").testing.expectEqual(@as(f32, 40), vm.eval(f32_program, .{ f32_vector, f32_vector }));

    const f64_program = comptime compile(f64, F64.call, .{ .tensor_backend = .scalar });
    try @import("std").testing.expectEqual(
        [_]f64{ 17, 39 },
        vm.eval(f64_program, .{ [2][2]f64{ .{ 1, 2 }, .{ 3, 4 } }, [2]f64{ 5, 6 } }),
    );

    const f80_program = comptime compile(f80, F80.call, .{ .tensor_backend = .scalar });
    try @import("std").testing.expectEqual(@as(f80, 9), vm.eval(f80_program, .{@as(f80, 3)}));
}

test "public grad preserves IR backend and computes selected derivatives" {
    const vm = @import("vm.zig");
    const S = dag.Scalar(f32);
    const V = dag.Vector(f32, 4);
    const model = struct {
        fn call(x: *const V, scale: *const S) V {
            return x.mul(scale);
        }
    }.call;
    const program = comptime compile(f32, model, .{});
    const derivative = comptime grad(f32, program, .{
        .wrt = .{ .index = 4 },
        .outputs = .all,
    });
    try @import("std").testing.expectEqual(ir.TensorBackend.simd, derivative.tensor_backend);
    try @import("std").testing.expectEqual(
        [_]f32{ 1, 2, 3, 4 },
        vm.eval(derivative, .{ [4]f32{ 1, 2, 3, 4 }, @as(f32, 5) }),
    );
}
