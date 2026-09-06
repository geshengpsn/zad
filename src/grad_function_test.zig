const std = @import("std");
const hr = @import("hr.zig");
const compiler = @import("compile.zig");

const Scalar = hr.Scalar(f64);
const Vec2 = hr.Vector(2, f64);

const mass_pole = Scalar.init(1.0);
const gravity = Scalar.init(9.81);
const pole_length = Scalar.init(1.0);

fn potential(x: Vec2, dx: Vec2) Scalar {
    _ = dx;
    const theta = x.get(1);
    return mass_pole.mul(gravity).mul(pole_length).mul(theta.cos());
}

const potential_gradient = compiler.grad(potential, .{
    .input_index = 0,
    .output_index = 0,
});

const PotentialAndGradient = @Tuple(&.{ Scalar, Vec2 });

fn potentialAndGradient(x: Vec2, dx: Vec2) PotentialAndGradient {
    return .{ potential(x, dx), potential_gradient(x, dx) };
}

test "grad result is callable inside another graph function" {
    const function = compiler.compile(potentialAndGradient);
    const x = @Vector(2, f64){ 0, 0.5 };
    const dx = @Vector(2, f64){ 0, 0 };
    const result = function(x, dx);

    try std.testing.expectApproxEqAbs(9.81 * @cos(0.5), result[0], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0), result[1][0], 1e-12);
    try std.testing.expectApproxEqAbs(-9.81 * @sin(0.5), result[1][1], 1e-12);
}

fn scaleVector(scale: Scalar, vector: Vec2) Vec2 {
    return scale.mul(vector);
}

const scale_derivative = compiler.grad(scaleVector, .{
    .input_index = 0,
    .output_index = 0,
});

fn scaleDerivativeSum(scale: Scalar, vector: Vec2) Scalar {
    return scale_derivative(scale, vector).sum();
}

test "Vector output over Scalar input returns a composable Vector" {
    const function = compiler.compile(scaleDerivativeSum);
    try std.testing.expectEqual(@as(f64, 7), function(3, @Vector(2, f64){ 2, 5 }));
}

fn squareVector(vector: Vec2) Vec2 {
    return vector.mul(vector);
}

const square_jacobian = compiler.grad(squareVector, .{});

fn jacobianTrace(vector: Vec2) Scalar {
    const rows = square_jacobian(vector);
    return rows[0].get(0).add(rows[1].get(1));
}

test "Vector Jacobian rows are usable inside a graph function" {
    const function = compiler.compile(jacobianTrace);
    try std.testing.expectEqual(@as(f64, 14), function(@Vector(2, f64){ 3, 4 }));
}

fn cubic(x: Scalar) Scalar {
    return x.mul(x).mul(x);
}

const cubic_gradient = compiler.grad(cubic, .{});
const cubic_hessian = compiler.grad(cubic_gradient, .{});

fn combinedDerivatives(x: Scalar) Scalar {
    return cubic_gradient(x).add(cubic_hessian(x));
}

test "nested grad results remain composable HR functions" {
    const function = compiler.compile(combinedDerivatives);
    try std.testing.expectEqual(@as(f64, 24), function(2));
}

fn constantGradientUse(x: Scalar) Scalar {
    const constant_derivative = cubic_gradient(Scalar.init(3));
    return constant_derivative.add(x.mul(Scalar.init(0)));
}

test "grad function accepts constant-only arguments inside another graph" {
    const function = compiler.compile(constantGradientUse);
    try std.testing.expectEqual(@as(f64, 27), function(100));
}

const MixedOutputs = @Tuple(&.{ Scalar, Vec2 });

fn mixedOutputs(vector: Vec2, scale: Scalar) MixedOutputs {
    const scaled = vector.mul(scale);
    return .{ scaled.sum(), scaled };
}

const selected_derivative = compiler.grad(mixedOutputs, .{
    .input_index = 1,
    .output_index = 1,
});

fn selectedDerivativeSum(vector: Vec2, scale: Scalar) Scalar {
    return selected_derivative(vector, scale).sum();
}

test "grad function preserves selected input and output semantics" {
    const function = compiler.compile(selectedDerivativeSum);
    try std.testing.expectEqual(@as(f64, 7), function(@Vector(2, f64){ 3, 4 }, 2));
}

fn sumEight(a: Scalar, b: Scalar, c: Scalar, d: Scalar, e: Scalar, f: Scalar, g: Scalar, h: Scalar) Scalar {
    return a.add(b).add(c).add(d).add(e).add(f).add(g).add(h);
}

test "grad callable wrapper supports eight parameters" {
    const derivative = compiler.compile(compiler.grad(sumEight, .{ .input_index = 7 }));
    try std.testing.expectEqual(@as(f64, 1), derivative(1, 2, 3, 4, 5, 6, 7, 8));
}
