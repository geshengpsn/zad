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

test "cartpole DEL Jacobian composes nested gradients" {
    const CartPole = struct {
        const mass_cart = Scalar.init(5.0);
        const h = Scalar.init(0.01);

        fn kinetic(x: Vec2, dx: Vec2) Scalar {
            const theta = x.get(1);
            const sin_theta = theta.sin();
            const cos_theta = theta.cos();
            const x_dot = dx.get(0);
            const theta_dot = dx.get(1);
            const half = Scalar.init(0.5);
            const cart = half.mul(mass_cart).mul(x_dot).mul(x_dot);
            var pole_vel = Vec2.init(.{ 0, 0 });
            pole_vel = pole_vel.set(0, x_dot.add(pole_length.mul(cos_theta).mul(theta_dot)));
            pole_vel = pole_vel.set(1, pole_length.mul(sin_theta).mul(theta_dot).neg());
            const pole = half.mul(mass_pole).mul(pole_vel.dot(pole_vel));
            return cart.add(pole);
        }

        fn lagrangian(x: Vec2, dx: Vec2) Scalar {
            return kinetic(x, dx).sub(potential(x, dx));
        }

        fn discreteLagrangian(x1: Vec2, x2: Vec2) Scalar {
            const dx = x2.sub(x1).div(h);
            const x = x1.add(x2).mul(Scalar.init(0.5));
            return lagrangian(x, dx).mul(h);
        }

        const d2_ld = compiler.grad(discreteLagrangian, .{ .input_index = 1 });
        const d1_ld = compiler.grad(discreteLagrangian, .{ .input_index = 0 });

        fn del(x1: Vec2, x2: Vec2, x3: Vec2) Vec2 {
            return d2_ld(x1, x2).add(d1_ld(x2, x3));
        }
    };

    const del_function = compiler.compile(CartPole.del);
    const d3_del = compiler.compile(compiler.grad(CartPole.del, .{ .input_index = 2 }));
    const cases = [2][3]@Vector(2, f64){
        .{ .{ 0, 0.1 }, .{ 0, 0.1 }, .{ 0, 0.1 } },
        .{ .{ -0.03, 0.08 }, .{ 0.02, 0.12 }, .{ 0.05, 0.14 } },
    };
    for (cases, 0..) |points, case_index| {
        const x1 = points[0];
        const x2 = points[1];
        const x3 = points[2];
        const jacobian = d3_del(x1, x2, x3);
        const u = (x3[0] - x2[0]) / 0.01;
        const w = (x3[1] - x2[1]) / 0.01;
        const c = @cos((x2[1] + x3[1]) / 2);
        const s = @sin((x2[1] + x3[1]) / 2);
        const expected: [2][2]f64 = if (case_index == 0)
            .{
                .{ -600, -99.50041652780259 },
                .{ -99.50041652780259, -99.97559752284656 },
            }
        else
            .{
                .{ -600, -c / 0.01 + s * w / 2 },
                .{ -c / 0.01 - s * w / 2, -100 + 0.01 * c * (9.81 - u * w) / 4 },
            };

        const epsilon = 1e-6;
        inline for (0..2) |column| {
            var plus = x3;
            var minus = x3;
            plus[column] += epsilon;
            minus[column] -= epsilon;
            const del_plus = del_function(x1, x2, plus);
            const del_minus = del_function(x1, x2, minus);
            inline for (0..2) |row| {
                try std.testing.expectApproxEqAbs(expected[row][column], jacobian[row][column], 1e-9);
                const difference = (del_plus[row] - del_minus[row]) / (2 * epsilon);
                try std.testing.expectApproxEqAbs(difference, jacobian[row][column], 1e-5);
            }
        }
    }
}
