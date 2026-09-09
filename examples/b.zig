const zad = @import("zad");
const std = @import("std");

const Scalar = zad.Scalar(f64);
const mass_pole = Scalar.init(1.0);
const mass_cart = Scalar.init(5.0);
const length = Scalar.init(1.0);
const gravity = Scalar.init(9.81);
const h = Scalar.init(0.01);

// cartpole dynamics
fn U(x: zad.Vector(2, f64), dx: zad.Vector(2, f64)) Scalar {
    _ = dx;
    const theta = x.get(1);
    const cos_theta = theta.cos();
    return mass_pole.mul(gravity).mul(length).mul(cos_theta);
}

fn T(x: zad.Vector(2, f64), dx: zad.Vector(2, f64)) Scalar {
    const theta = x.get(1);
    const sin_theta = theta.sin();
    const cos_theta = theta.cos();
    const x_dot = dx.get(0);
    const theta_dot = dx.get(1);
    const half = Scalar.init(0.5);
    const cart = half.mul(mass_cart).mul(x_dot).mul(x_dot);
    var pole_vel = zad.Vector(2, f64).init(.{ 0, 0 });
    pole_vel = pole_vel.set(0, x_dot.add(length.mul(cos_theta).mul(theta_dot)));
    pole_vel = pole_vel.set(1, length.mul(sin_theta).mul(theta_dot).neg());
    const pole = half.mul(mass_pole).mul(pole_vel.dot(pole_vel));
    return cart.add(pole);
}

fn E_def(x: zad.Vector(2, f64), dx: zad.Vector(2, f64)) Scalar {
    return T(x, dx).add(U(x, dx));
}

fn L(x: zad.Vector(2, f64), dx: zad.Vector(2, f64)) Scalar {
    return T(x, dx).sub(U(x, dx));
}

fn Ld_def(x1: zad.Vector(2, f64), x2: zad.Vector(2, f64)) Scalar {
    const dx = x2.sub(x1).div(h);
    const x = x1.add(x2).mul(Scalar.init(0.5));
    return L(x, dx).mul(h);
}

const E = zad.compile(E_def);
const D2DL = zad.grad(Ld_def, .{ .input_index = 1 });
const D1DL = zad.grad(Ld_def, .{ .input_index = 0 });

fn DEL_def(
    x1: zad.Vector(2, f64),
    x2: zad.Vector(2, f64),
    x3: zad.Vector(2, f64),
) zad.Vector(2, f64) {
    return D2DL(x1, x2).add(D1DL(x2, x3));
}

const DEL = zad.compile(DEL_def);

fn DEL_fixed_x1x2(x1: @Vector(2, f64), x2: @Vector(2, f64)) fn (@Vector(2, f64)) @Vector(2, f64) {
    return struct {
        fn call(x3: @Vector(2, f64)) @Vector(2, f64) {
            return DEL(x1, x2, x3);
        }
    }.call;
}

const D3DEL_def = zad.grad(DEL_def, .{ .input_index = 2 });

const D3DEL = zad.compile(D3DEL_def);

// fn D3DEL_fixed_x1x2(x1: @Vector(2, f64), x2: @Vector(2, f64)) fn (@Vector(2, f64)) zla.Mat(2, 2, f64) {
//     return struct {
//         fn call(x3: @Vector(2, f64)) zla.Mat(2, 2, f64) {
//             const result = D3DEL(x1, x2, x3);
//             return zla.Mat(2, 2, f64).init(.{
//                 result[0][0], result[1][0],
//                 result[0][1], result[1][1],
//             });
//         }
//     }.call;
// }

pub fn main() void {
    const x1: @Vector(2, f64) = .{ 0, 0.1 };
    const x2: @Vector(2, f64) = .{ 0, 0.1 };
    const x3: @Vector(2, f64) = .{ 0, 0.1 };
    const result = DEL(x1, x2, x3);
    const fixed_x1x2 = DEL_fixed_x1x2(x1, x2);
    const result2 = fixed_x1x2(x3);

    const d3del = D3DEL(x1, x2, x3);
    std.debug.print("result: {}\n", .{result});
    std.debug.print("result2: {}\n", .{result2});
    std.debug.print("d3del: {}\n", .{d3del});
}
