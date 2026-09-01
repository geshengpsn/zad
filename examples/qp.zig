const std = @import("std");
const zad = @import("zad");

const Scalar = zad.Scalar(f64);
const Vec2 = zad.Vec(f64, 2);
const Mat2 = zad.Mat(f64, 2, 2);

fn quadraticProgram(x: *const Vec2) Scalar {
    const q = Mat2.c(.{
        .{ 1.0, 2.0 },
        .{ 2.0, 1.0 },
    });
    const qx = q.matMul(x);
    const xtqx = x.dot(&qx);
    const two = Scalar.c(2.0);
    return xtqx.div(&two);
}

const program = zad.compile(f64, quadraticProgram, .{});
const gradient = zad.grad(f64, program, .{});
const hessian = zad.grad(f64, gradient, .{});

pub fn main() !void {
    const inputs = .{[2]f64{ 1.0, 2.0 }};
    const value = zad.eval(program, inputs);
    const grad_value = zad.eval(gradient, inputs);
    const hessian_value = zad.eval(hessian, inputs);

    if (value != 6.5 or
        !std.mem.eql(f64, &grad_value, &.{ 5.0, 4.0 }) or
        !std.meta.eql(hessian_value, [2][2]f64{ .{ 1.0, 2.0 }, .{ 2.0, 1.0 } }))
    {
        return error.UnexpectedDerivative;
    }

    std.debug.print("Result: {d}\n", .{value});
    std.debug.print("Grad: {any}\n", .{grad_value});
    std.debug.print("Hessian: {any}\n", .{hessian_value});
    std.debug.print("IR nodes: primal={d}, grad={d}, hessian={d}\n", .{ program.len, gradient.len, hessian.len });
}
