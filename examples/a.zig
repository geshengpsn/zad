const zad = @import("zad");
const std = @import("std");

fn qp_def(x: zad.Vector(2, f64)) zad.Scalar(f64) {
    const Q = zad.Matrix(2, 2, f64).init(.{
        .{ 1, 0 },
        .{ 0, 2 },
    });
    return Q.mul(x).dot(x);
}

const qp = zad.compile(qp_def);
const qp_grad = zad.compile(zad.grad(qp_def, .{}));
const qp_hess = zad.compile(zad.grad(zad.grad(qp_def, .{}), .{}));

pub fn main() void {
    const x = @Vector(2, f64){ 1, 2 };
    const y = qp(x);
    const grad = qp_grad(x);
    const hess = qp_hess(x);
    std.debug.print("y = {}\n", .{y});
    std.debug.print("grad = {}\n", .{grad});
    std.debug.print("hess = {}\n", .{@TypeOf(hess)});
}
