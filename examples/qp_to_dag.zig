const std = @import("std");
const zad = @import("zad");
const Scalar = zad.Scalar(f64);
const Vec2 = zad.Vec(f64, 2);
const Mat2 = zad.Mat(f64, 2, 2);

fn quadraticProgram(x: *const Vec2) Scalar {
    const q = Mat2.c(.{
        1.0, 2.0,
        2.0, 1.0,
    });
    const qx = q.matMul(x);
    const xtqx = x.dot(&qx);
    const two = Scalar.c(2.0);
    return xtqx.div(&two);
}

const qp_raw = zad.to_dag_raw(f64, quadraticProgram);
const qp = zad.to_dag(f64, quadraticProgram);
const qp_grad = zad.grad(f64, &qp);
const qp_hess = zad.grad(f64, &qp_grad.nodes);

pub fn main() void {
    var values = [_]f64{ 1.0, 2.0 };
    const result = zad.eval(f64, &qp, &values)[0];
    const grad = zad.eval(f64, &qp_grad.nodes, &values);
    const hess = zad.eval(f64, &qp_hess.nodes, &values);

    std.debug.print("Result: {}\n", .{result});
    std.debug.print("Grad: {any} shape: {d}x{d}\n", .{ grad, qp_grad.rows, qp_grad.cols });
    std.debug.print("Hess: {any} shape: {d}x{d}\n", .{ hess, qp_hess.rows, qp_hess.cols });
    std.debug.print("QP Raw Nodes: {d}\n", .{qp_raw.len});
    std.debug.print("QP Simplified Nodes: {d}\n", .{qp.len});
    std.debug.print("Grad Nodes: {d}\n", .{qp_grad.nodes.len});
    std.debug.print("Hess Nodes: {d}\n", .{qp_hess.nodes.len});
}
