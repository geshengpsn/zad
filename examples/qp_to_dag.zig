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

const qp = zad.to_dag(f64, quadraticProgram);
const qp_grad = zad.grad(f64, &qp);
const qp_hess = zad.grad(f64, &qp_grad.nodes);

fn qp_func(values: []const f64) f64 {
    return zad.eval(f64, &qp, values)[0];
}

fn qp_grad_func(values: []const f64) [qp_grad.cols * qp_grad.rows]f64 {
    return zad.eval(f64, &qp_grad.nodes, values);
}

fn qp_hess_func(values: []const f64) [qp_hess.cols * qp_hess.rows]f64 {
    return zad.eval(f64, &qp_hess.nodes, values);
}

pub fn main() void {
    var values = [_]f64{ 1.0, 2.0 };
    const result = qp_func(&values);
    const grad = qp_grad_func(&values);
    std.debug.print("Result: {}\n", .{result});
    std.debug.print("Grad: {any} shape: {d}x{d}\n", .{ grad, qp_grad.rows, qp_grad.cols });
    const hess = qp_hess_func(&values);
    std.debug.print("Hess: {any} shape: {d}x{d}\n", .{ hess, qp_hess.rows, qp_hess.cols });
    std.debug.print("Hess Nodes: {d}\n", .{qp_hess.nodes.len});
    std.debug.print("Grad Nodes: {d}\n", .{qp_grad.nodes.len});
    std.debug.print("QP Nodes: {d}\n", .{qp.len});
    for (qp_hess.nodes) |node| {
        std.debug.print("Node: {any}\n", .{node});
    }
}
