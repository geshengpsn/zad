const std = @import("std");
const zad = @import("zad");
const builder = zad.Builder;

const qp = blk: {
    var b = builder(f64, 80){};
    const x = b.vec_x(2);
    const Q = b.mat_c(2, 2, &[_]f64{
        1.0, 2.0,
        2.0, 1.0,
    });
    const v1 = b.mat_mul(2, 2, 1, &Q, &x);
    const v2 = b.mat_mul(1, 2, 1, &x, &v1);
    const v3 = b.div(v2[0], b.c(2.0));
    b.output(v3);
    break :blk b.dag();
};

const qp_grad = zad.grad(f64, &qp);
const qp_grad_dag_simplified = zad.simplify(f64, &qp_grad.nodes);

const qp_hess = zad.grad(f64, &qp_grad.nodes);
const qp_hess_dag_simplified = zad.simplify(f64, &qp_hess.nodes);

fn qp_func(values: []const f64) f64 {
    return zad.eval(f64, &qp, values)[0];
}

fn qp_grad_func(values: []const f64) [qp_grad.cols * qp_grad.rows]f64 {
    return zad.eval(f64, &qp_grad_dag_simplified, values);
}

fn qp_hess_func(values: []const f64) [qp_hess.cols * qp_hess.rows]f64 {
    return zad.eval(f64, &qp_hess_dag_simplified, values);
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
    std.debug.print("Hess Simplified: {d}\n", .{qp_hess_dag_simplified.len});
    std.debug.print("Grad Simplified: {d}\n", .{qp_grad_dag_simplified.len});
    for (qp_hess_dag_simplified) |node| {
        std.debug.print("Node: {any}\n", .{node});
    }
}
