const std = @import("std");
const zad = @import("zad");

const Scalar = zad.Scalar(f64);
const Vec2 = zad.Vec(f64, 2);

fn dotProduct(x: *const Vec2, y: *const Vec2) Scalar {
    return x.dot(y);
}

const function_dag = zad.to_dag(f64, dotProduct);

// Function inputs are flattened in declaration order:
// x -> input indices 0..2, y -> input indices 2..4.
const grad_x = zad.grad(f64, &function_dag, .{
    .wrt = .{ .range = .{ .start = 0, .len = 2 } },
    .outputs = .{ .index = 0 },
});

const grad_y = zad.grad(f64, &function_dag, .{
    .wrt = .{ .range = .{ .start = 2, .len = 2 } },
    .outputs = .{ .index = 0 },
});

pub fn main() !void {
    var inputs = [_]f64{
        1.0, 2.0, // x
        3.0, 4.0, // y
    };

    const value = zad.eval(f64, &function_dag, &inputs)[0];
    const partial_x = zad.eval(f64, &grad_x.nodes, &inputs);
    const partial_y = zad.eval(f64, &grad_y.nodes, &inputs);

    if (value != 11.0 or
        !std.mem.eql(f64, &partial_x, &.{ 3.0, 4.0 }) or
        !std.mem.eql(f64, &partial_y, &.{ 1.0, 2.0 }))
    {
        return error.UnexpectedDerivative;
    }

    std.debug.print("f(x, y): {d}\n", .{value});
    std.debug.print("partial f / partial x: {any}\n", .{partial_x});
    std.debug.print("partial f / partial y: {any}\n", .{partial_y});
}
