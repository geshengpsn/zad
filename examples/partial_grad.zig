const std = @import("std");
const zad = @import("zad");

const Scalar = zad.Scalar(f32);
const Vec2 = zad.Vec(f32, 2);

fn dotProduct(x: *const Vec2, y: *const Vec2) Scalar {
    return x.dot(y);
}

const program = zad.compile(f32, dotProduct, .{});
const grad_x = zad.grad(f32, program, .{
    .wrt = .{ .range = .{ .start = 0, .len = 2 } },
    .outputs = .{ .index = 0 },
});
const grad_y = zad.grad(f32, program, .{
    .wrt = .{ .range = .{ .start = 2, .len = 2 } },
    .outputs = .{ .index = 0 },
});

pub fn main() !void {
    const inputs = .{ [2]f32{ 1, 2 }, [2]f32{ 3, 4 } };
    const value = zad.eval(program, inputs);
    const partial_x = zad.eval(grad_x, inputs);
    const partial_y = zad.eval(grad_y, inputs);

    if (value != 11 or
        !std.mem.eql(f32, &partial_x, &.{ 3, 4 }) or
        !std.mem.eql(f32, &partial_y, &.{ 1, 2 }))
    {
        return error.UnexpectedDerivative;
    }

    std.debug.print("f(x, y): {d}\n", .{value});
    std.debug.print("partial f / partial x: {any}\n", .{partial_x});
    std.debug.print("partial f / partial y: {any}\n", .{partial_y});
}
