const std = @import("std");
const zad = @import("zad");
const DAGNode = zad.DAGNode;
const eval = zad.eval;
const builder = zad.Builder;

const iterations = 20_000_000;

const graph = blk: {
    var b = builder(f64, 8){};
    const x1 = b.x(0);
    const x2 = b.x(1);
    const v1 = b.log(x1);
    const v2 = b.mul(x1, x2);
    const v3 = b.sin(x2);
    const v4 = b.add(v1, v2);
    const v5 = b.sub(v4, v3);
    b.output(v5);
    break :blk b.dag();
};

fn nowNs(io: std.Io) i96 {
    return std.Io.Clock.Timestamp.now(io, .awake).raw.nanoseconds;
}

noinline fn handwritten_eval(values: []const f64) f64 {
    return @log(values[0]) + values[0] * values[1] - @sin(values[1]);
}

noinline fn generated_eval(values: []const f64) f64 {
    return eval(f64, &graph, values);
}

fn runBench(io: std.Io, comptime name: []const u8, func: *const fn ([]const f64) f64) !void {
    const start = nowNs(io);
    var values = [_]f64{ 2.0, 3.0 };
    var sum: f64 = 0.0;

    for (0..iterations) |i| {
        values[0] = 2.0 + @as(f64, @floatFromInt(i & 1023)) * 0.000001;
        values[1] = 3.0 + @as(f64, @floatFromInt((i >> 10) & 1023)) * 0.000001;
        sum += func(&values);
    }

    const elapsed_ns = nowNs(io) - start;
    std.mem.doNotOptimizeAway(sum);

    const ns_per_iter = @as(f64, @floatFromInt(elapsed_ns)) / @as(f64, @floatFromInt(iterations));
    std.debug.print("{s}: {d:.3} ns/iter ({d} ns total), checksum={d:.6}\n", .{ name, ns_per_iter, elapsed_ns, sum });
}

pub fn main(init: std.process.Init) !void {
    const io = init.io;
    var values = [_]f64{ 2.0, 3.0 };
    const expected = handwritten_eval(&values);
    const actual = generated_eval(&values);
    if (expected != actual) return error.BenchmarkFunctionsDiffer;

    try runBench(io, "handwritten", handwritten_eval);
    try runBench(io, "generated", generated_eval);
}
