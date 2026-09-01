const std = @import("std");
const zad = @import("zad");

const Scalar = zad.Scalar(f64);
const iterations = 20_000_000;

fn expression(x: *const Scalar, y: *const Scalar) Scalar {
    const log_x = x.log();
    const product = x.mul(y);
    const sin_y = y.sin();
    return log_x.add(&product).sub(&sin_y);
}

const program = zad.compile(f64, expression, .{});

fn nowNs(io: std.Io) i96 {
    return std.Io.Clock.Timestamp.now(io, .awake).raw.nanoseconds;
}

noinline fn handwrittenEval(values: []const f64) f64 {
    return @log(values[0]) + values[0] * values[1] - @sin(values[1]);
}

noinline fn generatedEval(values: []const f64) f64 {
    return zad.eval(program, .{ values[0], values[1] });
}

fn runBench(io: std.Io, comptime name: []const u8, function: *const fn ([]const f64) f64) void {
    const start = nowNs(io);
    var values = [_]f64{ 2.0, 3.0 };
    var sum: f64 = 0.0;
    for (0..iterations) |index| {
        values[0] = 2.0 + @as(f64, @floatFromInt(index & 1023)) * 0.000001;
        values[1] = 3.0 + @as(f64, @floatFromInt((index >> 10) & 1023)) * 0.000001;
        sum += function(&values);
    }
    const elapsed_ns = nowNs(io) - start;
    std.mem.doNotOptimizeAway(sum);
    const ns_per_iter = @as(f64, @floatFromInt(elapsed_ns)) / @as(f64, @floatFromInt(iterations));
    std.debug.print("{s}: {d:.3} ns/iter ({d} ns total), checksum={d:.6}\n", .{ name, ns_per_iter, elapsed_ns, sum });
}

pub fn main(init: std.process.Init) !void {
    const io = init.io;
    var values = [_]f64{ 2.0, 3.0 };
    if (handwrittenEval(&values) != generatedEval(&values)) return error.BenchmarkFunctionsDiffer;
    runBench(io, "handwritten", handwrittenEval);
    runBench(io, "generated-vm", generatedEval);
}
