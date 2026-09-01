const zad = @import("zad");
const Scalar = zad.Scalar(f64);

fn expression(x: *const Scalar, y: *const Scalar) Scalar {
    const log_x = x.log();
    const product = x.mul(y);
    const sin_y = y.sin();
    const value = log_x.add(&product).sub(&sin_y);
    const zero = Scalar.c(0);
    const one = Scalar.c(1);
    return value.add(&zero).mul(&one).neg().neg();
}

const raw_program = zad.compile(f64, expression, .{ .optimize = false });
const optimized_program = zad.compile(f64, expression, .{});

comptime {
    if (raw_program.len <= optimized_program.len) @compileError("assembly benchmark requires IR optimization to reduce the program");
}

export fn handwritten_eval_ptr(values: [*]const f64) f64 {
    return @log(values[0]) + values[0] * values[1] - @sin(values[1]);
}

export fn generated_raw_eval_ptr(values: [*]const f64) f64 {
    return zad.eval(raw_program, .{ values[0], values[1] });
}

export fn generated_optimized_eval_ptr(values: [*]const f64) f64 {
    return zad.eval(optimized_program, .{ values[0], values[1] });
}
