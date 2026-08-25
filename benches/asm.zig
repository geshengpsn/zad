const zad = @import("zad");
const Scalar = zad.Scalar(f64);

fn expression(x: *const Scalar, y: *const Scalar) Scalar {
    const log_x = x.log();
    const product = x.mul(y);
    const sin_y = y.sin();
    const sum = log_x.add(&product);
    const value = sum.sub(&sin_y);

    const zero = Scalar.c(0.0);
    const one = Scalar.c(1.0);
    const with_zero = value.add(&zero);
    const scaled = with_zero.mul(&one);
    const negated = scaled.neg();
    return negated.neg();
}

const raw_graph = zad.to_dag_raw(f64, expression);
const simplified_graph = zad.to_dag(f64, expression);

comptime {
    if (raw_graph.len <= simplified_graph.len) {
        @compileError("assembly benchmark requires simplify to reduce the graph");
    }
}

export fn handwritten_eval_ptr(values: [*]const f64) f64 {
    return @log(values[0]) + values[0] * values[1] - @sin(values[1]);
}

export fn generated_raw_eval_ptr(values: [*]const f64) f64 {
    return zad.eval(f64, &raw_graph, values[0..2])[0];
}

export fn generated_simplified_eval_ptr(values: [*]const f64) f64 {
    return zad.eval(f64, &simplified_graph, values[0..2])[0];
}
