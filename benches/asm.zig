const zad = @import("zad");
const eval = zad.eval;
const builder = zad.Builder;

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

export fn handwritten_eval_ptr(values: [*]const f64) f64 {
    return @log(values[0]) + values[0] * values[1] - @sin(values[1]);
}

export fn generated_eval_ptr(values: [*]const f64) f64 {
    return eval(f64, &graph, values[0..2]);
}
