const zad = @import("zad");
const DAGNode = zad.DAGNode;
const eval = zad.eval;
const f = zad.f;

const graph = blk: {
    const x1 = f.x(0);
    const x2 = f.x(1);
    const v1 = f.log(&x1);
    const v2 = f.mul(&x1, &x2);
    const v3 = f.sin(&x2);
    const v4 = f.add(&v1, &v2);
    const g = f.sub(&v4, &v3);
    break :blk g.dag();
};

export fn handwritten_eval_ptr(values: [*]const f64) f64 {
    return @log(values[0]) + values[0] * values[1] - @sin(values[1]);
}

export fn generated_eval_ptr(values: [*]const f64) f64 {
    return eval(&graph, values[0..2]);
}
