# zad

`zad` is a small Zig library for building scalar expression DAGs, evaluating them, simplifying them at compile time, and generating Jacobian DAGs with forward- and reverse-mode automatic differentiation.

Current version: `0.1.0`.

The core idea is simple: build a computation graph at comptime, transform it at comptime, then evaluate the resulting graph at runtime with plain arrays.

## Features

- Capacity-free compile-time graph construction with typed `Scalar`, `Vec`, and `Mat` values.
- Reusable scalar and linear-algebra graph modules.
- Runtime evaluation with `eval`.
- Forward-, reverse-, and automatically selected differentiation modes with `grad`.
- Automatic DAG simplification in `to_dag` and `grad`, with `to_dag_raw` and a grad option to disable it.
- Jacobian generation for scalar-output and vector-output functions.
- Hessian generation by applying `grad` to a gradient DAG.
- Compile-time simplification passes:
  - dead-code elimination
  - constant folding
  - local algebraic rewrites
  - common subexpression elimination

## Requirements

This project currently targets Zig `0.16.0`.

Run tests:

```sh
zig build test
```

Run the quadratic example:

```sh
zig build qp
```

## Basic Usage

Build a DAG for:

```text
f(x, y) = log(x) + x * y
```

```zig
const std = @import("std");
const zad = @import("zad");

const Scalar = zad.Scalar(f64);

fn f(x: *const Scalar, y: *const Scalar) Scalar {
    const log_x = x.log();
    const product = x.mul(y);
    return log_x.add(&product);
}

const f_dag = zad.to_dag(f64, f);

pub fn main() void {
    var inputs = [_]f64{ 2.0, 3.0 };
    const out = zad.eval(f64, &f_dag, &inputs);
    std.debug.print("f = {d}\n", .{out[0]});
}
```

## Typed Graph Builder

`Scalar`, `Vec`, and `Mat` contain pointer-based graph nodes instead of DAG indexes. Graph construction therefore does not need a capacity, explicit input nodes, or explicit output nodes. `to_dag` receives a function, converts it directly to a DAG, and simplifies the result. Use `to_dag_raw` when the unsimplified graph is required.

Function parameters must be `*const Scalar`, `*const Vec`, or `*const Mat`. Parameter declaration order determines DAG input order. Values inside each parameter are flattened as follows:

- A `Scalar` contributes one input.
- A `Vec(T, n)` contributes `n` inputs in element order.
- A `Mat(T, rows, cols)` contributes `rows * cols` inputs in row-major order.

Graph functions must return exactly one `Scalar` or one `Vec`. A Scalar creates one DAG output; a Vec creates one output per element in order. Returning a Mat, raw node, array, tuple, or arbitrary struct is rejected at compile time.

During conversion, shared pointers are deduplicated and all value nodes are topologically ordered.

Graph values are immutable construction values because operation nodes retain pointers to their operands. Bind intermediate values with `const`; do not reassign a value after another operation references it.

The typed values provide scalar arithmetic, vector addition/subtraction, vector-scalar multiplication, vector dot products, and matrix-vector multiplication:

```zig
const std = @import("std");
const zad = @import("zad");

const Scalar = zad.Scalar(f64);
const Vec2 = zad.Vec(f64, 2);
const Mat2 = zad.Mat(f64, 2, 2);

fn transform(x: *const Vec2, bias: *const Vec2, scale: *const Scalar) Vec2 {
    const matrix = Mat2.c(.{
        1.0, 2.0,
        3.0, 4.0,
    });
    const product = matrix.matMul(x);
    const shifted = product.add(bias);
    return shifted.mul(scale);
}

const f_dag = zad.to_dag(f64, transform);

pub fn main() void {
    // x, bias, scale
    var inputs = [_]f64{ 1.0, 2.0, 10.0, 20.0, 0.5 };
    const outputs = zad.eval(f64, &f_dag, &inputs);
    std.debug.print("result = {any}\n", .{outputs});
}
```

## Gradients And Jacobians

`grad` takes a DAG and a comptime options value, generates the selected Jacobian, and returns a struct containing:

- `rows`: number of selected outputs
- `cols`: number of selected inputs
- `nodes`: a new DAG whose outputs are the Jacobian entries

The Jacobian output order is row-major:

```text
d output_0 / d input_0
d output_0 / d input_1
...
d output_1 / d input_0
d output_1 / d input_1
...
```

Example:

```zig
const g = zad.grad(f64, &f_dag, .{});

var inputs = [_]f64{ 2.0, 3.0 };
const jac = zad.eval(f64, &g.nodes, &inputs);

// For f(x, y) = log(x) + x * y:
// jac[0] = 1 / x + y
// jac[1] = x
```

The default options select every flattened input and output, choose forward or reverse mode from the smaller selected dimension, and simplify the generated DAG:

```zig
.{
    .wrt = .all,
    .outputs = .all,
    .mode = .auto,
    .simplify = true,
}
```

Selections support one index, a continuous range, or an explicit ordered index list:

```zig
const selected = zad.grad(f64, &f_dag, .{
    .wrt = .{ .range = .{ .start = 0, .len = 2 } },
    .outputs = .{ .index = 0 },
    .mode = .reverse,
    .simplify = false,
});
```

Indices refer to flattened scalar inputs and outputs. `Scalar` contributes one input, `Vec(T, n)` contributes `n`, and `Mat(T, rows, cols)` contributes `rows * cols` in row-major order. Selection order determines Jacobian row and column order.

Explicit index lists must contain unique indices. Missing or duplicate selected DAG output indices are rejected at compile time.

`.auto` uses forward mode when fewer inputs than outputs are selected and reverse mode otherwise. `.simplify = false` skips the generic simplifier but retains local zero/one elimination performed while constructing derivatives.

### Partial Gradients

`examples/partial_grad.zig` defines `f(x, y) = x^T y` with two `Vec(f64, 2)` inputs and one Scalar output. Since function inputs are flattened in declaration order, `x` occupies input range `0..2` and `y` occupies `2..4`:

```zig
const grad_x = zad.grad(f64, &function_dag, .{
    .wrt = .{ .range = .{ .start = 0, .len = 2 } },
    .outputs = .{ .index = 0 },
});

const grad_y = zad.grad(f64, &function_dag, .{
    .wrt = .{ .range = .{ .start = 2, .len = 2 } },
    .outputs = .{ .index = 0 },
});
```

Run it with:

```sh
zig build partial-grad
```

## Hessians

A Hessian can be generated by differentiating a gradient DAG:

```zig
const g = zad.grad(f64, &f_dag, .{});
const h = zad.grad(f64, &g.nodes, .{});
```

For a scalar function with `n` inputs, `h.nodes` evaluates to `n * n` outputs in row-major order.

## Quadratic Example

`examples/qp.zig` uses typed graph values and automatic simplification to build the model below.

```text
f(x) = 0.5 * x^T Q x
```

Then it computes:

- the function value
- the gradient
- the Hessian
- simplified DAG sizes

Run it with:

```sh
zig build qp
```

## Assembly Comparison

`benches/asm.zig` builds the same typed graph with `to_dag_raw` and `to_dag`, then exports handwritten, raw generated, and simplified generated evaluators. Emit optimized assembly with:

```sh
zig build bench-asm
```

The result is written to `zig-out/eval-bench.s`. The exported symbols are:

```text
handwritten_eval_ptr
generated_raw_eval_ptr
generated_simplified_eval_ptr
```

For the current benchmark, automatic simplification reduces the DAG from 14 nodes to 8. In the current Apple AArch64 `ReleaseFast` output, LLVM removes multiplication by one and double negation from the raw evaluator, but retains floating-point addition by zero:

```asm
movi d1, #0000000000000000
fadd d0, d0, d1
```

Those instructions are absent from the simplified evaluator. Exact assembly can vary by Zig version, optimization mode, and target.

## Public API

The root module exports:

```zig
pub const eval = @import("eval.zig").eval;
pub const validate_dag = @import("dag.zig").validate_dag;
pub const graph_builder = @import("graph_builder.zig");
pub const GraphNode = graph_builder.Node;
pub const Scalar = graph_builder.Scalar;
pub const Vec = graph_builder.Vec;
pub const Mat = graph_builder.Mat;
pub const to_dag = graph_builder.to_dag;
pub const to_dag_raw = graph_builder.to_dag_raw;
const grad_mod = @import("grad.zig");
pub const grad = grad_mod.grad;
pub const GradOptions = grad_mod.GradOptions;
pub const GradMode = grad_mod.GradMode;
pub const GradSelection = grad_mod.GradSelection;
pub const simplify = @import("simplify.zig").simplify;
```

## DAG Invariants

DAG nodes must only reference earlier nodes.

For commutative binary operations, the project normalizes operand order:

```text
add: lhs <= rhs
mul: lhs <= rhs
```

Non-commutative operations preserve operand order:

```text
sub: lhs - rhs
div: lhs / rhs
```

`validate_dag` checks these invariants.

## Supported Operations

Unary operations:

- `neg`
- `abs`
- `exp`
- `log`
- `sqrt`
- `sin`
- `cos`
- `tan`

Binary operations:

- `add`
- `sub`
- `mul`
- `div`

## Notes And Limitations

- `grad` supports forward, reverse, and automatic mode selection and emits a simplified Jacobian DAG by default.
- `to_dag_raw` preserves the graph before generic simplification. Pass `.simplify = false` to `grad` to preserve its generated DAG before generic simplification.
- The current implementation is compile-time heavy by design.
- Larger graphs may require careful simplification to keep compile times reasonable.
- `grad` currently supports float values and vector-of-float values.
- Mathematical domain issues are not guarded at graph construction time:
  - `log(x)` requires `x > 0`
  - `sqrt(x)` requires `x >= 0`
  - `div(x, y)` requires `y != 0`
  - `abs(x)` is not differentiable at `x = 0`
- `to_dag` and `grad` simplify automatically. Some algebraic rewrites may be unsafe at singular points, such as `x / x = 1`; use `to_dag_raw` or `.simplify = false` respectively when this distinction matters.

## Development

Useful commands:

```sh
zig build test
zig build qp
zig build partial-grad
zig build bench
zig build bench-asm
```

Format modified Zig files with:

```sh
zig fmt <files>
```
