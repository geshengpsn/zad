# zad

![中文](https://img.shields.io/badge/README-中文-red.svg)

`zad` is a compile-time automatic differentiation library for Zig. Define computations with Scalar, Vector, and Matrix values, then use `compile` to obtain ordinary callable Zig functions.

Current version: `0.2.0`

Requires Zig `0.16.0`.

## Features

- Compile-time graph construction, simplification, and differentiation
- `f16`, `f32`, and `f64` support
- Scalar, Vector, and Matrix operations
- Gradients, Jacobians, and repeated differentiation
- Selectable input and output differentiation
- Callable functions returned by `compile`
- Callable gradient functions that can be reused inside other graph definitions
- Constant folding, CSE, dead-code elimination, and algebraic simplification
- Zig `@Vector` runtime values
- No graph capacity declaration required



## Quick Start

This example defines:

```text
f(x) = x^T Q x
```

```zig
const std = @import("std");
const zad = @import("zad");

const Scalar = zad.Scalar(f64);
const Vec2 = zad.Vector(2, f64);
const Mat2 = zad.Matrix(2, 2, f64);

fn quadratic(x: Vec2) Scalar {
    const q = Mat2.init(.{
        .{ 1, 0 },
        .{ 0, 2 },
    });
    return q.mul(x).dot(x);
}

const f = zad.compile(quadratic);
const gradient = zad.compile(zad.grad(quadratic, .{}));
const hessian = zad.compile(zad.grad(zad.grad(quadratic, .{}), .{}));

pub fn main() void {
    const x = @Vector(2, f64){ 1, 2 };

    const y = f(x);
    const g = gradient(x);
    const h = hessian(x);

    std.debug.print("y = {}\n", .{y});
    std.debug.print("gradient = {}\n", .{g});
    std.debug.print("hessian = {}\n", .{h});
}
```

Expected values:

```text
y = 9
gradient = { 2, 8 }
hessian = .{ { 2, 0 }, { 0, 4 } }
```

A similar executable example is available in `examples/a.zig`.

## Defining Functions

Graph functions may accept any combination of Scalar and Vector arguments:

```zig
const Scalar = zad.Scalar(f32);
const Vec3 = zad.Vector(3, f32);

fn transform(scale: Scalar, input: Vec3, offset: Scalar) Vec3 {
    return scale.mul(input).add(offset);
}
```

A graph function may return:

- one Scalar
- one Vector
- a non-empty Tuple containing Scalar and Vector values

Multiple outputs are represented with a Tuple:

```zig
const Outputs = @Tuple(&.{ Scalar, Vec3 });

fn transformWithSum(scale: Scalar, input: Vec3) Outputs {
    const result = scale.mul(input);
    return .{ result.sum(), result };
}

const compiled = zad.compile(transformWithSum);

const output = compiled(
    2,
    @Vector(3, f32){ 1, 2, 3 },
);

// output[0] == 12
// output[1] == { 2, 4, 6 }
```

`compile` supports 0 to 8 function arguments. Callable functions returned by `grad` support 1 to 8 arguments.

## Scalar

Create constants with `init`:

```zig
const two = Scalar.init(2);
```

Unary operations:

```text
neg, sqrt, exp, log, sin, cos, abs
```

Binary operations:

```text
add, sub, mul, div, atan2
```

Fused multiply-add is also available:

```zig
const result = a.mulAdd(b, c); // a * b + c
```



## Vector

Define a Vector type and constant value:

```zig
const Vec4 = zad.Vector(4, f32);
const value = Vec4.init(.{ 1, 2, 3, 4 });
```

Vector supports:

- element-wise `add`, `sub`, `mul`, `div`, and `atan2`
- Scalar-to-Vector broadcasting
- `neg`, `sqrt`, `exp`, `log`, `sin`, `cos`, and `abs`
- `sum`
- `dot`
- `get`
- `set`
- `mulAdd`

```zig
const first = vector.get(0);
const updated = vector.set(1, first);
const total = updated.sum();
```

`set` returns a new Vector and does not mutate its source.

## Matrix

Matrix is currently intended for constants and intermediate linear algebra inside graph functions:

```zig
const Mat2 = zad.Matrix(2, 2, f64);

const matrix = Mat2.init(.{
    .{ 1, 2 },
    .{ 3, 4 },
});

const result = matrix.mul(vector);
```

Matrix supports:

- `init`
- Matrix addition and subtraction
- Scalar scaling
- Matrix-Vector multiplication

Matrix cannot currently be used as a graph function input or output.

## Compiling Functions

`compile` constructs and optimizes the graph at compile time and returns an ordinary Zig function:

```zig
const function = zad.compile(definition);
const result = function(arguments...);
```

Runtime type mapping:

```text
zad.Scalar(T)    -> T
zad.Vector(N, T) -> @Vector(N, T)
Tuple            -> corresponding runtime Tuple
```

No builder, allocator, workspace, or graph object is required at runtime.

## Automatic Differentiation

`grad` accepts a graph function or another function returned by `grad`:

```zig
const first = zad.grad(function, .{});
const second = zad.grad(first, .{});

const gradient = zad.compile(first);
const hessian = zad.compile(second);
```

Gradient functions are regular HR functions and can be called inside another graph definition:

```zig
const gradientFunction = zad.grad(function, .{});

fn gradientEnergy(x: Vec2) Scalar {
    const gradient = gradientFunction(x);
    return gradient.dot(gradient);
}

const compiled = zad.compile(gradientEnergy);
```

This makes gradients, Jacobian rows, and higher-order derivatives reusable graph modules.

By default, `grad` differentiates the first output with respect to the first input. Select logical input and output positions with options:

```zig
const derivative = zad.grad(function, .{
    .input_index = 1,
    .output_index = 0,
});

const compiledDerivative = zad.compile(derivative);
```

Indices refer to function argument and output Tuple positions, not Vector lanes.

Derivative result types:

```text
Scalar output / Scalar input -> Scalar
Scalar output / Vector input -> Vector gradient
Vector output / Scalar input -> Vector derivative
Vector output / Vector input -> Tuple of Vector Jacobian rows
```



## File-Level Constants

Scalar, Vector, and Matrix constants may be declared outside graph functions:

```zig
const Scalar = zad.Scalar(f64);
const Vec2 = zad.Vector(2, f64);

const mass = Scalar.init(1.0);
const gravity = Scalar.init(9.81);

fn potential(x: Vec2) Scalar {
    const theta = x.get(1);
    return mass.mul(gravity).mul(theta.cos());
}
```

File-level constants participate normally in graph construction, simplification, compilation, and differentiation.

## IR Simplification

IR is simplified while it is generated and once more before it is returned. Current rules include:

- constant folding
- common-subexpression elimination (CSE)
- dead-code elimination
- `neg(neg(x)) -> x`
- `abs(abs(x)) -> abs(x)`
- `cos(neg(x)) -> cos(x)`
- `log(exp(x)) -> x`
- `x + 0 -> x`
- `x - 0 -> x`
- `x - x -> 0`
- `x * 0 -> 0`
- `x * 1 -> x`
- `x * -1 -> neg(x)`
- `x / 1 -> x`
- `get(set(v, i, x), i) -> x`
- `set(v, i, get(v, i)) -> v`

The same rules are applied to IR generated by automatic differentiation.

## Low-Level Differentiation

Most users only need `compile` and `grad`. Direct IR differentiation is available through:

```zig
const source = zad.toIRCode(function);
const derivative = zad.gradIR(T, &source, input_index, output_index);
const value = zad.evalIRCode(T, &derivative, inputs);
```

`gradIR` returns another IR value that can be passed to `gradIR` again, simplified with `simplifyIR`, or evaluated with `evalIRCode`.

## Build And Test

Run all tests:

```sh
zig build test
```

Compile every executable under `examples/`:

```sh
zig build examples
```

Compiled examples are installed in:

```text
zig-out/bin/
```

Run the current example:

```sh
./zig-out/bin/a
```



## Current Limitations

- Only f16, f32, and f64 are supported
- All values in one graph function must use the same floating-point type
- `compile` supports at most 8 direct-call arguments
- Callable gradient functions support 1 to 8 arguments
- Vector length must be at least 2; use Scalar for one value
- Matrix cannot currently be used as a function input or output
- Matrix row and column counts must both be at least 2
- Vector/Vector Jacobians are returned as Tuples of Vector rows
- `abs` is not differentiable at zero
- Domains for `log`, `sqrt`, and division are not checked during graph construction
- Algebraic rules such as `x * 0 -> 0` and `x - x -> 0` may change NaN, infinity, and signed-zero behavior
