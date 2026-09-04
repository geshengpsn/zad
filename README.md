# zad

`zad` is a compile-time automatic differentiation library for Zig. User functions build a typed tensor DAG, which is lowered to a homogeneous floating-point IR, optimized, differentiated, and executed by a small virtual machine.

Current version: `0.1.0`. The project targets Zig `0.16.0`.

## Features

- Capacity-free function-style DAG construction.
- Distinct `Scalar`, `Vector`, and `Matrix` semantics in the public graph.
- One homogeneous floating-point type per DAG and IR program.
- Native tensor IR operations including elementwise operations, dot products, matrix-vector multiplication, outer products, and transposed matrix-vector multiplication.
- Compile-time forward- and reverse-mode IR differentiation.
- IR identity rewriting, common-subexpression elimination, and dead-code elimination.
- A dual-stack VM with dedicated scalar and `@Vector` storage.
- SIMD tensor kernels by default, with a scalar backend available.
- First and second derivatives remain tensor IR and can be optimized or differentiated again.

## Quick Start

```zig
const std = @import("std");
const zad = @import("zad");

const Scalar = zad.Scalar(f64);
const Vec2 = zad.Vec(f64, 2);
const Mat2 = zad.Mat(f64, 2, 2);

fn quadraticProgram(x: *const Vec2) Scalar {
    const q = Mat2.c(.{
        .{ 1.0, 2.0 },
        .{ 2.0, 1.0 },
    });
    const qx = q.matMul(x);
    const xtqx = x.dot(&qx);
    const two = Scalar.c(2.0);
    return xtqx.div(&two);
}

const program = zad.compile(f64, quadraticProgram, .{});
const gradient = zad.grad(f64, program, .{});
const hessian = zad.grad(f64, gradient, .{});

pub fn main() void {
    const inputs = .{[2]f64{ 1.0, 2.0 }};

    const value: f64 = zad.eval(program, inputs);
    const grad_value: [2]f64 = zad.eval(gradient, inputs);
    const hessian_value: [2][2]f64 = zad.eval(hessian, inputs);

    std.debug.print("value={d}\n", .{value});
    std.debug.print("gradient={any}\n", .{grad_value});
    std.debug.print("hessian={any}\n", .{hessian_value});
}
```

Run the included version with:

```sh
zig build qp
```

## Architecture

The compilation pipeline is:

```text
typed function
    -> indexed tensor DAG
    -> indexed tensor IR
    -> IR optimization
    -> IR differentiation
    -> IR optimization
    -> VM execution
```

### Typed DAG

The external DAG preserves user-level semantics:

```zig
zad.Scalar(T)
zad.Vector(T, len) // zad.Vec alias
zad.Matrix(T, rows, cols) // zad.Mat alias
```

Function parameters are `*const` graph values. `to_dag` executes the function twice at compile time: a count pass assigns node indices without storage, then a build pass writes an exact-size DAG. Graph values contain stable context-local indices, so shared values and accumulator-style reassignment remain linear and do not require a user capacity.

Inputs are flattened internally in function parameter order. Matrices use row-major storage, but users pass and receive nested arrays through `eval`.

### Tensor IR

Every IR program has one scalar type `T`; mixed f16/f32/f64 programs are rejected at compile time. The tested scalar types are f16, f32, and f64.

IR nodes contain:

- a tensor `Shape`
- a scalar or SIMD `Kernel`
- an SSA operation with references to earlier nodes

The primal operation set includes:

```text
parameter, scalar_constant, tensor_constant
unary, binary, scale, reduce_dot, mat_vec
```

Automatic differentiation can additionally generate:

```text
fill, basis, extract
outer, transpose_mat_vec
```

These operations remain tensor-level. For example, reverse-mode differentiation of:

```text
y = A * x
```

generates:

```text
dA += outer(dy, x)
dx += transpose_mat_vec(A, dy)
```

It does not expand the operation into a scalar DAG.

### IR Optimization

`compile` and `grad` optimize their generated IR by default. The current optimizer performs:

- zero/one identity rewriting
- double-negation elimination
- common-subexpression elimination
- output-rooted dead-code elimination

Disable optimization when inspecting raw IR:

```zig
const raw = zad.compile(f32, model, .{ .optimize = false });
const raw_grad = zad.grad(f32, raw, .{ .optimize = false });
```

### Virtual Machine

The VM uses two internal stacks:

```text
scalar_stack: []T
vector_stack: []@Vector(lanes, T)
```

Scalar nodes and scalar-backend tensors use `scalar_stack`. SIMD Vector and Matrix nodes are packed into `vector_stack`; Matrix rows are padded independently to the logical SIMD width. Intermediate SIMD operations read and write `@Vector` values directly. Array-to-vector packing happens only when loading parameters or tensor constants, and vector-to-array unpacking happens only when exporting outputs.

For a function with `Vector(f32, 2)` and `Matrix(f32, 2, 2)` inputs:

```zig
const result = zad.eval(program, .{
    [2]f32{ 1, 2 },
    [2][2]f32{ .{ 1, 2 }, .{ 3, 4 } },
});
```

The result type is inferred from `program.result_shape`:

```text
Scalar       -> T
Vector(N)    -> [N]T
Matrix(R, C) -> [R][C]T
```

`zad.eval_flat(T, program, inputs)` is available for VM testing, integration with existing flat buffers, and low-level benchmarking.

`eval` uses stack storage for small programs. Programs requiring more than 1 MiB of input plus frame storage must use caller-owned workspace:

```zig
var workspace: zad.Workspace(program) = .{};
const result = zad.eval_with_workspace(program, inputs, &workspace);
```

For large result tensors, place the result in caller-owned storage as well:

```zig
var result: zad.vm.Result(program) = undefined;
zad.eval_into(program, inputs, &workspace, &result);
```

The flat equivalents are `eval_flat_with_workspace` and `eval_flat_into`.

## SIMD Execution

Vector and matrix nodes use SIMD kernels by default. The default logical vector width is 1024 bits:

```text
f16 -> @Vector(64, f16)
f32 -> @Vector(32, f32)
f64 -> @Vector(16, f64)
```

Zig and LLVM may split or combine this logical vector width according to the selected CPU target. Every SIMD tensor uses fixed 1024-bit chunks in `vector_stack`; the final chunk is zero-padded. Tensor kernels clear invalid padding lanes after each operation so values such as `log(0)` or `0 / 0` cannot contaminate later reductions.

Configure compilation with:

```zig
const simd_program = zad.compile(f32, model, .{
    .tensor_backend = .simd,
    .vector_bits = 1024,
});

const scalar_program = zad.compile(f32, model, .{
    .tensor_backend = .scalar,
});
```

Scalar DAG nodes always use scalar storage. The backend option changes Vector and Matrix storage and execution: `.simd` uses `vector_stack`, while `.scalar` keeps every tensor element in `scalar_stack`.

Transcendental vector operations such as `sin`, `log`, and `exp` depend on Zig/LLVM target lowering and may become multiple native vectors or scalar library calls.

## Gradients And Jacobians

`grad` differentiates an IR program, not the external typed DAG:

```zig
const derivative = zad.grad(f32, program, .{
    .wrt = .all,
    .outputs = .all,
    .mode = .auto,
    .optimize = true,
});
```

Selections use flattened scalar indices:

```zig
const grad_x = zad.grad(f32, program, .{
    .wrt = .{ .range = .{ .start = 0, .len = 2 } },
    .outputs = .{ .index = 0 },
});
```

Available selections are:

```text
.all
.{ .index = i }
.{ .range = .{ .start = i, .len = n } }
.{ .indices = &.{ ... } }
```

`.auto` uses forward mode when fewer input components than output components are selected, and reverse mode otherwise. Jacobian values are emitted in row-major order.

The typed result shape is:

```text
1 x 1 -> Scalar
1 x N -> Vector(N)
M x 1 -> Vector(M)
M x N -> Matrix(M, N)
```

Run the two-Vector partial gradient example with:

```sh
zig build partial-grad
```

## Public API

```zig
pub const Scalar = zad.Scalar;
pub const Vector = zad.Vector;
pub const Vec = zad.Vec;
pub const Matrix = zad.Matrix;
pub const Mat = zad.Mat;

pub const to_dag = zad.to_dag;
pub const compile = zad.compile;
pub const grad = zad.grad;
pub const eval = zad.eval;
pub const eval_into = zad.eval_into;
pub const eval_with_workspace = zad.eval_with_workspace;
pub const eval_flat = zad.eval_flat;
pub const eval_flat_into = zad.eval_flat_into;
pub const eval_flat_with_workspace = zad.eval_flat_with_workspace;
pub const Workspace = zad.Workspace;

pub const dag = zad.dag;
pub const ir = zad.ir;
pub const ir_opt = zad.ir_opt;
pub const ir_grad = zad.ir_grad;
pub const vm = zad.vm;
```

## Commands

```sh
zig build test
zig build qp
zig build partial-grad
zig build bench
zig build bench-asm
```

`zig build bench-asm` writes optimized assembly to `zig-out/eval-bench.s`.

## Limitations

- Graph construction and all transformations are compile-time heavy by design.
- SIMD width is a logical IR choice; native instruction width depends on the target and optimizer.
- SIMD reductions may use a different floating-point summation order than the scalar backend.
- Matrix storage is row-major.
- The current matrix primitive is matrix-vector multiplication; matrix-matrix multiplication is not implemented yet.
- Algebraic identity optimization follows ordinary floating-point algebra and can differ at NaN, infinity, signed zero, or singular points.
- Mathematical domain constraints such as `log(x)`, `sqrt(x)`, and division by zero are not checked during graph construction.
