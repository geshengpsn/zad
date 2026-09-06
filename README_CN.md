# zad

[![English](https://img.shields.io/badge/README-English-blue.svg)](README.md)

`zad` 是一个面向 Zig 的编译时自动微分库。你可以使用接近普通数学代码的方式定义 Scalar、Vector 和 Matrix 运算，再通过 `compile` 得到可直接调用的 Zig 函数。

当前版本：`0.2.0`

要求：Zig `0.16.0`

## 特性

- 在编译期构建、简化和微分计算图
- 支持 `f16`、`f32` 和 `f64`
- 支持 Scalar、Vector 和 Matrix 语义
- 支持一阶梯度、Jacobian 和重复微分
- 支持选择指定 input 与 output 进行微分
- `compile` 返回可直接调用的 Zig 函数
- 自动执行常量折叠、CSE、死代码清除和代数简化
- Vector 使用 Zig `@Vector` 作为运行时类型
- 不需要预先指定节点数量或 capacity

## 快速开始

下面定义二次函数：

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

结果：

```text
y = 9
gradient = { 2, 8 }
hessian = .{ { 2, 0 }, { 0, 4 } }
```

同类完整示例位于 `examples/a.zig`。

## 定义函数

函数输入可以包含任意组合的 Scalar 和 Vector：

```zig
const Scalar = zad.Scalar(f32);
const Vec3 = zad.Vector(3, f32);

fn transform(scale: Scalar, input: Vec3, offset: Scalar) Vec3 {
    return scale.mul(input).add(offset);
}
```

函数可以返回：

- 一个 Scalar
- 一个 Vector
- 由 Scalar 和 Vector 组成的非空 Tuple

多输出示例：

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

`compile` 当前支持 0 到 8 个函数参数；可调用的 `grad` 结果支持 1 到 8 个函数参数。

## Scalar

创建常量：

```zig
const two = Scalar.init(2);
```

支持的一元操作：

```text
neg, sqrt, exp, log, sin, cos, abs
```

支持的二元操作：

```text
add, sub, mul, div, atan2
```

还支持融合乘加：

```zig
const result = a.mulAdd(b, c); // a * b + c
```

## Vector

定义类型和常量：

```zig
const Vec4 = zad.Vector(4, f32);
const value = Vec4.init(.{ 1, 2, 3, 4 });
```

Vector 支持：

- 逐元素 `add`、`sub`、`mul`、`div` 和 `atan2`
- Scalar 与 Vector 广播运算
- `neg`、`sqrt`、`exp`、`log`、`sin`、`cos` 和 `abs`
- `sum`
- `dot`
- `get`
- `set`
- `mulAdd`

示例：

```zig
const first = vector.get(0);
const updated = vector.set(1, first);
const total = updated.sum();
```

`set` 返回一个新的 Vector，不修改原值。

## Matrix

Matrix 当前主要用于函数内部的线性代数计算：

```zig
const Mat2 = zad.Matrix(2, 2, f64);

const matrix = Mat2.init(.{
    .{ 1, 2 },
    .{ 3, 4 },
});

const result = matrix.mul(vector);
```

Matrix 支持：

- `init`
- Matrix 加法和减法
- Scalar 缩放
- Matrix 与 Vector 相乘

Matrix 暂时不能作为用户函数的 input 或 output，但可以在函数内部作为常量和中间计算模块。

## 编译函数

`compile` 在编译期完成计算图生成和优化，并返回普通 Zig 函数：

```zig
const function = zad.compile(definition);
const result = function(arguments...);
```

运行时类型映射：

```text
zad.Scalar(T)      -> T
zad.Vector(N, T)   -> @Vector(N, T)
Tuple              -> 对应的运行时 Tuple
```

因此运行阶段不需要传入 builder、allocator、workspace 或图对象。

## 自动微分

`grad` 接收 Zig 函数或另一个 `grad` 的结果：

```zig
const first = zad.grad(function, .{});
const second = zad.grad(first, .{});

const gradient = zad.compile(first);
const hessian = zad.compile(second);
```

`grad` 的结果本身也是一个 HR 函数，可以直接在另一个计算图函数中调用：

```zig
const gradientFunction = zad.grad(function, .{});

fn gradientEnergy(x: Vec2) Scalar {
    const gradient = gradientFunction(x);
    return gradient.dot(gradient);
}

const compiled = zad.compile(gradientEnergy);
```

这允许将梯度、Jacobian 行和高阶导数作为普通可复用计算模块继续组合。

默认对第一个 input 和第一个 output 微分。

可以通过 options 选择逻辑 input 和 output：

```zig
const derivative = zad.grad(function, .{
    .input_index = 1,
    .output_index = 0,
});

const compiledDerivative = zad.compile(derivative);
```

索引对应函数参数和返回 Tuple 的位置，而不是 Vector 内部的 lane。

导数返回类型：

```text
Scalar / Scalar -> Scalar
Scalar / Vector -> Vector gradient
Vector / Scalar -> Vector derivative
Vector / Vector -> Tuple of Vector Jacobian rows
```

## IR 简化

IR 在生成过程中自动简化，最终输出前还会执行一次完整简化。

当前包括：

- 常量折叠
- 公共子表达式消除（CSE）
- 死代码清除
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

这些规则也会应用于自动微分产生的 IR。

## 底层接口

普通用户只需要 `compile` 和 `grad`。需要直接处理 IR 时可以使用：

```zig
zad.gradIR(T, ir, input_index, output_index)
```

`gradIR` 根据逻辑 input/output index 生成新的导数 IR。生成结果仍然可以继续微分和编译。

## 构建与测试

运行全部测试：

```sh
zig build test
```

编译 `examples/` 下的所有可执行示例：

```sh
zig build examples
```

编译结果位于：

```text
zig-out/bin/
```

运行当前示例：

```sh
./zig-out/bin/a
```

## 当前限制

- 仅支持 f16、f32 和 f64
- 一个函数中的所有值必须使用同一种浮点类型
- `compile` 最多支持 8 个直接调用参数
- Vector 长度至少为 2；单个值使用 Scalar
- Matrix 暂时不能作为函数 input 或 output
- Matrix 的行数和列数目前都必须至少为 2
- Vector/Vector Jacobian 返回 Vector Tuple，暂未提供专用 Matrix 返回类型
- `abs` 在零点不可微
- `log`、`sqrt` 和除法的定义域不会在构图时检查
- `x * 0 -> 0`、`x - x -> 0` 等代数规则可能改变 NaN、Inf 和 signed zero 行为
