const ir = @import("ir.zig");
const hr = @import("hr.zig");
const simplify = @import("simplify.zig");
const differentiation = @import("grad.zig");
const compiler = @import("compile.zig");

pub const Scalar = hr.Scalar;
pub const Vector = hr.Vector;
pub const Matrix = hr.Matrix;
pub const toIRCode = hr.toIRCode;
pub const evalIRCode = ir.evalIRCode;
pub const IRBuilder = simplify.Builder;
pub const simplifyIR = simplify.simplifyIR;
pub const gradIR = differentiation.gradIR;
pub const compile = compiler.compile;
pub const grad = compiler.grad;
pub const GradOptions = compiler.GradOptions;

test "all" {
    _ = ir;
    _ = hr;
    _ = simplify;
    _ = differentiation;
    _ = compiler;
    _ = @import("grad_function_test.zig");
}
