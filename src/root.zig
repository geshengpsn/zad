pub const dag = @import("dag.zig");
pub const ir = @import("ir.zig");
pub const ir_opt = @import("ir_opt.zig");
pub const ir_grad = @import("ir_grad.zig");
pub const vm = @import("vm.zig");
pub const compiler = @import("compiler.zig");

pub const Scalar = dag.Scalar;
pub const Vector = dag.Vector;
pub const Vec = dag.Vector;
pub const Matrix = dag.Matrix;
pub const Mat = dag.Matrix;
pub const to_dag = dag.toDag;

pub const CompileOptions = compiler.CompileOptions;
pub const GradOptions = compiler.GradOptions;
pub const GradMode = compiler.GradMode;
pub const GradSelection = compiler.GradSelection;
pub const compile = compiler.compile;
pub const grad = compiler.grad;
pub const eval = vm.eval;
pub const eval_into = vm.evalInto;
pub const eval_with_workspace = vm.evalWithWorkspace;
pub const eval_flat = vm.evalFlat;
pub const eval_flat_into = vm.evalFlatInto;
pub const eval_flat_with_workspace = vm.evalFlatWithWorkspace;
pub const Workspace = vm.Workspace;

test {
    _ = dag;
    _ = ir;
    _ = ir_opt;
    _ = ir_grad;
    _ = vm;
    _ = compiler;
}
