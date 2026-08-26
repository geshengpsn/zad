pub const eval = @import("eval.zig").eval;
pub const validate_dag = @import("dag.zig").validate_dag;
const grad_mod = @import("grad.zig");
pub const grad = grad_mod.grad;
pub const GradOptions = grad_mod.GradOptions;
pub const GradMode = grad_mod.GradMode;
pub const GradSelection = grad_mod.GradSelection;
pub const simplify = @import("simplify.zig").simplify;
pub const graph_builder = @import("graph_builder.zig");
pub const GraphNode = graph_builder.Node;
pub const Scalar = graph_builder.Scalar;
pub const Vec = graph_builder.Vec;
pub const Mat = graph_builder.Mat;
pub const to_dag = graph_builder.to_dag;
pub const to_dag_raw = graph_builder.to_dag_raw;
test {
    _ = @import("eval.zig");
    _ = @import("dag.zig");
    _ = @import("grad.zig");
    _ = @import("simplify.zig");
    _ = @import("graph_builder.zig");
}
