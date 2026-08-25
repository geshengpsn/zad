pub const eval = @import("eval.zig").eval;
pub const validate_dag = @import("dag.zig").validate_dag;
pub const Builder = @import("dag_builder.zig").Builder;
pub const grad = @import("grad.zig").grad;
pub const grad_raw = @import("grad.zig").grad_raw;
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
    _ = @import("dag_builder.zig");
    _ = @import("grad.zig");
    _ = @import("simplify.zig");
    _ = @import("graph_builder.zig");
}
