pub const eval = @import("eval.zig").eval;
pub const validate_dag = @import("dag.zig").validate_dag;
pub const Builder = @import("dag_builder.zig").Builder;
pub const grad = @import("grad.zig").grad;
pub const simplify = @import("simplify.zig").simplify;
test {
    _ = @import("eval.zig");
    _ = @import("dag.zig");
    _ = @import("dag_builder.zig");
    _ = @import("grad.zig");
    _ = @import("simplify.zig");
}
