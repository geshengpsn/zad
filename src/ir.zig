const Code = enum { add, sub, mul, div, sin, cos, log, exp, abs };
const IR = struct {
    a: usize,
    b: usize,
    dim: usize,
    op: Code,
};
fn input_type(comptime ir: []const IR) void {}
// fn eval(ir: []const IR, input: anytype) void {}
