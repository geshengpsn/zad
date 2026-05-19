const dag_mod = @import("dag.zig");
const output_size = dag_mod.output_size;
const DAGNode = dag_mod.DAGNode;
const std = @import("std");

pub fn eval(comptime T: type, comptime dag: []const DAGNode(T), inputs: []const T) [output_size(T, dag)]T {
    var frame: [dag.len]T = undefined;
    var outputs: [output_size(T, dag)]T = undefined;
    inline for (dag, 0..) |node, i| {
        switch (node) {
            .scalar_constant => |v| {
                frame[i] = v;
            },
            .scalar_parameter => |index| {
                frame[i] = inputs[index];
            },
            .op1 => |op| {
                frame[i] = switch (op.op) {
                    .neg => -frame[op.node],
                    .abs => @abs(frame[op.node]),
                    .exp => @exp(frame[op.node]),
                    .log => @log(frame[op.node]),
                    .sqrt => @sqrt(frame[op.node]),
                    .sin => @sin(frame[op.node]),
                    .cos => @cos(frame[op.node]),
                    .tan => @tan(frame[op.node]),
                };
            },
            .op2 => |op| {
                frame[i] = switch (op.op) {
                    .add => frame[op.lhs] + frame[op.rhs],
                    .sub => frame[op.lhs] - frame[op.rhs],
                    .mul => frame[op.lhs] * frame[op.rhs],
                    .div => frame[op.lhs] / frame[op.rhs],
                };
            },
            .output => |out| {
                outputs[out.index] = frame[out.node];
            },
        }
    }
    return outputs;
}

fn get_test_dag(comptime T: type) [8]DAGNode(T) {
    return .{
        DAGNode(T){ .scalar_parameter = 0 },
        DAGNode(T){ .op1 = .{ .node = 0, .op = .log } },
        DAGNode(T){ .scalar_parameter = 1 },
        DAGNode(T){ .op2 = .{ .lhs = 0, .rhs = 2, .op = .mul } },
        DAGNode(T){ .op2 = .{ .lhs = 1, .rhs = 3, .op = .add } },
        DAGNode(T){ .op1 = .{ .node = 2, .op = .sin } },
        DAGNode(T){ .op2 = .{ .lhs = 4, .rhs = 5, .op = .sub } },
        DAGNode(T){ .output = .{ .index = 0, .node = 6 } },
    };
}

test "eval dag" {
    const test_funcs = struct {
        fn f(x: []f64) f64 {
            return eval(f64, &get_test_dag(f64), x)[0];
        }

        fn handwritten_f(x: []f64) f64 {
            return @log(x[0]) + x[0] * x[1] - @sin(x[1]);
        }
    };

    var input = [_]f64{ 2.0, 3.0 };
    try std.testing.expectEqual(test_funcs.f(&input), test_funcs.handwritten_f(&input));
}

test "eval vector" {
    const test_funcs = struct {
        fn f(x: []@Vector(2, f64)) @Vector(2, f64) {
            return eval(@Vector(2, f64), &get_test_dag(@Vector(2, f64)), x)[0];
        }

        fn handwritten_f(x: []@Vector(2, f64)) @Vector(2, f64) {
            return @log(x[0]) + x[0] * x[1] - @sin(x[1]);
        }
    };
    var input = [_]@Vector(2, f64){ .{ 2.0, 3.0 }, .{ 4.0, 5.0 } };
    try std.testing.expectEqual(test_funcs.f(&input), test_funcs.handwritten_f(&input));
}
