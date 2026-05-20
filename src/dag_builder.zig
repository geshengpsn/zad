const dag_mod = @import("dag.zig");
const DAGNode = dag_mod.DAGNode;
const validate_dag = dag_mod.validate_dag;
const Op1 = dag_mod.Op1;
const Op2 = dag_mod.Op2;

pub fn Builder(comptime T: type, comptime capacity: usize) type {
    return struct {
        nodes: [capacity]DAGNode(T) = undefined,
        len: usize = 0,
        param_count: usize = 0,
        output_count: usize = 0,

        pub fn dag(comptime self: @This()) [self.len]DAGNode(T) {
            return self.nodes[0..self.len].*;
        }

        pub fn append(self: *@This(), node: DAGNode(T)) usize {
            if (self.len >= capacity) @compileError("DAG builder capacity exceeded");
            const index = self.len;
            self.nodes[index] = node;
            self.len += 1;
            return index;
        }

        pub fn x(self: *@This()) usize {
            const index = self.append(.{ .scalar_parameter = self.param_count });
            self.param_count += 1;
            return index;
        }

        pub fn vec_x(self: *@This(), count: usize) [count]usize {
            @setEvalBranchQuota(count);
            var result: [count]usize = undefined;
            for (0..count) |i| {
                result[i] = self.x();
            }
            return result;
        }

        pub fn c(self: *@This(), value: T) usize {
            return self.append(.{ .scalar_constant = value });
        }

        pub fn vec_c(self: *@This(), values: []const T) [values.len]usize {
            @setEvalBranchQuota(values.len);
            var result: [values.len]usize = undefined;
            for (0..values.len) |i| {
                result[i] = self.c(values[i]);
            }
            return result;
        }

        pub fn mat_c(self: *@This(), rows: usize, cols: usize, values: []const T) [rows * cols]usize {
            @setEvalBranchQuota(rows * cols);
            var result: [rows * cols]usize = undefined;
            for (0..rows) |i| {
                for (0..cols) |j| {
                    result[i * cols + j] = self.c(values[i * cols + j]);
                }
            }
            return result;
        }

        pub fn mat_mul(
            self: *@This(),
            rows: usize,
            inner: usize,
            cols: usize,
            lhs: []const usize,
            rhs: []const usize,
        ) [rows * cols]usize {
            @setEvalBranchQuota(rows * cols * inner);
            var result: [rows * cols]usize = undefined;
            for (0..rows) |i| {
                for (0..cols) |j| {
                    for (0..inner) |k| {
                        if (k == 0) {
                            result[i * cols + j] = self.mul(lhs[i * inner + k], rhs[k * cols + j]);
                        } else {
                            result[i * cols + j] = self.add(result[i * cols + j], self.mul(lhs[i * inner + k], rhs[k * cols + j]));
                        }
                    }
                }
            }
            return result;
        }

        pub fn op1(self: *@This(), node: usize, op: Op1) usize {
            return self.append(.{ .op1 = .{ .node = node, .op = op } });
        }

        pub fn op2(self: *@This(), lhs: usize, rhs: usize, op: Op2) usize {
            return self.append(dag_mod.op2_node(T, lhs, rhs, op));
        }

        pub fn output(self: *@This(), node: usize) void {
            _ = self.append(.{ .output = .{ .index = self.output_count, .node = node } });
            self.output_count += 1;
        }

        pub fn neg(self: *@This(), node: usize) usize {
            return self.op1(node, .neg);
        }

        pub fn abs(self: *@This(), node: usize) usize {
            return self.op1(node, .abs);
        }

        pub fn exp(self: *@This(), node: usize) usize {
            return self.op1(node, .exp);
        }

        pub fn log(self: *@This(), node: usize) usize {
            return self.op1(node, .log);
        }

        pub fn sqrt(self: *@This(), node: usize) usize {
            return self.op1(node, .sqrt);
        }

        pub fn sin(self: *@This(), node: usize) usize {
            return self.op1(node, .sin);
        }

        pub fn cos(self: *@This(), node: usize) usize {
            return self.op1(node, .cos);
        }

        pub fn tan(self: *@This(), node: usize) usize {
            return self.op1(node, .tan);
        }

        pub fn add(self: *@This(), lhs: usize, rhs: usize) usize {
            if (lhs > rhs) {
                return self.op2(rhs, lhs, .add);
            } else {
                return self.op2(lhs, rhs, .add);
            }
        }

        pub fn sub(self: *@This(), lhs: usize, rhs: usize) usize {
            return self.op2(lhs, rhs, .sub);
        }

        pub fn mul(self: *@This(), lhs: usize, rhs: usize) usize {
            if (lhs > rhs) {
                return self.op2(rhs, lhs, .mul);
            } else {
                return self.op2(lhs, rhs, .mul);
            }
        }

        pub fn div(self: *@This(), lhs: usize, rhs: usize) usize {
            return self.op2(lhs, rhs, .div);
        }
    };
}

fn simple_func_dag(comptime T: type) [8]DAGNode(T) {
    var b = Builder(T, 8){};
    const x1 = b.x();
    const x2 = b.x();
    const v1 = b.log(x1);
    const v2 = b.mul(x1, x2);
    const v3 = b.sin(x2);
    const v4 = b.add(v1, v2);
    const v5 = b.sub(v4, v3);
    b.output(v5);
    return b.dag();
}

test "builder" {
    const std = @import("std");
    const test_dag = [_]DAGNode(f64){
        DAGNode(f64){ .scalar_parameter = 0 },
        DAGNode(f64){ .scalar_parameter = 1 },
        DAGNode(f64){ .op1 = .{ .node = 0, .op = .log } },
        DAGNode(f64){ .op2 = .{ .lhs = 0, .rhs = 1, .op = .mul } },
        DAGNode(f64){ .op1 = .{ .node = 1, .op = .sin } },
        DAGNode(f64){ .op2 = .{ .lhs = 2, .rhs = 3, .op = .add } },
        DAGNode(f64){ .op2 = .{ .lhs = 5, .rhs = 4, .op = .sub } },
        DAGNode(f64){ .output = .{ .index = 0, .node = 6 } },
    };
    const build_dag = comptime simple_func_dag(f64);
    for (build_dag, test_dag) |actual, expected| {
        try std.testing.expectEqual(expected, actual);
    }
    validate_dag(f64, &build_dag);
}

pub fn append(comptime T: type, comptime dag: []const DAGNode(T), node: DAGNode(T)) [dag.len + 1]DAGNode(T) {
    const old_dag: [dag.len]DAGNode(T) = dag[0..dag.len].*;
    return old_dag ++ [1]DAGNode(T){node};
}

test "append" {
    const std = @import("std");
    const dag = [_]DAGNode(f64){
        DAGNode(f64){ .scalar_parameter = 0 },
        DAGNode(f64){ .scalar_parameter = 1 },
    };
    const new_dag = append(f64, &dag, DAGNode(f64){ .op1 = .{ .node = 0, .op = .log } });
    try std.testing.expectEqual(new_dag, [_]DAGNode(f64){
        DAGNode(f64){ .scalar_parameter = 0 },
        DAGNode(f64){ .scalar_parameter = 1 },
        DAGNode(f64){ .op1 = .{ .node = 0, .op = .log } },
    });
}
