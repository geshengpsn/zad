const dag_mod = @import("dag.zig");
const DAGNode = dag_mod.DAGNode;
const Op1 = dag_mod.Op1;
const Op2 = dag_mod.Op2;

pub fn DAGWriter(comptime T: type, comptime capacity: usize) type {
    return struct {
        nodes: [capacity]DAGNode(T) = undefined,
        len: usize = 0,
        param_count: usize = 0,
        output_count: usize = 0,

        pub fn dag(comptime self: @This()) [self.len]DAGNode(T) {
            return self.nodes[0..self.len].*;
        }

        pub fn append(self: *@This(), node: DAGNode(T)) usize {
            if (self.len >= capacity) @compileError("DAG writer capacity exceeded");
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

        pub fn c(self: *@This(), value: T) usize {
            return self.append(.{ .scalar_constant = value });
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
            return self.op2(lhs, rhs, .add);
        }

        pub fn sub(self: *@This(), lhs: usize, rhs: usize) usize {
            return self.op2(lhs, rhs, .sub);
        }

        pub fn mul(self: *@This(), lhs: usize, rhs: usize) usize {
            return self.op2(lhs, rhs, .mul);
        }

        pub fn div(self: *@This(), lhs: usize, rhs: usize) usize {
            return self.op2(lhs, rhs, .div);
        }
    };
}
