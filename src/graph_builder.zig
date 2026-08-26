const std = @import("std");
const dag_mod = @import("dag.zig");
const DAGNode = dag_mod.DAGNode;
const Op1 = dag_mod.Op1;
const Op2 = dag_mod.Op2;
const simplify = @import("simplify.zig").simplify;

pub fn Node(comptime T: type) type {
    return union(enum) {
        const Self = @This();

        parameter: usize,
        constant: T,
        op1: struct {
            node: *const Self,
            op: Op1,
        },
        op2: struct {
            lhs: *const Self,
            rhs: *const Self,
            op: Op2,
        },

        pub fn c(value: T) Self {
            return .{ .constant = value };
        }

        pub fn vec_c(comptime values: []const T) [values.len]Self {
            var result: [values.len]Self = undefined;
            for (values, 0..) |value, i| result[i] = Self.c(value);
            return result;
        }

        pub fn op1_node(node: *const Self, op: Op1) Self {
            return .{ .op1 = .{ .node = node, .op = op } };
        }

        pub fn op2_node(lhs: *const Self, rhs: *const Self, op: Op2) Self {
            return .{ .op2 = .{ .lhs = lhs, .rhs = rhs, .op = op } };
        }

        pub fn neg(node: *const Self) Self {
            return Self.op1_node(node, .neg);
        }

        pub fn abs(node: *const Self) Self {
            return Self.op1_node(node, .abs);
        }

        pub fn exp(node: *const Self) Self {
            return Self.op1_node(node, .exp);
        }

        pub fn log(node: *const Self) Self {
            return Self.op1_node(node, .log);
        }

        pub fn sqrt(node: *const Self) Self {
            return Self.op1_node(node, .sqrt);
        }

        pub fn sin(node: *const Self) Self {
            return Self.op1_node(node, .sin);
        }

        pub fn cos(node: *const Self) Self {
            return Self.op1_node(node, .cos);
        }

        pub fn tan(node: *const Self) Self {
            return Self.op1_node(node, .tan);
        }

        pub fn add(lhs: *const Self, rhs: *const Self) Self {
            return Self.op2_node(lhs, rhs, .add);
        }

        pub fn sub(lhs: *const Self, rhs: *const Self) Self {
            return Self.op2_node(lhs, rhs, .sub);
        }

        pub fn mul(lhs: *const Self, rhs: *const Self) Self {
            return Self.op2_node(lhs, rhs, .mul);
        }

        pub fn div(lhs: *const Self, rhs: *const Self) Self {
            return Self.op2_node(lhs, rhs, .div);
        }
    };
}

const GraphValueKind = enum {
    scalar,
    vector,
    matrix,
};

pub fn Scalar(comptime T: type) type {
    return struct {
        const Self = @This();

        pub const graph_value_kind = GraphValueKind.scalar;
        pub const graph_value_type = T;
        pub const graph_node_count = 1;

        node: Node(T),

        pub fn c(value: T) Self {
            return .{ .node = Node(T).c(value) };
        }

        pub fn neg(self: *const Self) Self {
            return .{ .node = Node(T).neg(&self.node) };
        }

        pub fn abs(self: *const Self) Self {
            return .{ .node = Node(T).abs(&self.node) };
        }

        pub fn exp(self: *const Self) Self {
            return .{ .node = Node(T).exp(&self.node) };
        }

        pub fn log(self: *const Self) Self {
            return .{ .node = Node(T).log(&self.node) };
        }

        pub fn sqrt(self: *const Self) Self {
            return .{ .node = Node(T).sqrt(&self.node) };
        }

        pub fn sin(self: *const Self) Self {
            return .{ .node = Node(T).sin(&self.node) };
        }

        pub fn cos(self: *const Self) Self {
            return .{ .node = Node(T).cos(&self.node) };
        }

        pub fn tan(self: *const Self) Self {
            return .{ .node = Node(T).tan(&self.node) };
        }

        pub fn add(lhs: *const Self, rhs: *const Self) Self {
            return .{ .node = Node(T).add(&lhs.node, &rhs.node) };
        }

        pub fn sub(lhs: *const Self, rhs: *const Self) Self {
            return .{ .node = Node(T).sub(&lhs.node, &rhs.node) };
        }

        pub fn mul(lhs: *const Self, rhs: *const Self) Self {
            return .{ .node = Node(T).mul(&lhs.node, &rhs.node) };
        }

        pub fn div(lhs: *const Self, rhs: *const Self) Self {
            return .{ .node = Node(T).div(&lhs.node, &rhs.node) };
        }
    };
}

pub fn Vec(comptime T: type, comptime len: usize) type {
    return struct {
        const Self = @This();

        pub const graph_value_kind = GraphValueKind.vector;
        pub const graph_value_type = T;
        pub const graph_node_count = len;
        pub const length = len;

        nodes: [len]Node(T),

        pub fn c(values: [len]T) Self {
            var result: Self = undefined;
            for (values, 0..) |value, i| result.nodes[i] = Node(T).c(value);
            return result;
        }

        pub fn at(self: *const Self, comptime index: usize) Scalar(T) {
            if (index >= len) @compileError("vector index out of bounds");
            return .{ .node = self.nodes[index] };
        }

        pub fn add(lhs: *const Self, rhs: *const Self) Self {
            var result: Self = undefined;
            for (0..len) |i| result.nodes[i] = Node(T).add(&lhs.nodes[i], &rhs.nodes[i]);
            return result;
        }

        pub fn sub(lhs: *const Self, rhs: *const Self) Self {
            var result: Self = undefined;
            for (0..len) |i| result.nodes[i] = Node(T).sub(&lhs.nodes[i], &rhs.nodes[i]);
            return result;
        }

        pub fn mul(vector: *const Self, scalar: *const Scalar(T)) Self {
            var result: Self = undefined;
            for (0..len) |i| result.nodes[i] = Node(T).mul(&vector.nodes[i], &scalar.node);
            return result;
        }

        pub fn dot(lhs: *const Self, rhs: *const Self) Scalar(T) {
            if (len == 0) @compileError("dot product requires a non-empty vector");

            var products: [len]Node(T) = undefined;
            for (0..len) |i| products[i] = Node(T).mul(&lhs.nodes[i], &rhs.nodes[i]);
            if (len == 1) return .{ .node = products[0] };

            var sums: [len - 1]Node(T) = undefined;
            sums[0] = Node(T).add(&products[0], &products[1]);
            for (2..len) |i| sums[i - 1] = Node(T).add(&sums[i - 2], &products[i]);
            return .{ .node = sums[len - 2] };
        }
    };
}

pub fn Mat(comptime T: type, comptime rows: usize, comptime cols: usize) type {
    return struct {
        const Self = @This();

        pub const graph_value_kind = GraphValueKind.matrix;
        pub const graph_value_type = T;
        pub const graph_node_count = rows * cols;
        pub const row_count = rows;
        pub const col_count = cols;

        nodes: [rows * cols]Node(T),

        pub fn c(values: [rows * cols]T) Self {
            var result: Self = undefined;
            for (values, 0..) |value, i| result.nodes[i] = Node(T).c(value);
            return result;
        }

        pub fn at(self: *const Self, comptime row: usize, comptime col: usize) Scalar(T) {
            if (row >= rows or col >= cols) @compileError("matrix index out of bounds");
            return .{ .node = self.nodes[row * cols + col] };
        }

        pub fn matMul(matrix: *const Self, vector: *const Vec(T, cols)) Vec(T, rows) {
            if (cols == 0) @compileError("matrix-vector multiplication requires a non-empty inner dimension");

            var products: [rows * cols]Node(T) = undefined;
            for (0..rows) |row| {
                for (0..cols) |col| {
                    const index = row * cols + col;
                    products[index] = Node(T).mul(&matrix.nodes[index], &vector.nodes[col]);
                }
            }

            var result: Vec(T, rows) = undefined;
            if (cols == 1) {
                for (0..rows) |row| result.nodes[row] = products[row];
                return result;
            }

            var sums: [rows * (cols - 1)]Node(T) = undefined;
            for (0..rows) |row| {
                const product_start = row * cols;
                const sum_start = row * (cols - 1);
                sums[sum_start] = Node(T).add(&products[product_start], &products[product_start + 1]);
                for (2..cols) |col| {
                    sums[sum_start + col - 1] = Node(T).add(&sums[sum_start + col - 2], &products[product_start + col]);
                }
                result.nodes[row] = sums[sum_start + cols - 2];
            }
            return result;
        }
    };
}

fn index_of(comptime T: type, nodes: []const *const Node(T), needle: *const Node(T)) ?usize {
    for (nodes, 0..) |node, i| {
        if (node == needle) return i;
    }
    return null;
}

fn same_graph_node(comptime T: type, lhs: *const Node(T), rhs: *const Node(T)) bool {
    if (lhs == rhs) return true;
    return switch (lhs.*) {
        .parameter => |lhs_index| switch (rhs.*) {
            .parameter => |rhs_index| lhs_index == rhs_index,
            else => false,
        },
        else => false,
    };
}

fn Path(comptime T: type) type {
    return struct {
        node: *const Node(T),
        parent: ?*const @This(),
    };
}

fn Exclusion(comptime T: type) type {
    return struct {
        root: *const Node(T),
        parent: ?*const @This(),
    };
}

fn path_contains(comptime T: type, path: ?*const Path(T), node: *const Node(T)) bool {
    var current = path;
    while (current) |entry| : (current = entry.parent) {
        if (entry.node == node) return true;
    }
    return false;
}

fn is_reachable(
    comptime T: type,
    root: *const Node(T),
    needle: *const Node(T),
    path: ?*const Path(T),
) bool {
    if (same_graph_node(T, root, needle)) return true;
    if (path_contains(T, path, root)) @compileError("graph contains a cycle");

    const current = Path(T){ .node = root, .parent = path };
    return switch (root.*) {
        .parameter, .constant => false,
        .op1 => |op| is_reachable(T, op.node, needle, &current),
        .op2 => |op| is_reachable(T, op.lhs, needle, &current) or
            is_reachable(T, op.rhs, needle, &current),
    };
}

fn graph_root(
    comptime T: type,
    inputs: []const *const Node(T),
    outputs: []const *const Node(T),
    index: usize,
) *const Node(T) {
    return if (index < inputs.len) inputs[index] else outputs[index - inputs.len];
}

fn is_excluded(
    comptime T: type,
    node: *const Node(T),
    inputs: []const *const Node(T),
    outputs: []const *const Node(T),
    prior_root_count: usize,
    exclusions: ?*const Exclusion(T),
) bool {
    for (0..prior_root_count) |i| {
        if (is_reachable(T, graph_root(T, inputs, outputs, i), node, null)) return true;
    }

    var current = exclusions;
    while (current) |entry| : (current = entry.parent) {
        if (is_reachable(T, entry.root, node, null)) return true;
    }
    return false;
}

fn subtree_node_count(
    comptime T: type,
    node: *const Node(T),
    inputs: []const *const Node(T),
    outputs: []const *const Node(T),
    prior_root_count: usize,
    exclusions: ?*const Exclusion(T),
    path: ?*const Path(T),
) usize {
    if (path_contains(T, path, node)) @compileError("graph contains a cycle");
    if (is_excluded(T, node, inputs, outputs, prior_root_count, exclusions)) return 0;

    const current_path = Path(T){ .node = node, .parent = path };
    return switch (node.*) {
        .parameter, .constant => 1,
        .op1 => |op| 1 + subtree_node_count(
            T,
            op.node,
            inputs,
            outputs,
            prior_root_count,
            exclusions,
            &current_path,
        ),
        .op2 => |op| {
            const rhs_exclusion = Exclusion(T){ .root = op.lhs, .parent = exclusions };
            return 1 +
                subtree_node_count(T, op.lhs, inputs, outputs, prior_root_count, exclusions, &current_path) +
                subtree_node_count(T, op.rhs, inputs, outputs, prior_root_count, &rhs_exclusion, &current_path);
        },
    };
}

fn graph_node_count(
    comptime T: type,
    comptime inputs: []const *const Node(T),
    comptime outputs: []const *const Node(T),
) usize {
    @setEvalBranchQuota(1_000_000);

    var count: usize = 0;
    for (0..inputs.len + outputs.len) |i| {
        count += subtree_node_count(
            T,
            graph_root(T, inputs, outputs, i),
            inputs,
            outputs,
            i,
            null,
            null,
        );
    }
    return count;
}

fn collect_subtree(
    comptime T: type,
    node: *const Node(T),
    inputs: []const *const Node(T),
    outputs: []const *const Node(T),
    prior_root_count: usize,
    exclusions: ?*const Exclusion(T),
    path: ?*const Path(T),
    ordered: anytype,
    ordered_len: *usize,
) void {
    if (path_contains(T, path, node)) @compileError("graph contains a cycle");
    if (is_excluded(T, node, inputs, outputs, prior_root_count, exclusions)) return;

    const current_path = Path(T){ .node = node, .parent = path };
    switch (node.*) {
        .parameter, .constant => {},
        .op1 => |op| collect_subtree(
            T,
            op.node,
            inputs,
            outputs,
            prior_root_count,
            exclusions,
            &current_path,
            ordered,
            ordered_len,
        ),
        .op2 => |op| {
            collect_subtree(T, op.lhs, inputs, outputs, prior_root_count, exclusions, &current_path, ordered, ordered_len);
            const rhs_exclusion = Exclusion(T){ .root = op.lhs, .parent = exclusions };
            collect_subtree(T, op.rhs, inputs, outputs, prior_root_count, &rhs_exclusion, &current_path, ordered, ordered_len);
        },
    }

    ordered[ordered_len.*] = node;
    ordered_len.* += 1;
}

fn collect_graph_nodes(
    comptime T: type,
    comptime inputs: []const *const Node(T),
    comptime outputs: []const *const Node(T),
) [graph_node_count(T, inputs, outputs)]*const Node(T) {
    @setEvalBranchQuota(1_000_000);

    var ordered: [graph_node_count(T, inputs, outputs)]*const Node(T) = undefined;
    var ordered_len: usize = 0;
    for (0..inputs.len + outputs.len) |i| {
        collect_subtree(
            T,
            graph_root(T, inputs, outputs, i),
            inputs,
            outputs,
            i,
            null,
            null,
            &ordered,
            &ordered_len,
        );
    }
    return ordered;
}

fn nodes_to_dag(
    comptime T: type,
    comptime inputs: []const *const Node(T),
    comptime outputs: []const *const Node(T),
) [graph_node_count(T, inputs, outputs) + outputs.len]DAGNode(T) {
    @setEvalBranchQuota(1_000_000);

    const value_count = graph_node_count(T, inputs, outputs);
    const ordered = collect_graph_nodes(T, inputs, outputs);
    var dag: [value_count + outputs.len]DAGNode(T) = undefined;

    const dag_index = struct {
        fn get(
            nodes: []const *const Node(T),
            node: *const Node(T),
            input_count: usize,
        ) ?usize {
            return switch (node.*) {
                .parameter => |index| if (index < input_count) index else null,
                else => index_of(T, nodes, node),
            };
        }
    }.get;

    for (ordered, 0..) |node, i| {
        dag[i] = switch (node.*) {
            .parameter => |index| .{ .scalar_parameter = index },
            .constant => |value| .{ .scalar_constant = value },
            .op1 => |op| .{ .op1 = .{
                .node = dag_index(ordered[0..i], op.node, inputs.len) orelse unreachable,
                .op = op.op,
            } },
            .op2 => |op| dag_mod.op2_node(
                T,
                dag_index(ordered[0..i], op.lhs, inputs.len) orelse unreachable,
                dag_index(ordered[0..i], op.rhs, inputs.len) orelse unreachable,
                op.op,
            ),
        };
    }

    for (outputs, 0..) |output, i| {
        dag[value_count + i] = .{ .output = .{
            .index = i,
            .node = dag_index(&ordered, output, inputs.len) orelse unreachable,
        } };
    }

    dag_mod.validate_dag(T, &dag);
    return dag;
}

fn value_kind(comptime T: type, comptime Value: type) GraphValueKind {
    if (@typeInfo(Value) != .@"struct" or
        !@hasDecl(Value, "graph_value_kind") or
        !@hasDecl(Value, "graph_value_type") or
        !@hasDecl(Value, "graph_node_count"))
    {
        @compileError("graph values must be Scalar(T), Vec(T, n), or Mat(T, rows, cols)");
    }
    if (Value.graph_value_type != T) @compileError("graph value scalar type does not match to_dag's T");
    return Value.graph_value_kind;
}

fn value_node_count(comptime T: type, comptime Value: type) usize {
    _ = value_kind(T, Value);
    return Value.graph_node_count;
}

fn input_value_type(comptime T: type, comptime Param: type) type {
    const pointer = switch (@typeInfo(Param)) {
        .pointer => |pointer| pointer,
        else => @compileError("graph function parameters must be pointers to Scalar, Vec, or Mat values"),
    };
    if (pointer.size != .one or !pointer.is_const) {
        @compileError("graph function parameters must be *const pointers");
    }
    _ = value_kind(T, pointer.child);
    return pointer.child;
}

fn graph_function_info(comptime T: type, comptime graph_fn: anytype) std.builtin.Type.Fn {
    const info = @typeInfo(@TypeOf(graph_fn));
    if (info != .@"fn") @compileError("to_dag expects a function");
    if (info.@"fn".is_var_args) @compileError("graph functions cannot be variadic");

    inline for (info.@"fn".params) |param| {
        const Param = param.type orelse @compileError("graph function parameters cannot be anytype");
        _ = input_value_type(T, Param);
    }
    return info.@"fn";
}

fn graph_return_type(comptime T: type, comptime graph_fn: anytype) type {
    const Return = graph_function_info(T, graph_fn).return_type orelse
        @compileError("graph function must have an explicit return type");
    switch (value_kind(T, Return)) {
        .scalar, .vector => {},
        .matrix => @compileError("graph functions must return one Scalar or one Vec"),
    }
    return Return;
}

fn input_storage_type(comptime T: type, comptime graph_fn: anytype) type {
    const fn_info = graph_function_info(T, graph_fn);
    var types: [fn_info.params.len]type = undefined;
    inline for (fn_info.params, 0..) |param, i| {
        types[i] = input_value_type(T, param.type.?);
    }
    return @Tuple(&types);
}

fn function_input_count(comptime T: type, comptime graph_fn: anytype) usize {
    const fn_info = graph_function_info(T, graph_fn);
    var count: usize = 0;
    inline for (fn_info.params) |param| {
        count += value_node_count(T, input_value_type(T, param.type.?));
    }
    return count;
}

fn parameter_value(
    comptime T: type,
    comptime Value: type,
    start_index: usize,
) Value {
    var result: Value = undefined;
    switch (value_kind(T, Value)) {
        .scalar => result.node = .{ .parameter = start_index },
        .vector, .matrix => for (&result.nodes, 0..) |*node, i| {
            node.* = .{ .parameter = start_index + i };
        },
    }
    return result;
}

fn value_node_at(
    comptime T: type,
    comptime Value: type,
    value: *const Value,
    comptime index: usize,
) *const Node(T) {
    return switch (value_kind(T, Value)) {
        .scalar => if (index == 0) &value.node else unreachable,
        .vector, .matrix => &value.nodes[index],
    };
}

fn function_dag_len(comptime T: type, comptime graph_fn: anytype) usize {
    @setEvalBranchQuota(1_000_000);

    const fn_info = graph_function_info(T, graph_fn);
    const Return = graph_return_type(T, graph_fn);
    const input_count = function_input_count(T, graph_fn);
    const output_count = value_node_count(T, Return);
    var storage: input_storage_type(T, graph_fn) = undefined;
    var args: std.meta.ArgsTuple(@TypeOf(graph_fn)) = undefined;
    var input_offset: usize = 0;
    inline for (fn_info.params, 0..) |param, i| {
        const Input = input_value_type(T, param.type.?);
        storage[i] = parameter_value(T, Input, input_offset);
        args[i] = &storage[i];
        input_offset += value_node_count(T, Input);
    }
    const result: Return = @call(.auto, graph_fn, args);

    var inputs: [input_count]*const Node(T) = undefined;
    input_offset = 0;
    inline for (fn_info.params, 0..) |param, i| {
        const Input = input_value_type(T, param.type.?);
        inline for (0..value_node_count(T, Input)) |j| {
            inputs[input_offset + j] = value_node_at(T, Input, &storage[i], j);
        }
        input_offset += value_node_count(T, Input);
    }

    var outputs: [output_count]*const Node(T) = undefined;
    inline for (0..output_count) |i| outputs[i] = value_node_at(T, Return, &result, i);
    return graph_node_count(T, &inputs, &outputs) + outputs.len;
}

fn build_function_dag(
    comptime T: type,
    comptime graph_fn: anytype,
) [function_dag_len(T, graph_fn)]DAGNode(T) {
    @setEvalBranchQuota(1_000_000);

    const fn_info = graph_function_info(T, graph_fn);
    const Return = graph_return_type(T, graph_fn);
    const input_count = function_input_count(T, graph_fn);
    const output_count = value_node_count(T, Return);
    var storage: input_storage_type(T, graph_fn) = undefined;
    var args: std.meta.ArgsTuple(@TypeOf(graph_fn)) = undefined;
    var input_offset: usize = 0;
    inline for (fn_info.params, 0..) |param, i| {
        const Input = input_value_type(T, param.type.?);
        storage[i] = parameter_value(T, Input, input_offset);
        args[i] = &storage[i];
        input_offset += value_node_count(T, Input);
    }
    const result: Return = @call(.auto, graph_fn, args);

    var inputs: [input_count]*const Node(T) = undefined;
    input_offset = 0;
    inline for (fn_info.params, 0..) |param, i| {
        const Input = input_value_type(T, param.type.?);
        inline for (0..value_node_count(T, Input)) |j| {
            inputs[input_offset + j] = value_node_at(T, Input, &storage[i], j);
        }
        input_offset += value_node_count(T, Input);
    }

    var outputs: [output_count]*const Node(T) = undefined;
    inline for (0..output_count) |i| outputs[i] = value_node_at(T, Return, &result, i);
    return nodes_to_dag(T, &inputs, &outputs);
}

pub fn to_dag_raw(
    comptime T: type,
    comptime graph_fn: anytype,
) [function_dag_len(T, graph_fn)]DAGNode(T) {
    return comptime build_function_dag(T, graph_fn);
}

fn simplified_function_dag_type(comptime T: type, comptime graph_fn: anytype) type {
    const raw = comptime to_dag_raw(T, graph_fn);
    const simplified = comptime simplify(T, &raw);
    return @TypeOf(simplified);
}

pub fn to_dag(
    comptime T: type,
    comptime graph_fn: anytype,
) simplified_function_dag_type(T, graph_fn) {
    const raw = comptime to_dag_raw(T, graph_fn);
    return comptime simplify(T, &raw);
}

const TestScalar = Scalar(f64);
const TestVec2 = Vec(f64, 2);
const TestMat2 = Mat(f64, 2, 2);

fn scalar_expression(lhs: *const TestScalar, rhs: *const TestScalar) TestScalar {
    const log_lhs = lhs.log();
    const product = lhs.mul(rhs);
    return log_lhs.add(&product);
}

test "to_dag accepts scalar inputs and output" {
    const test_dag = comptime to_dag(f64, scalar_expression);
    var inputs = [_]f64{ 2.0, 3.0 };
    const actual = @import("eval.zig").eval(f64, &test_dag, &inputs);
    try std.testing.expectEqual(@log(2.0) + 6.0, actual[0]);
}

fn add_and_scale(lhs: *const TestVec2, rhs: *const TestVec2, scale: *const TestScalar) TestVec2 {
    const sum = lhs.add(rhs);
    return sum.mul(scale);
}

test "vector addition and scalar multiplication produce one vector output" {
    const test_dag = comptime to_dag(f64, add_and_scale);
    try std.testing.expectEqual(@as(usize, 5), dag_mod.input_size(f64, &test_dag));
    try std.testing.expectEqual(@as(usize, 2), dag_mod.output_size(f64, &test_dag));

    var inputs = [_]f64{ 1.0, 2.0, 3.0, 4.0, 2.0 };
    const actual = @import("eval.zig").eval(f64, &test_dag, &inputs);
    try std.testing.expectEqual([_]f64{ 8.0, 12.0 }, actual);
}

fn matrix_vector_product(matrix: *const TestMat2, vector: *const TestVec2) TestVec2 {
    return matrix.matMul(vector);
}

test "matrix-vector multiplication uses row-major input order" {
    const test_dag = comptime to_dag(f64, matrix_vector_product);
    try std.testing.expectEqual(@as(usize, 6), dag_mod.input_size(f64, &test_dag));

    var inputs = [_]f64{ 1.0, 2.0, 3.0, 4.0, 5.0, 6.0 };
    const actual = @import("eval.zig").eval(f64, &test_dag, &inputs);
    try std.testing.expectEqual([_]f64{ 17.0, 39.0 }, actual);
}

fn rectangular_matrix_vector_product(
    matrix: *const Mat(f64, 2, 3),
    vector: *const Vec(f64, 3),
) TestVec2 {
    return matrix.matMul(vector);
}

test "rectangular matrix-vector multiplication" {
    const test_dag = comptime to_dag(f64, rectangular_matrix_vector_product);
    var inputs = [_]f64{
        1.0, 2.0, 3.0,
        4.0, 5.0, 6.0,
        7.0, 8.0, 9.0,
    };
    const actual = @import("eval.zig").eval(f64, &test_dag, &inputs);
    try std.testing.expectEqual([_]f64{ 50.0, 122.0 }, actual);
}

fn single_column_matrix_vector_product(
    matrix: *const Mat(f64, 2, 1),
    vector: *const Vec(f64, 1),
) TestVec2 {
    return matrix.matMul(vector);
}

test "single-column matrix-vector multiplication" {
    const test_dag = comptime to_dag(f64, single_column_matrix_vector_product);
    var inputs = [_]f64{ 2.0, 3.0, 4.0 };
    const actual = @import("eval.zig").eval(f64, &test_dag, &inputs);
    try std.testing.expectEqual([_]f64{ 8.0, 12.0 }, actual);
}

fn quadratic_form(vector: *const TestVec2) TestScalar {
    const matrix = TestMat2.c(.{
        1.0, 2.0,
        2.0, 1.0,
    });
    const product = matrix.matMul(vector);
    const xtqx = vector.dot(&product);
    const two = TestScalar.c(2.0);
    return xtqx.div(&two);
}

test "linear algebra modules compose into a scalar function" {
    const raw = comptime to_dag_raw(f64, quadratic_form);
    const test_dag = comptime to_dag(f64, quadratic_form);
    try std.testing.expectEqual(@as(usize, 18), raw.len);
    try std.testing.expect(test_dag.len < raw.len);

    var inputs = [_]f64{ 1.0, 2.0 };
    const actual = @import("eval.zig").eval(f64, &test_dag, &inputs);
    try std.testing.expectEqual(@as(f64, 6.5), actual[0]);
}

fn first_vector_element(vector: *const TestVec2) TestScalar {
    return vector.at(0);
}

test "automatic simplification preserves unused function inputs" {
    const test_dag = comptime to_dag(f64, first_vector_element);
    try std.testing.expectEqual(@as(usize, 2), dag_mod.input_size(f64, &test_dag));
}

fn vector_identity(vector: *const TestVec2) TestVec2 {
    return vector.*;
}

test "returning a vector input preserves parameter and output order" {
    const test_dag = comptime to_dag(f64, vector_identity);
    const expected = [_]DAGNode(f64){
        .{ .scalar_parameter = 0 },
        .{ .scalar_parameter = 1 },
        .{ .output = .{ .index = 0, .node = 0 } },
        .{ .output = .{ .index = 1, .node = 1 } },
    };
    try std.testing.expectEqualSlices(DAGNode(f64), &expected, &test_dag);
}

fn deeply_shared(input: *const TestScalar) TestScalar {
    var nodes: [64]TestScalar = undefined;
    nodes[0] = input.add(input);
    for (1..nodes.len) |i| nodes[i] = nodes[i - 1].add(&nodes[i - 1]);
    return nodes[nodes.len - 1];
}

test "deeply shared graph does not require expanded tree capacity" {
    const test_dag = comptime to_dag(f64, deeply_shared);
    try std.testing.expectEqual(@as(usize, 66), test_dag.len);
}
