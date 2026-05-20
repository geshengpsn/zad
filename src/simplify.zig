const std = @import("std");
const DAGNode = @import("dag.zig").DAGNode;
const Builder = @import("dag_builder.zig").Builder;

const deadcode_mod = @import("simplify/deadcode.zig");
const constant_fold_mod = @import("simplify/constant_fold.zig");
const op1_mod = @import("simplify/op1.zig");
const op2_mod = @import("simplify/op2.zig");
const cse_mod = @import("simplify/cse.zig");
const rewire_mod = @import("simplify/rewire.zig");

test {
    _ = deadcode_mod;
    _ = constant_fold_mod;
    _ = op1_mod;
    _ = op2_mod;
    _ = cse_mod;
    _ = rewire_mod;
}

fn eval_branch_quota(comptime dag_len: usize) comptime_int {
    comptime var scale: usize = dag_len;
    if (scale == 0) scale = 1;

    comptime var quota: usize = scale;
    quota *= scale;
    quota *= scale;
    quota *= 100;
    return @max(quota, 100_000);
}

fn simplify_result(comptime T: type, comptime dag: []const DAGNode(T)) type {
    @setEvalBranchQuota(eval_branch_quota(dag.len));
    var result = dag;
    for (0..dag.len) |_| {
        if (deadcode_mod.has_deadcode(T, result)) {
            result = &deadcode_mod.deadcode_elimination(T, result);
            continue;
        }
        if (constant_fold_mod.has_unfold_constant(T, result)) {
            result = &constant_fold_mod.constant_fold(T, result);
            continue;
        }
        if (op1_mod.has_neg_neg(T, result)) {
            result = &op1_mod.neg_neg(T, result);
            continue;
        }
        if (op1_mod.has_abs_abs(T, result)) {
            result = &op1_mod.abs_abs(T, result);
            continue;
        }
        if (op1_mod.has_cos_neg(T, result)) {
            result = &op1_mod.cos_neg(T, result);
            continue;
        }
        if (op1_mod.has_log_exp(T, result)) {
            result = &op1_mod.log_exp(T, result);
            continue;
        }
        if (op1_mod.has_exp_log(T, result)) {
            result = &op1_mod.exp_log(T, result);
            continue;
        }
        if (op2_mod.has_add_zero(T, result)) {
            result = &op2_mod.add_zero(T, result);
            continue;
        }
        if (op2_mod.has_sub_zero(T, result)) {
            result = &op2_mod.sub_zero(T, result);
            continue;
        }
        if (op2_mod.has_self_sub(T, result)) {
            result = &op2_mod.self_sub(T, result);
            continue;
        }
        if (op2_mod.has_mul_zero(T, result)) {
            result = &op2_mod.mul_zero(T, result);
            continue;
        }
        if (op2_mod.has_mul_one(T, result)) {
            result = &op2_mod.mul_one(T, result);
            continue;
        }
        if (op2_mod.has_mul_neg(T, result)) {
            result = &op2_mod.mul_neg(T, result);
            continue;
        }
        if (op2_mod.has_div_one(T, result)) {
            result = &op2_mod.div_one(T, result);
            continue;
        }
        if (op2_mod.has_zero_div(T, result)) {
            result = &op2_mod.zero_div(T, result);
            continue;
        }
        if (op2_mod.has_self_div(T, result)) {
            result = &op2_mod.self_div(T, result);
            continue;
        }
        if (cse_mod.has_common_subexpression(T, result)) {
            result = &cse_mod.cse(T, result);
            continue;
        }
        break;
    }
    const reduced = deadcode_mod.deadcode_elimination(T, result);
    return @TypeOf(reduced);
}

fn simplify_impl(comptime T: type, comptime dag: []const DAGNode(T)) simplify_result(T, dag) {
    @setEvalBranchQuota(eval_branch_quota(dag.len));
    var result = dag;
    for (0..dag.len) |_| {
        if (deadcode_mod.has_deadcode(T, result)) {
            result = &deadcode_mod.deadcode_elimination(T, result);
            continue;
        }
        if (constant_fold_mod.has_unfold_constant(T, result)) {
            result = &constant_fold_mod.constant_fold(T, result);
            continue;
        }
        if (op1_mod.has_neg_neg(T, result)) {
            result = &op1_mod.neg_neg(T, result);
            continue;
        }
        if (op1_mod.has_abs_abs(T, result)) {
            result = &op1_mod.abs_abs(T, result);
            continue;
        }
        if (op1_mod.has_cos_neg(T, result)) {
            result = &op1_mod.cos_neg(T, result);
            continue;
        }
        if (op1_mod.has_log_exp(T, result)) {
            result = &op1_mod.log_exp(T, result);
            continue;
        }
        if (op1_mod.has_exp_log(T, result)) {
            result = &op1_mod.exp_log(T, result);
            continue;
        }
        if (op2_mod.has_add_zero(T, result)) {
            result = &op2_mod.add_zero(T, result);
            continue;
        }
        if (op2_mod.has_sub_zero(T, result)) {
            result = &op2_mod.sub_zero(T, result);
            continue;
        }
        if (op2_mod.has_self_sub(T, result)) {
            result = &op2_mod.self_sub(T, result);
            continue;
        }
        if (op2_mod.has_mul_zero(T, result)) {
            result = &op2_mod.mul_zero(T, result);
            continue;
        }
        if (op2_mod.has_mul_one(T, result)) {
            result = &op2_mod.mul_one(T, result);
            continue;
        }
        if (op2_mod.has_mul_neg(T, result)) {
            result = &op2_mod.mul_neg(T, result);
            continue;
        }
        if (op2_mod.has_div_one(T, result)) {
            result = &op2_mod.div_one(T, result);
            continue;
        }
        if (op2_mod.has_zero_div(T, result)) {
            result = &op2_mod.zero_div(T, result);
            continue;
        }
        if (op2_mod.has_self_div(T, result)) {
            result = &op2_mod.self_div(T, result);
            continue;
        }
        if (cse_mod.has_common_subexpression(T, result)) {
            result = &cse_mod.cse(T, result);
            continue;
        }
        break;
    }
    const reduced = deadcode_mod.deadcode_elimination(T, result);
    return reduced;
}

pub fn simplify(comptime T: type, comptime dag: []const DAGNode(T)) simplify_result(T, dag) {
    @setEvalBranchQuota(eval_branch_quota(dag.len));
    return simplify_impl(T, dag);
}

test "constant fold" {
    const test_dag = comptime blk: {
        var b = Builder(f32, 100){};
        const v1 = b.c(std.math.pi / 2.0);
        const v2 = b.sin(v1);
        const v3 = b.c(1.0);
        const v4 = b.add(v2, v3);
        const v5 = b.x();
        const v6 = b.mul(v5, v4);
        b.output(v6);
        break :blk b.dag();
    };

    const expected_dag = comptime blk: {
        var b = Builder(f32, 100){};
        const v4 = b.c(@sin(std.math.pi / 2.0) + 1.0);
        const v5 = b.x();
        const v6 = b.mul(v5, v4);
        b.output(v6);
        break :blk b.dag();
    };
    const folded_dag = comptime simplify(f32, &test_dag);
    try std.testing.expectEqualSlices(DAGNode(f32), &folded_dag, &expected_dag);
}
