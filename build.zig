const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});
    const mod = b.addModule("zad", .{
        .root_source_file = b.path("src/root.zig"),
        .target = target,
        .optimize = optimize,
    });

    const tests = b.addTest(.{ .root_module = mod });
    const run_tests = b.addRunArtifact(tests);
    const test_step = b.step("test", "Run all tests");
    test_step.dependOn(&run_tests.step);

    // const qp = b.addExecutable(.{
    //     .name = "qp",
    //     .root_module = b.createModule(.{
    //         .root_source_file = b.path("examples/qp.zig"),
    //         .target = target,
    //         .optimize = optimize,
    //         .imports = &.{.{ .name = "zad", .module = mod }},
    //     }),
    // });
    // const qp_run = b.addRunArtifact(qp);
    // const qp_step = b.step("qp", "Run the quadratic program example");
    // qp_step.dependOn(&qp_run.step);

    // const partial_grad = b.addExecutable(.{
    //     .name = "partial-grad",
    //     .root_module = b.createModule(.{
    //         .root_source_file = b.path("examples/partial_grad.zig"),
    //         .target = target,
    //         .optimize = optimize,
    //         .imports = &.{.{ .name = "zad", .module = mod }},
    //     }),
    // });
    // const partial_grad_run = b.addRunArtifact(partial_grad);
    // const partial_grad_step = b.step("partial-grad", "Run the partial gradient example");
    // partial_grad_step.dependOn(&partial_grad_run.step);

    // const benchmark = b.addExecutable(.{
    //     .name = "eval-bench",
    //     .root_module = b.createModule(.{
    //         .root_source_file = b.path("benches/eval.zig"),
    //         .target = target,
    //         .optimize = .ReleaseFast,
    //         .imports = &.{.{ .name = "zad", .module = mod }},
    //     }),
    // });
    // const benchmark_run = b.addRunArtifact(benchmark);
    // const benchmark_step = b.step("bench", "Run the VM evaluation benchmark");
    // benchmark_step.dependOn(&benchmark_run.step);

    // const asm_object = b.addObject(.{
    //     .name = "eval-bench-asm",
    //     .root_module = b.createModule(.{
    //         .root_source_file = b.path("benches/asm.zig"),
    //         .target = target,
    //         .optimize = .ReleaseFast,
    //         .imports = &.{.{ .name = "zad", .module = mod }},
    //     }),
    // });
    // const install_asm = b.addInstallFile(asm_object.getEmittedAsm(), "eval-bench.s");
    // const asm_step = b.step("bench-asm", "Emit VM evaluation assembly");
    // asm_step.dependOn(&install_asm.step);
}
