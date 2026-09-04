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

    const examples_step = b.step("examples", "Compile all examples");
    const io = b.graph.io;
    var examples_dir = b.build_root.handle.openDir(io, "examples", .{ .iterate = true }) catch @panic("unable to open examples directory");
    defer examples_dir.close(io);
    var examples = examples_dir.iterate();
    while (examples.next(io) catch @panic("unable to iterate examples directory")) |entry| {
        if (entry.kind != .file or !std.mem.endsWith(u8, entry.name, ".zig")) continue;

        const name = b.dupe(entry.name[0 .. entry.name.len - ".zig".len]);
        const executable = b.addExecutable(.{
            .name = name,
            .root_module = b.createModule(.{
                .root_source_file = b.path(b.fmt("examples/{s}", .{entry.name})),
                .target = target,
                .optimize = optimize,
                .imports = &.{.{ .name = "zad", .module = mod }},
            }),
        });
        const install_executable = b.addInstallArtifact(executable, .{});
        examples_step.dependOn(&install_executable.step);
    }

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
