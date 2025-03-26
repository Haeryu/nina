const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});

    const optimize = b.standardOptimizeOption(.{});

    const tomo = b.dependency("tomo", .{ .target = target, .optimize = optimize });
    const tomorin = b.dependency("tomorin", .{ .target = target, .optimize = optimize });

    const lib_mod = b.createModule(.{
        .root_source_file = b.path("src/root.zig"),
        .target = target,
        .optimize = optimize,
    });

    lib_mod.addImport("tomo", tomo.module("tomo"));
    lib_mod.addImport("tomorin", tomorin.module("tomorin"));

    const exe_mod = b.createModule(.{
        .root_source_file = b.path("src/main.zig"),
        .target = target,
        .optimize = optimize,
    });

    exe_mod.addImport("nina", lib_mod);
    exe_mod.addImport("tomo", tomo.module("tomo"));
    exe_mod.addImport("tomorin", tomorin.module("tomorin"));

    // const lib = b.addLibrary(.{
    //     .linkage = .static,
    //     .name = "nina",
    //     .root_module = lib_mod,
    // });

    // b.installArtifact(lib);

    const exe = b.addExecutable(.{
        .name = "nina",
        .root_module = exe_mod,
    });

    b.installArtifact(exe);

    const run_cmd = b.addRunArtifact(exe);

    run_cmd.step.dependOn(b.getInstallStep());

    if (b.args) |args| {
        run_cmd.addArgs(args);
    }

    const run_step = b.step("run", "Run the app");
    run_step.dependOn(&run_cmd.step);

    const lib_unit_tests = b.addTest(.{
        .root_module = lib_mod,
    });

    const run_lib_unit_tests = b.addRunArtifact(lib_unit_tests);

    const exe_unit_tests = b.addTest(.{
        .root_module = exe_mod,
    });

    const run_exe_unit_tests = b.addRunArtifact(exe_unit_tests);

    const test_step = b.step("test", "Run unit tests");
    test_step.dependOn(&run_lib_unit_tests.step);
    test_step.dependOn(&run_exe_unit_tests.step);

    const export_tokenizer_step = b.step("tokenizer", "Run Python tokenizer export script");

    const run_script = b.addSystemCommand(&[_][]const u8{ "python", "./src/python/export_tokenizer_zig.py" });
    export_tokenizer_step.dependOn(&run_script.step);
}
