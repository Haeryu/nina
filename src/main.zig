const nina = @import("nina");
const std = @import("std");
const tomo = @import("tomo");
const tomorin = @import("tomorin");

pub fn main() !void {
    var gpa: std.heap.DebugAllocator(.{}) = .init;
    defer _ = gpa.deinit();

    const allocator = gpa.allocator();

    var stream: tomo.stream.Stream = try .create();
    defer stream.destroy();

    var cuda_context: tomo.cuda_context.CudaContext = try .init();
    defer cuda_context.deinit();

    var context: tomorin.context.Context = try .init(allocator, &cuda_context, &stream, .{
        .init_func_capacity = 100,
        .init_var_capacity = 100,
    });
    defer context.deinit();

    const base_chain = try context.createChain();
    defer base_chain.destroy();
    context.current_chain = base_chain;

    const T = f32;

    var gpt: nina.model.GPT(T, 12) = try .init(
        .{},
        &context,
        base_chain,
    );
    defer gpt.destroy();
}
