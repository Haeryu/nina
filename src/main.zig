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
    const max_epochs = 10;

    const block_size = 128;
    const batch_size = 4;
    const n_layer = 4;
    const n_embd = 128;
    const n_head = 4;

    var gpt: nina.model.GPT(T, n_layer) = try .init(
        .{
            .block_size = block_size,
            .n_head = n_head,
            .n_embd = n_embd,
        },
        &context,
        base_chain,
    );
    defer gpt.destroy();

    const tokenizer: nina.tokenizer.BpeTokenizer = try .init();
    var dataset: nina.dataset.GPT2Dataset = try .init(allocator, &tokenizer, block_size, &.{
        "./datas/tiny.txt",
    });
    defer dataset.deinit();

    var dataloader: tomorin.dataloader.DataLoader(nina.dataset.GPT2Dataset) = try .init(
        allocator,
        &dataset,
        batch_size,
        true,
        &context,
    );
    defer dataloader.deinit();

    var optimizer: tomorin.optimizer.AdamW(T) = try .init(.default, &context);
    defer optimizer.deinit();

    var indices_t: tomo.tensor.GPUTensor(usize) = try .initAsync(&.{ batch_size, block_size }, &stream);
    defer indices_t.deinitAsync(&stream);

    var indices = try base_chain.createVariable(usize, indices_t.move(), "indices");
    defer indices.destroy();

    var targets_t: tomo.tensor.GPUTensor(usize) = try .initAsync(&.{ batch_size, block_size }, &stream);
    defer targets_t.deinitAsync(&stream);

    var targets = try base_chain.createVariable(usize, targets_t.move(), "targets");
    defer targets.destroy();

    var iter_chain = try context.createChain();
    defer iter_chain.destroy();

    var timer = try std.time.Timer.start();

    for (0..max_epochs) |epoch| {
        dataloader.reset();
        var sum_loss: T = 0.0;
        var sum_acc: T = 0.0;
        timer.reset();

        while (try dataloader.writeNextBatch(.{ &indices.asUntagged(usize).data, &targets.asUntagged(usize).data })) |_| {

            // Forward pass
            const logits, const loss = try gpt.forward(indices, targets, iter_chain);

            indices.clearGrad();
            targets.clearGrad();
            gpt.clearGrads();

            try loss.?.backwardEx(iter_chain);
            try optimizer.update(&gpt.getParams());

            var host_loss = try loss.?.asUntagged(T).data.toHost(allocator, &stream);
            defer host_loss.deinit(allocator);

            const acc = try tomorin.util.accuracy(T, logits, targets, 2);

            try stream.sync();
            sum_loss += host_loss.at(&.{ 0, 0 }).*;
            sum_acc += acc;

            try stream.sync();

            std.debug.print("epoch {} loss {d} acc {d}\n", .{
                epoch + 1,
                host_loss.at(&.{ 0, 0 }).*,
                acc,
            });

            iter_chain.clear();
        }

        const len: T = @floatFromInt(dataloader.max_iter);
        const elapsed = timer.lap();

        std.debug.print("epoch {} avg loss {d} acc {d} elapsed {d}\n", .{
            epoch + 1,
            sum_loss / len,
            sum_acc / len,
            @as(f32, @floatFromInt(elapsed)) / @as(f32, @floatFromInt(std.time.ns_per_s)),
        });
        try gpt.saveBinary(allocator, "gpt_train.bin");
    }
}
