const nina = @import("nina");
const std = @import("std");
const tomo = @import("tomo");
const tomorin = @import("tomorin");

pub fn main() !void {
    // var gpa: std.heap.DebugAllocator(.{}) = .init;
    // defer _ = gpa.deinit();

    // TODO: make one big struct to train, chat, save (make dataset, dataloader Save code) -> cache system

    const allocator = std.heap.smp_allocator;

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

    const T = f32; // TODO -> BF16 support start..
    const max_epochs = 10;

    // const block_size = 128;
    // const batch_size = 4;
    // const n_layer = 4;
    // const n_embd = 128;
    // const n_head = 4;

    const block_size = 128;
    const batch_size = 8;
    const n_layer = 6;
    const n_embd = 256;
    const n_head = 8;
    const savefile = "gpt_train_tiny.bin";

    const tokenizer: nina.tokenizer.BpeTokenizer = try .init();

    var timer = try std.time.Timer.start();
    var dataset: nina.dataset.GPT2Dataset = try .init(allocator, &tokenizer, block_size, &.{
        "datas/bin/token_ids.bin",
    });
    defer dataset.deinit();

    std.debug.print("dataset loaded({d})\n", .{@as(f32, @floatFromInt(timer.lap())) / @as(f32, @floatFromInt(std.time.ns_per_s))});

    var gpt: nina.model.GPT(T, n_layer) = try .init(
        .{
            .block_size = block_size,
            .n_head = n_head,
            .n_embd = n_embd,
            .vocab_size = tokenizer.decoder_map.len,
        },
        &context,
        base_chain,
    );
    defer gpt.destroy();

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
    context.current_chain = iter_chain;

    _ = try gpt.forward(indices, targets, iter_chain);
    iter_chain.clear();
    indices.clearGrad();
    targets.clearGrad();
    gpt.clearGrads();

    timer.reset();
    try gpt.loadBinary(allocator, savefile);
    std.debug.print("binary loaded({d})\n", .{@as(f32, @floatFromInt(timer.lap())) / @as(f32, @floatFromInt(std.time.ns_per_s))});

    try clearLog();

    std.debug.print("start\n", .{});
    for (0..max_epochs) |epoch| {
        // dataloader.reset();
        var sum_loss: if (T != tomo.BF16) T else f32 = 0.0;
        var sum_acc: if (T != tomo.BF16) T else f32 = 0.0;

        std.debug.print("epoch {} start\n", .{epoch});
        while (try dataloader.writeNextBatch(.{ &indices.asUntagged(usize).data, &targets.asUntagged(usize).data })) |i| {
            std.debug.print("batch {} start\n", .{i});
            // try stream.sync();

            // Forward pass
            std.debug.print("forward start\n", .{});
            timer.reset();
            const logits, const loss = try gpt.forward(indices, targets, iter_chain);
            std.debug.print("forward done({d})\n", .{@as(f32, @floatFromInt(timer.lap())) / @as(f32, @floatFromInt(std.time.ns_per_s))});

            indices.clearGrad();
            targets.clearGrad();
            gpt.clearGrads();

            std.debug.print("backward start\n", .{});
            timer.reset();
            try loss.?.backwardEx(iter_chain);
            std.debug.print("backward done({d})\n", .{@as(f32, @floatFromInt(timer.lap())) / @as(f32, @floatFromInt(std.time.ns_per_s))});

            try optimizer.update(&gpt.getParams());

            var host_loss = try loss.?.asUntagged(T).data.toHost(allocator, &stream);
            defer host_loss.deinit(allocator);

            var pred = try logits.asUntagged(T).data.argmax(context.allocator, &.{2}, true, context.stream);
            defer pred.deinitAsync(context.stream);

            try stream.sync();

            try printBatchDetails(
                T,
                allocator,
                &tokenizer,
                &indices.asUntagged(usize).data,
                &targets.asUntagged(usize).data,
                &logits.asUntagged(T).data,
                epoch,
                i,
                &stream,
            );

            const acc = try tomorin.util.accuracy(T, logits, targets, 2);

            try stream.sync();
            sum_loss += if (T != tomo.BF16) host_loss.at(&.{ 0, 0 }).* else host_loss.at(&.{ 0, 0 }).toF32();
            sum_acc += if (T != tomo.BF16) acc else acc.toF32();

            // try stream.sync();

            std.debug.print("({}/{}) epoch {} loss {d} acc {d}\n", .{
                i,
                dataloader.max_iter,
                epoch + 1,
                host_loss.at(&.{ 0, 0 }).*,
                acc,
            });

            //std.debug.print("base func - {} var - {}\n", .{ base_chain.countFunctions(), base_chain.countVariables() });
            //std.debug.print("iter func - {} var - {}\n", .{ iter_chain.countFunctions(), iter_chain.countVariables() });
            try stream.sync();
            iter_chain.clear();
            //  try stream.sync();
            //std.debug.print("iter func - {} var - {}\n", .{ iter_chain.countFunctions(), iter_chain.countVariables() });

            if (i % 5 == 0) {
                try gpt.saveBinary(allocator, savefile);
                std.debug.print("saved!\n", .{});
            }

            // try stream.sync();
            //  std.Thread.sleep(1 * std.time.ns_per_s);

            std.debug.print("\n", .{});
        }

        const len: if (T != tomo.BF16) T else f32 = @floatFromInt(dataloader.max_iter);

        std.debug.print("epoch {} avg loss {d} acc {d}\n", .{
            epoch + 1,
            sum_loss / len,
            sum_acc / len,
        });
    }
}

fn clearLog() !void {
    var dir = try std.fs.cwd().openDir("log", .{ .iterate = true });
    defer dir.close();

    var it = dir.iterate();
    while (try it.next()) |entry| {
        try dir.deleteFile(entry.name);
    }
}

fn printBatchDetails(
    comptime T: type,
    allocator: std.mem.Allocator,
    tokenizer: *const nina.tokenizer.BpeTokenizer,
    indices: *tomo.tensor.GPUTensor(usize),
    targets: *tomo.tensor.GPUTensor(usize),
    logits: *tomo.tensor.GPUTensor(T),
    epoch: usize,
    i: usize,
    stream: *tomo.stream.Stream,
) !void {
    // Check if batch_index is valid
    const batch_size = indices.base.getShapeConst()[0];
    // if (batch_index >= batch_size) {
    //     std.debug.print("Batch index out of range: {d} >= {d}\n", .{ batch_index, batch_size });
    //     return;
    // }

    const fname = try std.fmt.allocPrint(allocator, "log/log{}_{}.txt", .{ epoch, i });
    defer allocator.free(fname);

    var log = try std.fs.cwd().createFile(fname, .{});
    defer log.close();

    var writer = log.writer();

    for (0..batch_size) |batch_index| {
        var host_indices = try indices.toHost(allocator, stream);
        defer host_indices.deinit(allocator);

        var host_targets = try targets.toHost(allocator, stream);
        defer host_targets.deinit(allocator);

        // Get sequence length
        const seq_len = indices.base.getShapeConst()[1];

        // Extract input sequence for the specified batch
        const input_ids = host_indices.data[batch_index * seq_len .. (batch_index + 1) * seq_len];

        // Extract target sequence for the specified batch
        const target_ids = host_targets.data[batch_index * seq_len .. (batch_index + 1) * seq_len];

        // Compute predicted output IDs from logits using argmax
        var pred = try logits.argmax(allocator, &.{2}, true, stream);
        defer pred.deinitAsync(stream);
        try stream.sync();

        var host_pred = try pred.toHost(allocator, stream);
        defer host_pred.deinit(allocator);

        const vocab_size = tokenizer.decoder_map.len; // Get the tokenizer's vocabulary size
        for (host_pred.data) |*id| {
            if (id.* >= vocab_size) {
                // Replace invalid token IDs with the <unk> token ID
                try writer.print("Warning: Clamping invalid token ID {d} to <unk> ({d})\n", .{ id.*, tokenizer.unk_token_id });
                id.* = tokenizer.unk_token_id; // Ensure this is a valid ID (e.g., 0 or a specific <unk> ID)
            }
        }

        const output_ids = host_pred.data[batch_index * seq_len .. (batch_index + 1) * seq_len];

        const input_decoded = try tokenizer.decodeAlloc(allocator, input_ids);
        defer allocator.free(input_decoded);

        const target_decoded = try tokenizer.decodeAlloc(allocator, target_ids);
        defer allocator.free(target_decoded);

        const output_decoded = try tokenizer.decodeAlloc(allocator, output_ids);
        defer allocator.free(output_decoded);

        // Print in the requested format
        try writer.print(
            // "<batch {d}>\n[input]\n{s}\n[target]\n{s}\n[output]\n{s}\n[output_tokens]\n{any}\n\n"
            \\<batch {d}>
            \\[input]
            \\{s}
            \\
            \\[target]
            \\{s}
            \\
            \\[output]
            \\{s}
            \\
            \\[output_tokens]
            \\{any}
            \\
            \\
        , .{
            batch_index,
            input_decoded,
            target_decoded,
            output_decoded,
            output_ids,
        });
    }

    // var host_logits = try logits.toHost(allocator, stream);
    // defer host_logits.deinit(allocator);
    // std.debug.print("[output_raw]\n{d}\n\n", .{host_logits});
}
