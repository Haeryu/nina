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

    const T = f32;
    const max_epochs = 100;

    // const block_size = 128;
    // const batch_size = 4;
    // const n_layer = 4;
    // const n_embd = 128;
    // const n_head = 4;

    const block_size = 128;
    const batch_size = 16;
    // const batch_size = 1;
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

    // model dynamically allocates weight so you must do dummy forward sorry
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
                &gpt,
                &tokenizer,
                &indices.asUntagged(usize).data,
                &targets.asUntagged(usize).data,
                &logits.asUntagged(T).data,
                epoch,
                i,
                block_size,
                "rise my child",
                &context,
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

            if (std.math.isNan(host_loss.at(&.{ 0, 0 }))) {
                @panic("nan detected");
            }

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
    gpt: anytype,
    tokenizer: *const nina.tokenizer.BpeTokenizer,
    indices: *tomo.tensor.GPUTensor(usize),
    targets: *tomo.tensor.GPUTensor(usize),
    logits: *tomo.tensor.GPUTensor(T),
    epoch: usize,
    i: usize,
    block_size: usize,
    prompt: []const u8,
    context: *tomorin.context.Context,
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
        var host_indices = try indices.toHost(allocator, context.stream);
        defer host_indices.deinit(allocator);

        var host_targets = try targets.toHost(allocator, context.stream);
        defer host_targets.deinit(allocator);

        const seq_len = indices.base.getShapeConst()[1];

        const input_ids = host_indices.data[batch_index * seq_len .. (batch_index + 1) * seq_len];

        const target_ids = host_targets.data[batch_index * seq_len .. (batch_index + 1) * seq_len];

        var pred = try logits.argmax(allocator, &.{2}, true, context.stream);
        defer pred.deinitAsync(context.stream);
        try context.stream.sync();

        var host_pred = try pred.toHost(allocator, context.stream);
        defer host_pred.deinit(allocator);

        const vocab_size = tokenizer.decoder_map.len;
        for (host_pred.data) |*id| {
            if (id.* >= vocab_size) {
                try writer.print("Warning: Clamping invalid token ID {d} to <unk> ({d})\n", .{ id.*, tokenizer.unk_token_id });
                id.* = tokenizer.unk_token_id;
            }
        }

        const output_ids = host_pred.data[batch_index * seq_len .. (batch_index + 1) * seq_len];

        const input_decoded = try tokenizer.decodeAlloc(allocator, input_ids);
        defer allocator.free(input_decoded);

        const target_decoded = try tokenizer.decodeAlloc(allocator, target_ids);
        defer allocator.free(target_decoded);

        const output_decoded = try tokenizer.decodeAlloc(allocator, output_ids);
        defer allocator.free(output_decoded);

        try writer.print(
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
        , .{
            batch_index,
            input_decoded,
            target_decoded,
            output_decoded,
            output_ids,
        });

        _ = gpt;
        _ = block_size;
        _ = prompt;
        // const out = try generatePromptAlloc(
        //     T,
        //     allocator,
        //     gpt,
        //     tokenizer,
        //     prompt,
        //     block_size,
        //     context,
        // );
        // defer allocator.free(out);

        // try writer.print(
        //     \\[prompt]
        //     \\{s}
        //     \\
        //     \\[generated]
        //     \\{s}
        //     \\
        //     \\
        // , .{
        //     prompt,
        //     out,
        // });
    }

    // var host_logits = try logits.toHost(allocator, stream);
    // defer host_logits.deinit(allocator);
    // std.debug.print("[output_raw]\n{d}\n\n", .{host_logits});
}

fn generatePromptAlloc(
    comptime T: type,
    allocator: std.mem.Allocator,
    gpt: anytype,
    tokenizer: *const nina.tokenizer.BpeTokenizer,
    prompt: []const u8,
    block_size: usize,
    context: *tomorin.context.Context,
) ![]u8 {
    const tokens = try tokenizer.encodeAlloc(allocator, prompt);
    defer allocator.free(tokens);

    var generated_tokens = std.ArrayList(usize).init(allocator);
    defer generated_tokens.deinit();

    try generated_tokens.appendSlice(tokens);

    var prompt_base_chain = try context.createChain();
    defer prompt_base_chain.destroy();

    var prompt_iter_chain = try context.createChain();
    defer prompt_iter_chain.destroy();

    var input_tensor = try tomo.tensor.GPUTensor(usize).initAsync(&.{ 1, block_size }, context.stream);
    defer input_tensor.deinitAsync(context.stream);

    const var_input = try prompt_base_chain.createVariable(usize, input_tensor.move(), null);
    defer var_input.destroy();

    const max_tokens = 32;
    while (generated_tokens.items.len < max_tokens) {
        // defer context.stream.sync() catch unreachable;
        //  try input_tensor.fill(std.math.maxInt(usize), context.stream);

        const input_len = @min(generated_tokens.items.len, block_size);
        const input_slice = generated_tokens.items[generated_tokens.items.len - input_len ..];

        try var_input.writeFromHost(usize, input_slice, 0);

        const logits, _ = try gpt.forward(var_input, null, prompt_iter_chain);
        defer logits.destroy();

        var pred = try logits.asUntagged(T).data.argmax(context.allocator, &.{2}, true, context.stream);
        defer pred.deinitAsync(context.stream);

        try context.stream.sync();
        var host_pred = try pred.toHost(allocator, context.stream);
        defer host_pred.deinit(allocator);

        try context.stream.sync();

        const new_token = host_pred.data[0];

        try generated_tokens.append(new_token);

        if (new_token == tokenizer.encoder_map.get("<|endoftext|>").?) break;

        prompt_iter_chain.clear();
    }

    const result = try tokenizer.decodeAlloc(allocator, generated_tokens.items);

    return result;
}
