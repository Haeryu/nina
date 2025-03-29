const std = @import("std");
const BpeTokenizer = @import("./tokenizer/tokenizer.zig").BpeTokenizer; // Adjust path as needed
const GPUTensor = @import("tomo").tensor.GPUTensor;
const Context = @import("tomorin").context.Context;

pub const GPT2Dataset = struct {
    allocator: std.mem.Allocator,
    tokenizer: *const BpeTokenizer,
    sequence_length: usize,
    token_ids: []usize,
    total_sequences: usize,
    pad_token_id: usize, // Added to store the <pad> token ID
    input_buffer: []usize,
    target_buffer: []usize,

    const Self = @This();

    pub fn init(
        allocator: std.mem.Allocator,
        tokenizer: *const BpeTokenizer,
        sequence_length: usize,
        token_paths: []const []const u8,
    ) !Self {
        var token_ids_list = std.ArrayList(usize).init(allocator);
        defer token_ids_list.deinit();

        for (token_paths) |path| {
            var file = try std.fs.cwd().openFile(path, .{});
            defer file.close();

            const reader = file.reader();

            const count = try reader.readInt(usize, .little);

            try token_ids_list.appendNTimes(0, count);

            const tok_end = token_ids_list.items.len;

            try reader.readNoEof(std.mem.sliceAsBytes(token_ids_list.items[tok_end - count .. tok_end]));
        }

        const token_ids = try token_ids_list.toOwnedSlice();
        errdefer allocator.free(token_ids);
        const total_sequences = try std.math.divCeil(usize, token_ids.len, sequence_length);
        const pad_token_id = tokenizer.encoder_map.get("<pad>") orelse return error.PadTokenNotFound;

        const input_buffer = try allocator.alloc(usize, sequence_length);
        errdefer allocator.free(input_buffer);
        const target_buffer = try allocator.alloc(usize, sequence_length);
        errdefer allocator.free(target_buffer);

        return .{
            .allocator = allocator,
            .tokenizer = tokenizer,
            .sequence_length = sequence_length,
            .token_ids = token_ids,
            .total_sequences = total_sequences,
            .pad_token_id = pad_token_id,
            .input_buffer = input_buffer,
            .target_buffer = target_buffer,
        };
    }

    pub fn deinit(self: *Self) void {
        self.allocator.free(self.token_ids);
        self.allocator.free(self.input_buffer);
        self.allocator.free(self.target_buffer);
    }

    pub fn len(self: *Self) usize {
        return self.total_sequences;
    }

    pub fn write(
        self: *Self,
        i: usize,
        batch_i: usize,
        batch: std.meta.Tuple(&.{ *GPUTensor(usize), *GPUTensor(usize) }),
        context: *Context,
    ) !void {
        const input_tensor, const target_tensor = batch;
        const start = batch_i * self.sequence_length;
        const end = @min(start + self.sequence_length, self.token_ids.len);

        // Use pre-allocated buffers
        const input_sequence = self.input_buffer;
        const target_sequence = self.target_buffer;

        const input_len = end - start;
        if (input_len > 0) {
            @memcpy(input_sequence[0..input_len], self.token_ids[start..end]);
        }
        for (input_len..self.sequence_length) |k| {
            input_sequence[k] = self.pad_token_id;
        }
        try input_tensor.writeFromHostAsync(
            input_sequence,
            i * self.sequence_length,
            context.stream,
        );

        // const input_decoded = try self.tokenizer.decodeAlloc(self.allocator, input_sequence);
        // defer self.allocator.free(input_decoded);
        // std.debug.print("[input] - {s}\n", .{input_decoded});

        const ignore_index: usize = std.math.maxInt(usize);
        for (0..self.sequence_length) |k| {
            const target_pos = start + k + 1;
            if (target_pos < self.token_ids.len) {
                target_sequence[k] = self.token_ids[target_pos];
            } else {
                target_sequence[k] = ignore_index;
            }
        }
        try target_tensor.writeFromHostAsync(
            target_sequence,
            i * self.sequence_length,
            context.stream,
        );

        // const target_decoded = try self.tokenizer.decodeAlloc(self.allocator, target_sequence);
        // defer self.allocator.free(target_decoded);
        // std.debug.print("[target] - {s}\n", .{target_decoded});
    }
};
