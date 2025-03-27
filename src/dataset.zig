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

    const Self = @This();

    pub fn init(
        allocator: std.mem.Allocator,
        tokenizer: *const BpeTokenizer,
        sequence_length: usize,
        file_paths: []const []const u8,
    ) !Self {
        var token_ids = try allocator.alloc(usize, 0);
        errdefer allocator.free(token_ids);

        for (file_paths) |path| {
            var file = try std.fs.cwd().openFile(path, .{});
            defer file.close();

            const content = try file.readToEndAlloc(allocator, std.math.maxInt(usize));
            defer allocator.free(content);

            const encoded = try tokenizer.encodeAlloc(allocator, content);
            defer allocator.free(encoded);

            const new_len = token_ids.len + encoded.len;
            token_ids = try allocator.realloc(token_ids, new_len);
            @memcpy(token_ids[token_ids.len - encoded.len ..], encoded);
        }

        // Use ceiling division to account for partial sequences
        const total_sequences = (token_ids.len + sequence_length - 1) / sequence_length;

        // Retrieve the <pad> token ID from the tokenizer
        const pad_token_id = tokenizer.encoder_map.get("<pad>") orelse return error.PadTokenNotFound;

        return .{
            .allocator = allocator,
            .tokenizer = tokenizer,
            .sequence_length = sequence_length,
            .token_ids = token_ids,
            .total_sequences = total_sequences,
            .pad_token_id = pad_token_id,
        };
    }

    pub fn deinit(self: *Self) void {
        self.allocator.free(self.token_ids);
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

        // Calculate sequence boundaries
        const start = batch_i * self.sequence_length;
        const end = @min(start + self.sequence_length, self.token_ids.len);

        // Prepare input sequence with padding
        var input_sequence = try self.allocator.alloc(usize, self.sequence_length);
        defer self.allocator.free(input_sequence);

        const input_len = end - start; // Actual number of tokens available
        if (input_len > 0) {
            @memcpy(input_sequence[0..input_len], self.token_ids[start..end]);
        }
        // Pad the rest with <pad> token
        for (input_len..self.sequence_length) |k| {
            input_sequence[k] = self.pad_token_id;
        }

        // Write input sequence to tensor
        try input_tensor.writeFromHostAsync(
            input_sequence,
            i * self.sequence_length,
            context.stream,
        );

        // Prepare target sequence with padding
        var target_sequence = try self.allocator.alloc(usize, self.sequence_length);
        defer self.allocator.free(target_sequence);

        const ignore_index: usize = std.math.maxInt(usize);

        // Fill target sequence
        for (0..self.sequence_length) |k| {
            const target_pos = start + k + 1;
            if (target_pos < self.token_ids.len) {
                target_sequence[k] = self.token_ids[target_pos];
            } else {
                target_sequence[k] = ignore_index; // Pad with ignore_index
            }
        }

        // Write target sequence to tensor
        try target_tensor.writeFromHostAsync(
            target_sequence,
            i * self.sequence_length,
            context.stream,
        );
    }
};
