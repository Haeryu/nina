const std = @import("std");

pub const BpeTokenizer = struct {
    merge_map: *const std.StaticStringMap(usize),
    encoder_map: *const std.StaticStringMap(usize),
    decoder_map: []const []const u8,
    unk_token_id: usize,

    pub fn initCustom(
        merge_map: *const std.StaticStringMap(usize),
        encoder_map: *const std.StaticStringMap(usize),
        decoder_map: []const []const u8,
    ) !BpeTokenizer {
        var self: BpeTokenizer = .{
            .merge_map = merge_map,
            .encoder_map = encoder_map,
            .decoder_map = decoder_map,
            .unk_token_id = 0,
        };
        self.unk_token_id = self.encoder_map.get("<unk>") orelse return error.UnkTokenNotFound;
        return self;
    }

    pub fn init() !BpeTokenizer {
        var self: BpeTokenizer = .{
            .merge_map = &@import("merge_map.zig").merge_map,
            .encoder_map = &@import("encoder_map.zig").encoder_map,
            .decoder_map = &@import("decoder_map.zig").decoder_map,
            .unk_token_id = 0,
        };
        self.unk_token_id = self.encoder_map.get("<unk>") orelse return error.UnkTokenNotFound;
        return self;
    }

    pub fn encodeAlloc(self: *const BpeTokenizer, allocator: std.mem.Allocator, input: []const u8) ![]usize {
        const tokens = try self.applyBpeAlloc(allocator, input);
        defer {
            for (tokens) |t| allocator.free(t);
            allocator.free(tokens);
        }

        var token_ids = try std.ArrayList(usize).initCapacity(allocator, tokens.len);
        defer token_ids.deinit();

        for (tokens) |token| {
            const id = self.encoder_map.get(token) orelse self.unk_token_id;
            try token_ids.append(id);
        }

        return token_ids.toOwnedSlice();
    }

    pub fn decodeAlloc(self: *const BpeTokenizer, allocator: std.mem.Allocator, ids: []const usize) ![]u8 {
        var decoded = std.ArrayList(u8).init(allocator);
        defer decoded.deinit();

        for (ids) |id| {
            if (id >= self.decoder_map.len) return error.TokenIdOutOfBounds;
            const token = self.decoder_map[id];
            try decoded.appendSlice(token);
        }

        return try decoded.toOwnedSlice();
    }

    pub fn applyBpeAlloc(self: *const BpeTokenizer, allocator: std.mem.Allocator, input: []const u8) ![][]const u8 {
        // Initialize token list with individual bytes
        var tokens = try std.ArrayList([]const u8).initCapacity(allocator, input.len);
        defer {
            for (tokens.items) |t| allocator.free(t);
            tokens.deinit();
        }

        for (input) |b| {
            const slice = try allocator.alloc(u8, 1);
            errdefer allocator.free(slice);
            slice[0] = b;
            try tokens.append(slice);
        }

        // Define merge struct for the priority queue
        const Merge = struct {
            rank: usize,
            index: usize,
        };

        // Define heap context for min-heap behavior
        const HeapContext = struct {
            pub fn lessThan(_: void, a: Merge, b: Merge) std.math.Order {
                return std.math.order(a.rank, b.rank);
            }
        };

        // Initialize min-heap for merges
        var heap = std.PriorityQueue(Merge, void, HeapContext.lessThan).init(allocator, {});
        defer heap.deinit();

        // Reusable buffer for building pair strings
        var buffer = std.ArrayList(u8).init(allocator);
        defer buffer.deinit();

        // Populate heap with initial adjacent pairs
        var it = std.mem.window([]const u8, tokens.items, 2, 1);
        var idx: usize = 0;
        while (it.next()) |pair| : (idx += 1) {
            if (pair.len != 2) break;
            buffer.clearRetainingCapacity();
            try buffer.appendSlice(pair[0]);
            try buffer.append(' ');
            try buffer.appendSlice(pair[1]);
            if (self.merge_map.get(buffer.items)) |rank| {
                try heap.add(.{ .rank = rank, .index = idx });
            }
        }

        // Process merges using the heap
        while (heap.count() > 0) {
            const merge = heap.remove();
            const i = merge.index;

            // Skip if index is invalid due to previous merges
            if (i >= tokens.items.len - 1) continue;

            // Verify the pair still matches the merge rank
            buffer.clearRetainingCapacity();
            try buffer.appendSlice(tokens.items[i]);
            try buffer.append(' ');
            try buffer.appendSlice(tokens.items[i + 1]);
            if (self.merge_map.get(buffer.items)) |rank| {
                if (rank != merge.rank) continue;
            } else {
                continue;
            }

            // Perform the merge
            const merged = try std.mem.concat(allocator, u8, &.{ tokens.items[i], tokens.items[i + 1] });
            allocator.free(tokens.items[i]);
            allocator.free(tokens.items[i + 1]);
            tokens.items[i] = merged;
            _ = tokens.orderedRemove(i + 1);

            // Add new potential merges to the heap
            if (i > 0) {
                buffer.clearRetainingCapacity();
                try buffer.appendSlice(tokens.items[i - 1]);
                try buffer.append(' ');
                try buffer.appendSlice(tokens.items[i]);
                if (self.merge_map.get(buffer.items)) |rank| {
                    try heap.add(.{ .rank = rank, .index = i - 1 });
                }
            }
            if (i < tokens.items.len - 1) {
                buffer.clearRetainingCapacity();
                try buffer.appendSlice(tokens.items[i]);
                try buffer.append(' ');
                try buffer.appendSlice(tokens.items[i + 1]);
                if (self.merge_map.get(buffer.items)) |rank| {
                    try heap.add(.{ .rank = rank, .index = i });
                }
            }
        }

        return try tokens.toOwnedSlice();
    }
};

test "applyBpeAlloc merges tokens correctly" {
    const allocator = std.testing.allocator;

    // Initialize merge_map
    var merge_map = std.StaticStringMap(usize).initComptime(.{
        .{ "a b", 0 },
        .{ "b c", 1 },
    });

    var encoder_map = std.StaticStringMap(usize).initComptime(&[_]struct { []const u8, usize }{
        .{ "<unk>", 0 },
    });

    var tokenizer = try BpeTokenizer.initCustom(&merge_map, &encoder_map, &.{});

    const input = "abcd"; // Bytes: [97, 98, 99, 100]
    const tokens = try tokenizer.applyBpeAlloc(allocator, input);
    defer {
        for (tokens) |t| allocator.free(t);
        allocator.free(tokens);
    }

    try std.testing.expectEqual(@as(usize, 3), tokens.len);
    try std.testing.expectEqualSlices(u8, "ab", tokens[0]);
    try std.testing.expectEqualSlices(u8, "c", tokens[1]);
    try std.testing.expectEqualSlices(u8, "d", tokens[2]);
}

test "encodeAlloc works correctly with known tokens" {
    const allocator = std.testing.allocator;

    // Define merge_map to merge "a b" and "b c"
    var merge_map = std.StaticStringMap(usize).initComptime(.{
        .{ "a b", 0 },
        .{ "b c", 1 },
    });

    // Define encoder_map with "<unk>" and token mappings
    var encoder_map = std.StaticStringMap(usize).initComptime(.{
        .{ "<unk>", 0 }, // Required for initCustom
        .{ "ab", 1 },
        .{ "c", 2 },
        .{ "d", 3 },
    });

    // Define decoder_map
    const decoder_map = &[_][]const u8{ "<unk>", "ab", "c", "d" };

    // Initialize tokenizer
    var tokenizer = try BpeTokenizer.initCustom(&merge_map, &encoder_map, decoder_map);

    // Input to encode
    const input = "abcd";
    const token_ids = try tokenizer.encodeAlloc(allocator, input);
    defer allocator.free(token_ids);

    // Expected IDs: "ab" (1), "c" (2), "d" (3)
    const expected_ids = [_]usize{ 1, 2, 3 };
    try std.testing.expectEqualSlices(usize, &expected_ids, token_ids);
}

test "encodeAlloc handles unknown tokens with unk_token_id" {
    const allocator = std.testing.allocator;

    // Define merge_map
    var merge_map = std.StaticStringMap(usize).initComptime(.{
        .{ "a b", 0 },
    });

    // Define encoder_map with "<unk>" but missing "c" and "d"
    var encoder_map = std.StaticStringMap(usize).initComptime(.{
        .{ "<unk>", 0 }, // unk_token_id will be 0
        .{ "ab", 1 },
    });

    // Define decoder_map
    const decoder_map = &[_][]const u8{ "<unk>", "ab" };

    // Initialize tokenizer
    var tokenizer = try BpeTokenizer.initCustom(&merge_map, &encoder_map, decoder_map);

    // Input to encode: "abcd" -> "ab", "c", "d" but "c" and "d" are unknown
    const input = "abcd";
    const token_ids = try tokenizer.encodeAlloc(allocator, input);
    defer allocator.free(token_ids);

    // Expected: "ab" (1), "<unk>" (0), "<unk>" (0)
    const expected_ids = [_]usize{ 1, 0, 0 };
    try std.testing.expectEqualSlices(usize, &expected_ids, token_ids);
}

test "decodeAlloc works correctly" {
    const allocator = std.testing.allocator;

    var merge_map = std.StaticStringMap(usize).initComptime(.{
        .{ "a b", 0 },
        .{ "b c", 1 },
    });

    var encoder_map = std.StaticStringMap(usize).initComptime(.{
        .{ "<unk>", 0 },
        .{ "ab", 1 },
        .{ "c", 2 },
        .{ "d", 3 },
    });

    const decoder_map = &[_][]const u8{ "<unk>", "ab", "c", "d" };

    var tokenizer = try BpeTokenizer.initCustom(&merge_map, &encoder_map, decoder_map);

    // Input IDs to decode
    const ids = [_]usize{ 1, 2, 3, 0 }; // "ab", "c", "d", "<unk>"
    const decoded = try tokenizer.decodeAlloc(allocator, &ids);
    defer allocator.free(decoded);

    // Expect "abcd<unk>" from concatenating "ab", "c", "d", "<unk>"
    try std.testing.expectEqualSlices(u8, "abcd<unk>", decoded);
}

test "encodeAlloc and decodeAlloc round-trip with unknown tokens" {
    const allocator = std.testing.allocator;

    var merge_map = std.StaticStringMap(usize).initComptime(.{
        .{ "a b", 0 },
    });

    var encoder_map = std.StaticStringMap(usize).initComptime(.{
        .{ "<unk>", 0 }, // unk_token_id will be 0
        .{ "ab", 1 },
    });

    const decoder_map = &[_][]const u8{ "<unk>", "ab" };

    var tokenizer = try BpeTokenizer.initCustom(&merge_map, &encoder_map, decoder_map);

    const input = "abcd"; // "ab" is known, "c" and "d" are unknown
    const token_ids = try tokenizer.encodeAlloc(allocator, input);
    defer allocator.free(token_ids);

    const decoded = try tokenizer.decodeAlloc(allocator, token_ids);
    defer allocator.free(decoded);

    // Expected tokens: "ab" (1), "<unk>" (0), "<unk>" (0) -> "ab<unk><unk>"
    try std.testing.expectEqualSlices(u8, "ab<unk><unk>", decoded);
}

test "decodeAlloc handles ID out of bounds" {
    const allocator = std.testing.allocator;

    var merge_map = std.StaticStringMap(usize).initComptime(.{});

    var encoder_map = std.StaticStringMap(usize).initComptime(.{
        .{ "<unk>", 0 },
    });

    const decoder_map = &[_][]const u8{ "<unk>", "ab", "c", "d" }; // Length 4, indices 0-3 valid

    var tokenizer = try BpeTokenizer.initCustom(&merge_map, &encoder_map, decoder_map);

    const ids = [_]usize{ 0, 1, 4 }; // 4 is out of bounds
    const result = tokenizer.decodeAlloc(allocator, &ids);
    try std.testing.expectError(error.TokenIdOutOfBounds, result);
}
