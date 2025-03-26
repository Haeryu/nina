const std = @import("std");

// const encoder_data = @import("encoder_data.zig").encoder_data;
// const decoder_map = @import("decoder_map.zig").decoder_map;
// const merge_map = @import("merge_map.zig").merge_map;
// const encoder_map = @import("encoder_map.zig").encoder_map;

pub const BpeTokenizer = struct {
    merge_map: *const std.StaticStringMap(usize),
    encoder_map: *const std.StaticStringMap(usize),
    decoder_map: []const []const u8,

    pub fn initCustom(
        merge_map: *const std.StaticStringMap(usize),
        encoder_map: *const std.StaticStringMap(usize),
        decoder_map: []const []const u8,
    ) BpeTokenizer {
        return .{
            .merge_map = merge_map,
            .encoder_map = encoder_map,
            .decoder_map = decoder_map,
        };
    }

    const default: BpeTokenizer = .{
        .merge_map = &@import("merge_map.zig").merge_map,
        .encoder_map = &@import("encoder_map.zig").encoder_map,
        .decoder_map = &@import("decoder_map.zig").decoder_map,
    };

    pub fn encodeAlloc(self: *BpeTokenizer, allocator: std.mem.Allocator, input: []const u8) ![]usize {
        const tokens = try self.applyBpeAlloc(allocator, input);
        defer {
            for (tokens) |t| allocator.free(t);
            allocator.free(tokens);
        }

        var token_ids = try std.ArrayList(usize).initCapacity(allocator, tokens.len);
        defer token_ids.deinit();

        for (tokens) |token| {
            if (self.encoder_map.get(token)) |id| {
                try token_ids.append(@intCast(id));
            } else {
                return error.TokenNotFound;
            }
        }

        return token_ids.toOwnedSlice();
    }

    pub fn decodeAlloc(self: *BpeTokenizer, allocator: std.mem.Allocator, ids: []const usize) ![]u8 {
        var decoded = std.ArrayList(u8).init(allocator);
        defer decoded.deinit();

        for (ids) |id| {
            if (id >= self.decoder_map.len) return error.TokenIdOutOfBounds;
            const token = self.decoder_map[id];
            try decoded.appendSlice(token);
        }

        return try decoded.toOwnedSlice();
    }

    pub fn applyBpeAlloc(self: *BpeTokenizer, allocator: std.mem.Allocator, input: []const u8) ![][]const u8 {
        var tokens = try std.ArrayList([]const u8).initCapacity(allocator, input.len);

        defer {
            for (tokens.items) |t| allocator.free(t);
            defer tokens.deinit();
        }

        for (input) |b| {
            const slice = try allocator.alloc(u8, 1);
            errdefer allocator.free(slice);
            slice[0] = b;
            try tokens.append(slice);
        }

        while (true) {
            var best_rank: usize = std.math.maxInt(usize);
            var best_index: ?usize = null;

            var it = std.mem.window([]const u8, tokens.items, 2, 1);
            var i: usize = 0;
            while (it.next()) |pair| : (i += 1) {
                const joined = try std.mem.concat(allocator, u8, &.{ pair[0], " ", pair[1] });
                defer allocator.free(joined);

                if (self.merge_map.get(joined)) |rank| {
                    if (rank < best_rank) {
                        best_rank = rank;
                        best_index = i;
                    }
                }
            }

            if (best_index == null) break;

            const i_merge = best_index.?;
            const merged = try std.mem.concat(allocator, u8, &.{ tokens.items[i_merge], tokens.items[i_merge + 1] });
            errdefer allocator.free(merged);

            allocator.free(tokens.items[i_merge]);
            allocator.free(tokens.items[i_merge + 1]);
            tokens.items[i_merge] = merged;
            _ = tokens.orderedRemove(i_merge + 1);
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

    var encoder_map = std.StaticStringMap(usize).initComptime(&[_]struct { []const u8, usize }{});

    var tokenizer: BpeTokenizer = .initCustom(&merge_map, &encoder_map, &.{});

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

test "encodeAlloc works correctly" {
    const allocator = std.testing.allocator;

    // Define merge_map to merge "a b" and "b c"
    var merge_map = std.StaticStringMap(usize).initComptime(.{
        .{ "a b", 0 },
        .{ "b c", 1 },
    });

    // Define encoder_map to map tokens "ab", "c", "d" to IDs
    var encoder_map = std.StaticStringMap(usize).initComptime(.{
        .{ "ab", 0 },
        .{ "c", 1 },
        .{ "d", 2 },
    });

    // Define decoder_map (not used in this test, but required for init)
    const decoder_map = &[_][]const u8{ "ab", "c", "d" };

    // Initialize tokenizer
    var tokenizer = BpeTokenizer.initCustom(&merge_map, &encoder_map, decoder_map);

    // Input to encode
    const input = "abcd";
    const token_ids = try tokenizer.encodeAlloc(allocator, input);
    defer allocator.free(token_ids);

    // Expected IDs based on tokens "ab", "c", "d" -> [0, 1, 2]
    const expected_ids = [_]usize{ 0, 1, 2 };
    try std.testing.expectEqualSlices(usize, &expected_ids, token_ids);
}

// Test that decodeAlloc correctly reconstructs the string from IDs
test "decodeAlloc works correctly" {
    const allocator = std.testing.allocator;

    var merge_map = std.StaticStringMap(usize).initComptime(.{
        .{ "a b", 0 },
        .{ "b c", 1 },
    });

    var encoder_map = std.StaticStringMap(usize).initComptime(.{
        .{ "ab", 0 },
        .{ "c", 1 },
        .{ "d", 2 },
    });

    const decoder_map = &[_][]const u8{ "ab", "c", "d" };

    var tokenizer = BpeTokenizer.initCustom(&merge_map, &encoder_map, decoder_map);

    // Input IDs to decode
    const ids = [_]usize{ 0, 1, 2 };
    const decoded = try tokenizer.decodeAlloc(allocator, &ids);
    defer allocator.free(decoded);

    // Expect "abcd" from concatenating "ab", "c", "d"
    try std.testing.expectEqualSlices(u8, "abcd", decoded);
}

// Test that encoding and then decoding returns the original input
test "encodeAlloc and decodeAlloc round-trip" {
    const allocator = std.testing.allocator;

    var merge_map = std.StaticStringMap(usize).initComptime(.{
        .{ "a b", 0 },
        .{ "b c", 1 },
    });

    var encoder_map = std.StaticStringMap(usize).initComptime(.{
        .{ "ab", 0 },
        .{ "c", 1 },
        .{ "d", 2 },
    });

    const decoder_map = &[_][]const u8{ "ab", "c", "d" };

    var tokenizer = BpeTokenizer.initCustom(&merge_map, &encoder_map, decoder_map);

    const input = "abcd";
    const token_ids = try tokenizer.encodeAlloc(allocator, input);
    defer allocator.free(token_ids);

    const decoded = try tokenizer.decodeAlloc(allocator, token_ids);
    defer allocator.free(decoded);

    try std.testing.expectEqualSlices(u8, input, decoded);
}

// Test that encodeAlloc returns an error when a token is not in encoder_map
test "encodeAlloc handles token not found" {
    const allocator = std.testing.allocator;

    var merge_map = std.StaticStringMap(usize).initComptime(.{
        .{ "a b", 0 },
    });

    // encoder_map missing "c" and "d"
    var encoder_map = std.StaticStringMap(usize).initComptime(.{
        .{ "ab", 0 },
    });

    const decoder_map = &[_][]const u8{"ab"};

    var tokenizer = BpeTokenizer.initCustom(&merge_map, &encoder_map, decoder_map);

    const input = "abcd"; // Will produce "ab", "c", "d", but "c" and "d" are not in encoder_map
    const result = tokenizer.encodeAlloc(allocator, input);
    try std.testing.expectError(error.TokenNotFound, result);
}

// Test that decodeAlloc returns an error when an ID is out of bounds
test "decodeAlloc handles ID out of bounds" {
    const allocator = std.testing.allocator;

    var merge_map = std.StaticStringMap(usize).initComptime(.{});

    var encoder_map = std.StaticStringMap(usize).initComptime(.{});

    const decoder_map = &[_][]const u8{ "ab", "c", "d" }; // Length 3, indices 0-2 valid

    var tokenizer = BpeTokenizer.initCustom(&merge_map, &encoder_map, decoder_map);

    const ids = [_]usize{ 0, 1, 3 }; // 3 is out of bounds (decoder_map.len = 3)
    const result = tokenizer.decodeAlloc(allocator, &ids);
    try std.testing.expectError(error.TokenIdOutOfBounds, result);
}
