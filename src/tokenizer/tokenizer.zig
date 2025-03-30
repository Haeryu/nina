const std = @import("std");

pub const BpeTokenizer = struct {
    merge_map: *const std.StaticStringMap(usize),
    encoder_map: *const std.StaticStringMap(usize),
    decoder_map: []const []const u8,
    unk_token_id: usize,

    fn preprocess(allocator: std.mem.Allocator, input: []const u8) ![]u8 {
        var result: std.ArrayList(u8) = .init(allocator);
        defer result.deinit();

        var iter = (try std.unicode.Utf8View.init(input)).iterator();
        while (iter.nextCodepointSlice()) |cp| {
            if (std.mem.eql(u8, cp, " ")) {
                try result.appendSlice("Ġ");
            } else if (std.mem.eql(u8, cp, "\n")) {
                try result.appendSlice("Ċ");
            }
            // } else if (std.mem.eql(u8, cp, "\r")) {
            //     continue;
            // }
            else {
                try result.appendSlice(cp);
            }
        }

        return result.toOwnedSlice();
    }

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
        const preprocessed = try preprocess(allocator, input);
        defer allocator.free(preprocessed);

        // const tokens = try self.applyBpeAlloc(allocator, input);
        const tokens = try self.applyBpeAlloc(allocator, preprocessed);
        defer {
            for (tokens) |t| {
                allocator.free(t);
            }
            allocator.free(tokens);
        }

        var token_ids: std.ArrayList(usize) = try .initCapacity(allocator, tokens.len);
        defer token_ids.deinit();

        for (tokens) |token| {
            const id = self.encoder_map.get(token) orelse self.unk_token_id;
            try token_ids.append(id);
        }

        return token_ids.toOwnedSlice();
    }

    pub fn decodeAlloc(self: *const BpeTokenizer, allocator: std.mem.Allocator, ids: []const usize) ![]u8 {
        var decoded: std.ArrayList(u8) = .init(allocator);
        defer decoded.deinit();

        for (ids) |id| {
            //if (id >= self.decoder_map.len) return error.TokenIdOutOfBounds;
            if (id >= self.decoder_map.len) {
                try decoded.appendSlice("<unk>");
                continue;
            }

            const token = self.decoder_map[id];

            if (std.mem.startsWith(u8, token, "Ġ")) {
                try decoded.appendSlice(" ");
                try decoded.appendSlice(token["Ġ".len..]);
            } else if (std.mem.startsWith(u8, token, "Ċ")) {
                try decoded.appendSlice("\n");
                try decoded.appendSlice(token["Ċ".len..]);
            } else {
                try decoded.appendSlice(token);
            }
        }

        return try decoded.toOwnedSlice();
    }

    // TODO: test
    pub fn applyBpeAlloc(self: *const BpeTokenizer, allocator: std.mem.Allocator, input: []const u8) ![][]const u8 {
        var tokens = try std.ArrayList([]const u8).initCapacity(allocator, input.len);
        defer tokens.deinit();

        var iter = (try std.unicode.Utf8View.init(input)).iterator();
        while (iter.nextCodepointSlice()) |cp| {
            const owned_cp = try allocator.dupe(u8, cp);
            try tokens.append(owned_cp);
        }

        const Merge = struct {
            rank: usize,
            index: usize,
        };

        const HeapContext = struct {
            pub fn lessThan(_: void, a: Merge, b: Merge) std.math.Order {
                return std.math.order(a.rank, b.rank);
            }
        };

        var heap: std.PriorityQueue(Merge, void, HeapContext.lessThan) = .init(allocator, {});
        defer heap.deinit();

        var buffer = std.ArrayList(u8).init(allocator);
        defer buffer.deinit();

        const addMerges = struct {
            fn add(
                hp: *std.PriorityQueue(Merge, void, HeapContext.lessThan),
                ts: *std.ArrayList([]const u8),
                merge_map: *const std.StaticStringMap(usize),
                buf: *std.ArrayList(u8),
            ) !void {
                var it = std.mem.window([]const u8, ts.items, 2, 1);
                var i: usize = 0;
                while (it.next()) |pair| : (i += 1) {
                    if (pair.len < 2) break;

                    buf.clearRetainingCapacity();
                    try buf.appendSlice(pair[0]);
                    try buf.append(' ');
                    try buf.appendSlice(pair[1]);
                    if (merge_map.get(buf.items)) |rank| {
                        try hp.add(.{ .rank = rank, .index = i });
                    }
                }
            }
        }.add;

        try addMerges(&heap, &tokens, self.merge_map, &buffer);

        while (heap.count() > 0) {
            const merge = heap.remove();
            const i = merge.index;

            if (i >= tokens.items.len - 1) {
                continue;
            }

            buffer.clearRetainingCapacity();
            try buffer.appendSlice(tokens.items[i]);
            try buffer.append(' ');
            try buffer.appendSlice(tokens.items[i + 1]);
            if (self.merge_map.get(buffer.items)) |rank| {
                if (rank != merge.rank) {
                    continue;
                }
            } else {
                continue;
            }

            const old_i = tokens.items[i];
            const old_i_plus_1 = tokens.items[i + 1];
            const merged = try std.mem.concat(allocator, u8, &.{ old_i, old_i_plus_1 });
            allocator.free(old_i);
            allocator.free(old_i_plus_1);
            tokens.items[i] = merged;
            _ = tokens.orderedRemove(i + 1);

            heap.clearRetainingCapacity();
            try addMerges(&heap, &tokens, self.merge_map, &buffer);
        }

        return try tokens.toOwnedSlice();
    }
};

// Test 1: Initialization with Custom Maps
test "Initialization with custom maps" {

    // Mock maps
    const merge_map = std.StaticStringMap(usize).initComptime(&.{
        .{ "a b", 1 },
        .{ "b c", 2 },
    });

    const encoder_map = std.StaticStringMap(usize).initComptime(&.{
        .{ "a", 0 },
        .{ "b", 1 },
        .{ "c", 2 },
        .{ "<unk>", 3 },
    });

    const decoder_map = &[_][]const u8{ "a", "b", "c", "<unk>" };

    const tokenizer = try BpeTokenizer.initCustom(&merge_map, &encoder_map, decoder_map);
    try std.testing.expectEqual(3, tokenizer.unk_token_id); // Verify unk_token_id is set correctly
}

// Test 2: Preprocessing
test "Preprocessing" {
    const allocator = std.testing.allocator;
    const input = "Hello world\nThis is a test";
    const expected = "HelloĠworldĊThisĠisĠaĠtest";

    const preprocessed = try BpeTokenizer.preprocess(allocator, input);
    defer allocator.free(preprocessed);

    try std.testing.expectEqualStrings(expected, preprocessed); // Check space and newline replacements
}

// Test 3: Encoding
test "Encoding" {
    const allocator = std.testing.allocator;

    // Mock maps
    const merge_map = std.StaticStringMap(usize).initComptime(&.{
        .{ "H e", 1 },
        .{ "l l", 2 },
    });

    const encoder_map = std.StaticStringMap(usize).initComptime(&.{
        .{ "H", 0 },   .{ "e", 1 }, .{ "l", 2 }, .{ "o", 3 },     .{ "Ġ", 4 },
        .{ "w", 5 },   .{ "r", 6 }, .{ "d", 7 }, .{ "<unk>", 8 }, .{ "He", 9 },
        .{ "ll", 10 },
    });

    const decoder_map = &[_][]const u8{ "H", "e", "l", "o", "Ġ", "w", "r", "d", "<unk>" };

    const tokenizer = try BpeTokenizer.initCustom(&merge_map, &encoder_map, decoder_map);

    const input = "Hello world";
    const expected_ids = &[_]usize{ 9, 10, 3, 4, 5, 3, 6, 2, 7 }; // "He ll o Ġ w o r l d"

    const ids = try tokenizer.encodeAlloc(allocator, input);
    defer allocator.free(ids);

    try std.testing.expectEqualSlices(usize, expected_ids, ids);
}

test "Encoding2" {
    const allocator = std.testing.allocator;

    // Expanded merge map with additional rules
    const merge_map = std.StaticStringMap(usize).initComptime(&.{
        .{ "H e", 1 }, // Merge "H e" into "He"
        .{ "l l", 2 }, // Merge "l l" into "ll"
        .{ "h e", 3 }, // Merge "h e" into "he"
        .{ "o r", 4 }, // Merge "o r" into "or"
    });

    // Expanded encoder map with new tokens and special characters
    const encoder_map = std.StaticStringMap(usize).initComptime(&.{
        .{ "H", 0 }, .{ "e", 1 }, .{ "l", 2 }, .{ "o", 3 }, .{ "Ġ", 4 }, // Basic tokens and space marker
        .{ "w", 5 }, .{ "r", 6 }, .{ "d", 7 }, .{ "<unk>", 8 }, .{ "He", 9 }, // Original merges and unknown token
        .{ "ll", 10 }, .{ "h", 11 }, .{ "he", 12 }, .{ "or", 13 }, .{ "C", 14 }, // New merges and additional tokens
        .{ "a", 15 },  .{ "f", 16 }, .{ ",", 17 },  .{ "!", 18 },
    });

    // Decoder map updated for consistency (not used in encoding but included for completeness)
    const decoder_map = &[_][]const u8{
        "H",  "e", "l",  "o",  "Ġ", "w", "r", "d", "<unk>", "He",
        "ll", "h", "he", "or", "C",  "a", "f", ",", "!",
    };

    // Initialize the tokenizer with custom maps
    const tokenizer = try BpeTokenizer.initCustom(&merge_map, &encoder_map, decoder_map);

    // Test Case 1: Multiple merges with repeated patterns
    {
        const input = "Hello hello";
        // Preprocessed: "HelloĠhello"
        // Initial tokens: ["H", "e", "l", "l", "o", "Ġ", "h", "e", "l", "l", "o"]
        // Merges: "H e" -> "He", "l l" -> "ll", "h e" -> "he", "l l" -> "ll"
        // Final tokens: ["He", "ll", "o", "Ġ", "he", "ll", "o"]
        const expected_ids = &[_]usize{ 9, 10, 3, 4, 12, 10, 3 };
        const ids = try tokenizer.encodeAlloc(allocator, input);
        defer allocator.free(ids);
        try std.testing.expectEqualSlices(usize, expected_ids, ids);
    }

    // Test Case 2: Original input with additional merge opportunity
    {
        const input = "Hello world";
        // Preprocessed: "HelloĠworld"
        // Initial tokens: ["H", "e", "l", "l", "o", "Ġ", "w", "o", "r", "l", "d"]
        // Merges: "H e" -> "He", "l l" -> "ll", "o r" -> "or"
        // Final tokens: ["He", "ll", "o", "Ġ", "w", "or", "l", "d"]
        const expected_ids = &[_]usize{ 9, 10, 3, 4, 5, 13, 2, 7 };
        const ids = try tokenizer.encodeAlloc(allocator, input);
        defer allocator.free(ids);
        try std.testing.expectEqualSlices(usize, expected_ids, ids);
    }

    // Test Case 3: Special characters and unknown tokens
    {
        const input = "Hello, Caché!";
        // Preprocessed: "Hello,ĠCaché!"
        // Initial tokens: ["H", "e", "l", "l", "o", ",", "Ġ", "C", "a", "c", "h", "é", "!"]
        // Merges: "H e" -> "He", "l l" -> "ll"
        // Final tokens: ["He", "ll", "o", ",", "Ġ", "C", "a", "c", "h", "é", "!"]
        // Note: "c" and "é" are not in encoder_map, so map to "<unk>"
        const expected_ids = &[_]usize{ 9, 10, 3, 17, 4, 14, 15, 8, 11, 8, 18 };
        const ids = try tokenizer.encodeAlloc(allocator, input);
        defer allocator.free(ids);
        try std.testing.expectEqualSlices(usize, expected_ids, ids);
    }

    // Test Case 4: Empty string
    {
        const input = "";
        // No tokens after preprocessing
        const expected_ids = &[_]usize{};
        const ids = try tokenizer.encodeAlloc(allocator, input);
        defer allocator.free(ids);
        try std.testing.expectEqualSlices(usize, expected_ids, ids);
    }

    // Test Case 5: String with only spaces
    {
        const input = "  ";
        // Preprocessed: "ĠĠ"
        // Final tokens: ["Ġ", "Ġ"] (no merges apply)
        const expected_ids = &[_]usize{ 4, 4 };
        const ids = try tokenizer.encodeAlloc(allocator, input);
        defer allocator.free(ids);
        try std.testing.expectEqualSlices(usize, expected_ids, ids);
    }
}

// Test 4: Decoding
test "Decoding" {
    const allocator = std.testing.allocator;

    // Mock maps
    const merge_map = std.StaticStringMap(usize).initComptime(&.{});

    const encoder_map = std.StaticStringMap(usize).initComptime(&.{
        .{ "H", 0 },
        .{ "e", 1 },
        .{ "l", 2 },
        .{ "o", 3 },
        .{ "Ġ", 4 },
        .{ "w", 5 },
        .{ "r", 6 },
        .{ "d", 7 },
        .{ "<unk>", 8 },
    });

    const decoder_map = &[_][]const u8{ "H", "e", "l", "o", "Ġ", "w", "r", "d", "<unk>" };

    const tokenizer = try BpeTokenizer.initCustom(&merge_map, &encoder_map, decoder_map);

    const ids = &[_]usize{ 0, 1, 2, 2, 3, 4, 5, 3, 6, 2, 7 }; // "H e l l o Ġ w o r l d"
    const expected_text = "Hello world";

    const text = try tokenizer.decodeAlloc(allocator, ids);
    defer allocator.free(text);

    try std.testing.expectEqualStrings(expected_text, text); // Verify decoded text matches expected
}

// Test 5: BPE Application with Merges
test "BPE Application with Merges" {
    const allocator = std.testing.allocator;

    // Define a merge_map with sequential merges
    const merge_map = std.StaticStringMap(usize).initComptime(&.{
        .{ "a b", 1 }, // Merge "a" and "b" first
        .{ "ab c", 2 }, // Then merge "ab" and "c"
    });

    // Dummy maps (not used by applyBpeAlloc directly)
    const encoder_map = std.StaticStringMap(usize).initComptime(&.{
        .{ "<unk>", 0 },
    });

    const decoder_map = &[_][]const u8{};

    const tokenizer = try BpeTokenizer.initCustom(&merge_map, &encoder_map, decoder_map);

    const input = "abc"; // Split into ["a", "b", "c"]
    const expected_tokens = &[_][]const u8{"abc"}; // After "a b" -> "ab", then "ab c" -> "abc"

    const tokens = try tokenizer.applyBpeAlloc(allocator, input);
    defer {
        for (tokens) |token| allocator.free(token);
        allocator.free(tokens);
    }

    try std.testing.expectEqual(expected_tokens.len, tokens.len);
    for (expected_tokens, 0..) |expected, i| {
        try std.testing.expectEqualStrings(expected, tokens[i]);
    }
}

// Test 6: BPE Application with No Merges
test "BPE Application with No Merges" {
    const allocator = std.testing.allocator;

    // Empty merge_map
    const merge_map = std.StaticStringMap(usize).initComptime(&.{});

    // Dummy maps
    const encoder_map = std.StaticStringMap(usize).initComptime(&.{
        .{ "<unk>", 0 },
    });

    const decoder_map = &[_][]const u8{};

    const tokenizer = try BpeTokenizer.initCustom(&merge_map, &encoder_map, decoder_map);

    const input = "abc";
    const expected_tokens = &[_][]const u8{ "a", "b", "c" };

    const tokens = try tokenizer.applyBpeAlloc(allocator, input);
    defer {
        for (tokens) |token| allocator.free(token);
        allocator.free(tokens);
    }

    try std.testing.expectEqual(expected_tokens.len, tokens.len);
    for (expected_tokens, 0..) |expected, i| {
        try std.testing.expectEqualStrings(expected, tokens[i]);
    }
}
