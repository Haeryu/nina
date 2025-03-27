const std = @import("std");

const merges_data = @import("merges_data.zig").merges_data;

pub const merge_entries = blk: {
    var entries: [merges_data.len]struct { []const u8, usize } = undefined;
    @setEvalBranchQuota(99999999);
    for (&merges_data, &entries, 0..) |pair, *entry, i| {
        const joined = pair[0] ++ " " ++ pair[1]; // concat with space
        entry.* = .{ joined, @intCast(i) };
    }
    break :blk entries;
};

pub const merge_map = std.StaticStringMap(usize).initComptime(merge_entries);
