const std = @import("std");
const StaticStringMap = std.StaticStringMap;
const encoder_data = @import("encoder_data.zig").encoder_data;

pub const encoder_map: StaticStringMap(usize) = .initComptime(encoder_data);
