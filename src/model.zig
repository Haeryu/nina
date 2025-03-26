const std = @import("std");
const tomo = @import("tomo");
const tomorin = @import("tomorin");
const GPUTensor = tomo.tensor.GPUTensor;
const Linear = tomorin.layer.Linear;
const Dropout = tomorin.layer.Dropout;
const Context = tomorin.context.Context;
const TaggedVar = tomorin.variable.TaggedVar;
const Chain = tomorin.chain.Chain;
const splitEx = tomorin.function.splitEx;
const transposeExEx = tomorin.function.transposeExEx;
const reshapeEx = tomorin.function.reshapeEx;
const matmulEx = tomorin.function.matmulEx;
const scaleEx = tomorin.function.scaleEx;
const maskedFillEx = tomorin.function.maskedFillEx;
const softmaxEx = tomorin.function.softmaxEx;

pub fn Config(comptime T: type) type {
    return struct {
        n_embd: usize,
        n_head: usize,
        block_size: usize,
        nobias: bool,
        dropout_ratio: T,
    };
}

pub fn CausalSelfAttention(comptime T: type) type {
    return struct {
        pub usingnamespace tomorin.layer.LayerDecorator(Self);

        fields: tomorin.layer.LayerFieldsFactory(
            &.{},
            &.{
                .{ "c_attn", Linear(T) },
                .{ "c_proj", Linear(T) },
                .{ "attn_dropout", Dropout(T) },
                .{ "resid_dropout", Dropout(T) },
            },
        ),
        n_head: usize,
        n_embd: usize,
        dropout_ratio: T,
        bias: GPUTensor(T),
        context: *Context,

        const Self = @This();
        pub fn init(
            config: *const Config,
            context: *Context,
            chain: *Chain,
        ) !Self {
            std.debug.assert(config.n_embd % config.n_head == 0);
            var c_attn: Linear(T) = try .init(config.n_embd, 3 * config.n_embd, config.nobias, .xavier, context, chain);
            errdefer c_attn.destroy();
            var c_proj: Linear(T) = try .init(config.n_embd, config.n_embd, config.nobias, .xavier, context, chain);
            errdefer c_proj.destroy();
            const attn_dropout: Dropout(T) = .init(config.dropout_ratio);
            const resid_dropout: Dropout(T) = .init(config.dropout_ratio);

            var bias: GPUTensor(T) = try .initAsync(.{ config.block_size, config.block_size }, context.stream);
            errdefer bias.deinitAsync(context.stream);
            try bias.tril(1.0, context.stream);
            try bias.reshape(.{ 1, 1, config.block_size, config.block_size });

            return .{
                .fields = .{
                    .c_attn = c_attn,
                    .c_proj = c_proj,
                    .attn_dropout = attn_dropout,
                    .resid_dropout = resid_dropout,
                },
                .n_head = config.n_head,
                .n_embd = config.n_embd,
                .bias = bias,
                .context = context,
            };
        }

        pub fn predestroy(self: *Self) void {
            self.bias.deinitAsync(self.context.stream);
        }

        pub fn forward(self: *Self, x: *TaggedVar, train: bool, chain: *Chain) !*TaggedVar {
            const x_shape = x.getShape();
            const b = x_shape[0];
            const t = x_shape[1];
            const c = x_shape[2];

            const attn = try self.fields.c_attn.forward(x, chain);
            var q, var k, var v = try splitEx(3, T, attn, 2, chain);
            q = try reshapeEx(T, q, &.{ b, t, self.n_head, c / self.n_head }, chain);
            k = try reshapeEx(T, k, &.{ b, t, self.n_head, c / self.n_head }, chain);
            v = try reshapeEx(T, v, &.{ b, t, self.n_head, c / self.n_head }, chain);
            q = try transposeExEx(T, q, &.{ 0, 2, 1, 3 }, chain); // (B, nh, T, hs)
            k = try transposeExEx(T, k, &.{ 0, 2, 1, 3 }, chain); // (B, nh, T, hs)
            v = try transposeExEx(T, v, &.{ 0, 2, 1, 3 }, chain); // (B, nh, T, hs)

            const k_transpose = try transposeExEx(T, k, &.{ 0, 1, 3, 2 }, chain);
            var att = try matmulEx(T, q, k_transpose, chain);
            att = try scaleEx(T, att, 1.0 / @sqrt(@as(T, @floatFromInt(k_transpose.getShape()[k_transpose.getShape().len - 1]))));

            var mask = try self.bias.getItem(
                self.context.allocator,
                &.{
                    .all,
                    .all,
                    .{
                        .start = 0,
                        .stop = T,
                        .step = 1,
                    },
                    .{
                        .start = 0,
                        .stop = T,
                        .step = 1,
                    },
                },
                self.context.stream,
            );
            defer mask.deinitAsync(self.context.stream);
            var mask_broad = try mask.broadcastTo(att.getShape(), self.context.stream);
            defer mask_broad.deinitAsync(self.context.stream);

            try mask_broad.scale(-1.0, self.context.stream);
            try mask_broad.shift(1.0, self.context.stream);

            att = try maskedFillEx(
                T,
                att,
                .{
                    .mask = mask_broad.move(),
                    .val = -std.math.inf(T),
                },
                chain,
            );
            att = try softmaxEx(T, att, .{3}, chain);
            att = try self.fields.attn_dropout.forward(att, train, chain);
            var y = try matmulEx(T, att, v, chain);
            y = try transposeExEx(T, y, &.{ 0, 2, 1, 3 }, chain);
            y = try reshapeEx(T, y, &.{ b, t, c }, chain);
            y = try self.fields.c_proj.forward(y, chain);
            y = try self.fields.resid_dropout.forward(y, train, chain);

            return y;
        }
    };
}
