const std = @import("std");
const tomo = @import("tomo");
const BF16 = tomo.BF16;
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
const addEx = tomorin.function.addEx;
const softmaxEx = tomorin.function.softmaxEx;
const broadcastToEx = tomorin.function.broadcastToEx;
const softmaxCrossEntropyEx = tomorin.function.softmaxCrossEntropyEx;
const getItemEx = tomorin.function.getItemEx;
const LayerDecorator = tomorin.layer.LayerDecorator;
const LayerFieldsFactory = tomorin.layer.LayerFieldsFactory;
const Gelu = tomorin.layer.Gelu;
const LayerNorm = tomorin.layer.LayerNorm;
const CausalSelfAttention = tomorin.layer.CausalSelfAttention;
const Embedding = tomorin.layer.Embedding;

const dbg = tomorin.util.debugPrintGpuTensor;

pub fn MLP(comptime T: type) type {
    return struct {
        pub usingnamespace LayerDecorator(Self);
        fields: LayerFieldsFactory(
            &.{},
            &.{
                .{ "c_fc", Linear(T) },
                .{ "gelu", Gelu(T) },
                .{ "c_proj", Linear(T) },
                .{ "dropout", Dropout(T) },
            },
        ),
        const Self = @This();

        const Config = struct {
            n_embd: usize,
            nobias: bool,
            linear_winit: Linear(T).WInit,
            dropout_ratio: if (T != BF16) T else f32,
        };

        pub fn init(
            config: Config,
            context: *Context,
            chain: *Chain,
        ) !Self {
            return .{
                .fields = .{
                    .c_fc = try .init(
                        config.n_embd,
                        config.n_embd * 4,
                        config.nobias,
                        config.linear_winit,
                        context,
                        chain,
                    ),
                    .gelu = .init,
                    .c_proj = try .init(
                        config.n_embd * 4,
                        config.n_embd,
                        config.nobias,
                        config.linear_winit,
                        context,
                        chain,
                    ),
                    .dropout = .init(config.dropout_ratio),
                },
            };
        }

        pub fn forward(self: *Self, x: *TaggedVar, train: bool, chain: *Chain) !*TaggedVar {
            var y = try self.fields.c_fc.forward(x, chain);
            y = try self.fields.gelu.forward(y, chain);
            y = try self.fields.c_proj.forward(y, chain);
            y = try self.fields.dropout.forward(y, train, chain);
            return y;
        }
    };
}

pub fn Block(comptime T: type) type {
    return struct {
        pub usingnamespace LayerDecorator(Self);
        fields: LayerFieldsFactory(
            &.{},
            &.{
                .{ "ln_1", LayerNorm(T) },
                .{ "attn", CausalSelfAttention(T) },
                .{ "ln_2", LayerNorm(T) },
                .{ "mlp", MLP(T) },
            },
        ),
        const Self = @This();

        const Config = struct {
            n_embd: usize,
            n_head: usize,
            block_size: usize,
            nobias: bool,
            dropout_ratio: if (T != tomo.BF16) T else f32,
            linear_winit: Linear(T).WInit,
            embedding_winit: Embedding(T).WInit,
        };

        pub fn init(
            config: Config,
            context: *Context,
            chain: *Chain,
        ) !Self {
            return .{
                .fields = .{
                    .ln_1 = .init(context, chain),
                    .attn = try .init(
                        .{
                            .n_embd = config.n_embd,
                            .n_head = config.n_head,
                            .block_size = config.block_size,
                            .nobias = config.nobias,
                            .dropout_ratio = config.dropout_ratio,
                        },
                        context,
                        chain,
                    ),
                    .ln_2 = .init(context, chain),
                    .mlp = try .init(
                        .{
                            .n_embd = config.n_embd,
                            .nobias = config.nobias,
                            .linear_winit = config.linear_winit,
                            .dropout_ratio = config.dropout_ratio,
                        },
                        context,
                        chain,
                    ),
                },
            };
        }

        pub fn forward(self: *Self, x: *TaggedVar, train: bool, chain: *Chain) !*TaggedVar {
            var y = try addEx(T, x, try self.fields.attn.forward(
                try self.fields.ln_1.forward(
                    x,
                    1e-5,
                    chain,
                ),
                train,
                chain,
            ), chain);
            y = try addEx(T, y, try self.fields.mlp.forward(
                try self.fields.ln_2.forward(
                    y,
                    1e-5,
                    chain,
                ),
                train,
                chain,
            ), chain);

            return y;
        }
    };
}

pub fn Transformer(comptime T: type, n_layer: comptime_int) type {
    return struct {
        pub usingnamespace LayerDecorator(Self);
        fields: LayerFieldsFactory(
            &.{},
            &(.{
                .{ "wte", Embedding(T) },
                .{ "wpe", Embedding(T) },
                .{ "drop", Dropout(T) },
            } ++ makeBlock() ++
                .{
                    .{ "ln_f", LayerNorm(T) },
                }),
        ),
        config: Config,
        context: *Context,
        const Self = @This();

        pub const Config = struct {
            n_embd: usize,
            n_head: usize,
            block_size: usize,
            nobias: bool,
            dropout_ratio: if (T != tomo.BF16) T else f32,
            linear_winit: Linear(T).WInit,
            embedding_winit: Embedding(T).WInit,
            vocab_size: usize,
        };

        fn makeBlock() [n_layer]std.meta.Tuple(&.{ [:0]const u8, type }) {
            var fields: [n_layer]std.meta.Tuple(&.{ [:0]const u8, type }) = undefined;
            for (&fields, 0..) |*field, i| {
                field.* = .{ std.fmt.comptimePrint("h{}", .{i}), Block(T) };
            }
            return fields;
        }

        pub fn init(
            config: Config,
            context: *Context,
            chain: *Chain,
        ) !Self {
            var self: Self = undefined;
            inline for (0..n_layer) |i| {
                @field(self.fields, std.fmt.comptimePrint("h{}", .{i})) = try .init(
                    .{
                        .n_embd = config.n_embd,
                        .n_head = config.n_head,
                        .block_size = config.block_size,
                        .nobias = config.nobias,
                        .dropout_ratio = config.dropout_ratio,
                        .linear_winit = config.linear_winit,
                        .embedding_winit = config.embedding_winit,
                    },
                    context,
                    chain,
                );
            }

            self.fields.wte = try .init(
                config.vocab_size,
                config.n_embd,
                config.embedding_winit,
                context,
                chain,
            );

            self.fields.wpe = try .init(
                config.block_size,
                config.n_embd,
                config.embedding_winit,
                context,
                chain,
            );

            self.fields.drop = .init(config.dropout_ratio);
            self.fields.ln_f = .init(context, chain);
            self.context = context;
            self.config = config;

            return self;
        }

        pub fn forward(self: *Self, indices: *TaggedVar, train: bool, chain: *Chain) !*TaggedVar {
            // const b = indices.getShape()[0];
            const t = indices.getShape()[1];

            var pos_data: GPUTensor(usize) = try .initAsync(&.{ 1, t, 1 }, self.context.stream);
            errdefer pos_data.deinitAsync(self.context.stream);
            try pos_data.arange(0, 1, self.context.stream);

            var pos = try chain.createVariable(usize, pos_data.move(), null);
            pos = try broadcastToEx(usize, pos, &.{ 1, t, self.config.n_embd }, chain);

            const tok_emb = try self.fields.wte.forward(indices, chain);
            var pos_emb = try self.fields.wpe.forward(pos, chain);
            pos_emb = try broadcastToEx(T, pos_emb, tok_emb.getShape(), chain);

            var x = try self.fields.drop.forward(try addEx(T, tok_emb, pos_emb, chain), train, chain);

            inline for (0..n_layer) |i| {
                x = try @field(self.fields, std.fmt.comptimePrint("h{}", .{i})).forward(x, train, chain);

                //  try tomorin.util.debugPrintGpuTensor(T, &x.asUntagged(T).data, x.getContext());
            }
            x = try self.fields.ln_f.forward(x, 1e-5, chain);

            return x;
        }
    };
}

pub fn GPT(comptime T: type, comptime n_layer: comptime_int) type {
    return struct {
        pub usingnamespace LayerDecorator(Self);
        fields: LayerFieldsFactory(
            &.{},
            &.{
                .{ "transformer", Transformer(T, n_layer) },
                .{ "lm_head", Linear(T) },
            },
        ),
        const Self = @This();

        const Config = struct {
            block_size: usize = 1024,
            //vocab_size: usize = 50257,
            vocab_size: usize = 50304,
            n_head: usize = 12,
            n_embd: usize = 768,
            dropout_ratio: if (T != tomo.BF16) T else f32 = 0.0,
            nobias: bool = false,
            linear_winit: Linear(T).WInit = .he_normal,
            embedding_winit: Embedding(T).WInit = .he_normal,

            pub const default: Config = .{};
            pub const dbg: Config = .{
                .n_embd = 256,
                .n_head = 4,
            };
        };

        pub fn init(
            config: Config,
            context: *Context,
            chain: *Chain,
        ) !Self {
            return .{
                .fields = .{
                    .transformer = try .init(.{
                        .n_embd = config.n_embd,
                        .n_head = config.n_head,
                        .block_size = config.block_size,
                        .nobias = config.nobias,
                        .dropout_ratio = config.dropout_ratio,
                        .linear_winit = config.linear_winit,
                        .embedding_winit = config.embedding_winit,
                        .vocab_size = config.vocab_size,
                    }, context, chain),
                    .lm_head = try .init(
                        config.n_embd,
                        config.vocab_size,
                        config.nobias,
                        config.linear_winit,
                        context,
                        chain,
                    ),
                },
            };
        }

        pub fn forward(self: *Self, indices: *TaggedVar, targets: ?*TaggedVar, chain: *Chain) !std.meta.Tuple(&.{ *TaggedVar, ?*TaggedVar }) {
            const x = try self.fields.transformer.forward(indices, targets != null, chain);

            if (targets) |targ| {
                const logits = try self.fields.lm_head.forward(x, chain);
                const batch = logits.getShape()[0];
                const seq_len = logits.getShape()[1];
                const vocab_size = logits.getShape()[2];
                const logits_reshaped = try reshapeEx(T, logits, &.{ batch * seq_len, vocab_size }, chain);
                const target_batch = targ.getShape()[0];
                const target_seq_len = targ.getShape()[1];
                const target = try reshapeEx(usize, targ, &.{target_batch * target_seq_len}, chain);
                const loss = try softmaxCrossEntropyEx(T, logits_reshaped, .{
                    .t = target,
                    .ignore_index = std.math.maxInt(usize),
                }, chain);

                return .{ logits, loss };
            } else {
                const x_last = try getItemEx(T, 3, x, .{ .all, .{
                    .start = @intCast(x.getShape()[1] - 1),
                    .stop = @intCast(x.getShape()[1]),
                    .step = 1,
                }, .all }, chain);
                const logits = try self.fields.lm_head.forward(x_last, chain);
                return .{ logits, null };
            }
        }
    };
}
