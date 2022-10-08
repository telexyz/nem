const std = @import("std");
const Matrix = @import("matrix.zig").Matrix;

pub const Softmax = struct {
    const LEARNING_RATE: Matrix.ValType = 0.2;
    pub const BATCH_SIZE: usize = 100;

    theta: Matrix, // softmax regression
    allocator: std.mem.Allocator,

    w1: Matrix, // lớp feature extraction của mạng nơ-ron 2 lớp
    w2: Matrix, // lớp softmax regression của mạng nơ-ron 2 lớp

    const Self = @This();

    pub fn deinit(self: *Self) void {
        self.theta.deinit();
        self.w1.deinit();
        self.w2.deinit();
    }
    pub fn init(
        self: *Self,
        inp_vec_len: usize,
        fea_vec_len: usize,
        out_vec_len: usize,
        allocator: std.mem.Allocator,
    ) !void {
        self.allocator = allocator;
        // theta ánh xạ trực tiếp input vector thành output vector
        try self.theta.init(out_vec_len, inp_vec_len, self.allocator);

        // w1 ánh xạ input vector thành feature vector
        try self.w1.init(fea_vec_len, inp_vec_len, self.allocator);
        self.w1.rnd();

        // w2 ánh xạ feature vector thành output vector
        try self.w2.init(out_vec_len, fea_vec_len, self.allocator);
        self.w2.rnd();
    }

    pub fn regression_validate(
        self: *Self,
        inputs: Matrix,
        labels: []const u8,
        comptime vec_len: usize,
    ) !void {
        std.debug.assert(inputs.cols == vec_len); // vec_len bằng số pixels trong ảnh (784=28*28)
        std.debug.assert(inputs.rows == labels.len); // mỗi ảnh có 1 nhãn tương ứng

        const VecType = std.meta.Vector(vec_len, Matrix.ValType);
        var correct_labels: usize = 0;

        for (labels) |label, i| {
            const v1: VecType = inputs.getRow(i)[0..vec_len].*;
            var max_belief: Matrix.ValType = -100;
            var max_label: usize = 0;

            var j: usize = 0;
            while (j < self.theta.rows) : (j += 1) {
                const v2: VecType = self.theta.getRow(j)[0..vec_len].*;
                const belief = @exp(@reduce(.Add, v1 * v2));
                if (max_belief < belief) {
                    max_belief = belief;
                    max_label = j;
                }
            }
            if (i % 100 == 99) {
                // Lấy mẫu 1 / 100
                std.debug.print("[ {d}: {d} => {d} ]   ", .{ i, label, max_label });
                // In 4 mẫu thì xuống dòng 1 lần
                if (i % 400 == 399) std.debug.print("\n", .{});
            }
            if (label == max_label) {
                correct_labels += 1;
            }
        } // image iter

        var percentage = @intToFloat(f64, correct_labels) / @intToFloat(f64, labels.len);
        percentage *= 100;
        std.debug.print("\nĐộ chính xác softmax regression: {d:>2.2}%\n", .{percentage});
    }

    pub fn regression(
        self: *Self,
        epoch: usize,
        inputs: Matrix,
        labels: []const u8,
        comptime inp_vec_len: usize,
        comptime out_vec_len: usize,
    ) !void {
        // inp_vec_len bằng số pixels trong ảnh (784=28*28)
        std.debug.assert(inputs.cols == inp_vec_len);
        std.debug.assert(inputs.rows == labels.len); // mỗi ảnh có 1 nhãn tương ứng
        std.debug.assert(self.theta.rows == out_vec_len);

        var batch_begin: usize = 0;
        var prev_percentage: usize = 0;

        while (batch_begin < inputs.rows) : (batch_begin += BATCH_SIZE) {
            const batch_end = batch_begin + BATCH_SIZE;
            if (batch_end > inputs.rows) break;

            const percentage = batch_end / (inputs.rows / 100);
            if (percentage > prev_percentage) {
                prev_percentage = percentage;
                std.log.info("Epoch {d}: {d}%", .{ epoch, percentage });
            }

            const inputs_batch = inputs.extractRows(batch_begin, batch_end);

            var z = try backward(&inputs_batch, &self.theta, //
                labels[batch_begin..batch_end], inp_vec_len, out_vec_len, .transpose);
            defer z.deinit();

            var x = try inputs_batch.transpose();
            defer x.deinit();

            var y = try x.matmul(z, BATCH_SIZE, .transpose);
            defer y.deinit(); // y ở dạng .transpose để khớp với theta

            const a = @intToFloat(Matrix.ValType, (2 * epoch + 1) * BATCH_SIZE);
            self.theta.minus(y, LEARNING_RATE / a);
        } // batch iteration
    }

    pub fn neural_network_validate(
        self: *Self,
        inputs: Matrix,
        labels: []const u8,
        comptime inp_vec_len: usize,
        comptime fea_vec_len: usize,
    ) !void {
        std.debug.assert(inputs.cols == inp_vec_len); // vec_len bằng số pixels trong ảnh (784=28*28)
        std.debug.assert(inputs.rows == labels.len); // mỗi ảnh có 1 nhãn tương ứng
        std.debug.assert(self.w1.cols == inp_vec_len);
        std.debug.assert(self.w1.rows == fea_vec_len);
        std.debug.assert(self.w2.cols == fea_vec_len);

        const InpVecType = std.meta.Vector(inp_vec_len, Matrix.ValType);
        const FeaVecType = std.meta.Vector(fea_vec_len, Matrix.ValType);
        var correct_labels: usize = 0;

        for (labels) |label, i| {
            const v1: InpVecType = inputs.getRow(i)[0..inp_vec_len].*;
            var max_belief: Matrix.ValType = -100;
            var max_label: usize = 0;

            var v3: FeaVecType = undefined;
            var k: usize = 0;
            while (k < self.w1.rows) : (k += 1) {
                const v2: InpVecType = self.w1.getRow(k)[0..inp_vec_len].*;
                const x = @reduce(.Add, v1 * v2);
                v3[k] = if (x < 0) 0 else x; // relu
            }

            var j: usize = 0;
            while (j < self.w2.rows) : (j += 1) {
                const v4: FeaVecType = self.w2.getRow(j)[0..fea_vec_len].*;
                const belief = @exp(@reduce(.Add, v3 * v4));
                if (max_belief < belief) {
                    max_belief = belief;
                    max_label = j;
                }
            }
            if (i % 100 == 99) {
                // Lấy mẫu 1 / 100
                std.debug.print("[ {d}: {d} => {d} ]   ", .{ i, label, max_label });
                // In 4 mẫu thì xuống dòng 1 lần
                if (i % 400 == 399) std.debug.print("\n", .{});
            }
            if (label == max_label) {
                correct_labels += 1;
            }
        } // image iter

        var percentage = @intToFloat(f64, correct_labels) / @intToFloat(f64, labels.len);
        percentage *= 100;
        std.debug.print("\nĐộ chính xác mạng nơ-ron 2 lớp: {d:>2.2}%\n", .{percentage});
    }

    pub fn neural_network_regression(
        self: *Self,
        epoch: usize,
        inputs: Matrix,
        labels: []const u8,
        comptime inp_vec_len: usize,
        comptime fea_vec_len: usize,
        comptime out_vec_len: usize,
    ) !void {
        std.debug.assert(inputs.rows == labels.len); // mỗi ảnh có 1 nhãn tương ứng
        std.debug.assert(inputs.cols == inp_vec_len); // input vec bằng số pixels trong ảnh
        std.debug.assert(self.w1.cols == inp_vec_len);
        std.debug.assert(self.w1.rows == fea_vec_len);
        std.debug.assert(self.w2.cols == fea_vec_len);
        std.debug.assert(self.w2.rows == out_vec_len);

        var batch_begin: usize = 0;
        var prev_percentage: usize = 0;

        while (batch_begin < inputs.rows) : (batch_begin += BATCH_SIZE) {
            const batch_end = batch_begin + BATCH_SIZE;
            if (batch_end > inputs.rows) break;

            const percentage = batch_end / (inputs.rows / 100);

            // Công thức update w1, w2
            // https://raw.githubusercontent.com/telexyz/nem/main/dlsys/files/gradient_wrt_w1_w2.png
            const inputs_batch = inputs.extractRows(batch_begin, batch_end);
            var z1 = try inputs_batch.matmul(self.w1, inp_vec_len, .no_combo);
            defer z1.deinit();
            z1.relu(.no_combo);

            // std.debug.assert(z1.rows == BATCH_SIZE);
            // std.debug.assert(z1.cols == fea_vec_len);

            // if (percentage > prev_percentage) self.w2.extractRows(0, 2).print(); // DEBUG

            var g2 = try backward(&z1, &self.w2, //
                labels[batch_begin..batch_end], fea_vec_len, out_vec_len, .no_combo);
            defer g2.deinit();

            // std.debug.assert(g2.rows == BATCH_SIZE);
            // std.debug.assert(g2.cols == out_vec_len);

            var g2w2t = try g2.matmul(self.w2, out_vec_len, .no_combo);
            defer g2w2t.deinit();

            // std.debug.assert(g2w2t.rows == BATCH_SIZE);
            // std.debug.assert(g2w2t.cols == fea_vec_len);

            var g1 = try z1.dup();
            defer g1.deinit();
            g1.relu(.derivate);
            // if (percentage > prev_percentage) g1.print(); // DEBUG
            g1.dotmul(g2w2t);

            // std.debug.assert(g1.rows == BATCH_SIZE);
            // std.debug.assert(g1.cols == fea_vec_len);

            var xt = try inputs_batch.transpose();
            defer xt.deinit();

            // std.debug.assert(xt.rows == inp_vec_len);
            // std.debug.assert(xt.cols == BATCH_SIZE);

            var xtg1 = try xt.matmul(g1, BATCH_SIZE, .transpose);
            defer xtg1.deinit(); // ở dạng .transpose để khớp với w1

            // std.debug.assert(xtg1.rows == fea_vec_len);
            // std.debug.assert(xtg1.cols == inp_vec_len);

            var z1t = try z1.transpose();
            defer z1t.deinit();

            var z1tg2 = try z1t.matmul(g2, BATCH_SIZE, .transpose);
            defer z1tg2.deinit(); // ở dạng .transpose để khớp với w2

            // std.debug.assert(z1tg2.rows == out_vec_len);
            // std.debug.assert(z1tg2.cols == fea_vec_len);

            const a = @intToFloat(Matrix.ValType, BATCH_SIZE);
            self.w1.minus(xtg1, LEARNING_RATE / a);
            self.w2.minus(z1tg2, LEARNING_RATE / a);

            if (percentage > prev_percentage) {
                prev_percentage = percentage;
                std.log.info("Epoch {d}: {d}%", .{ epoch, percentage });
            }
        } // batch iteration
    }

    pub fn backward(
        inputs: *const Matrix,
        theta: *const Matrix,
        labels: []const u8,
        comptime inp_vec_len: usize,
        comptime out_vec_len: usize,
        comptime combo: Matrix.Combo,
    ) !Matrix {
        //
        std.debug.assert(inputs.rows == labels.len);
        std.debug.assert(inputs.cols == inp_vec_len);

        std.debug.assert(theta.cols == inp_vec_len);
        std.debug.assert(theta.rows == out_vec_len);

        const InpVecType = std.meta.Vector(inp_vec_len, Matrix.ValType);
        const OutVecType = std.meta.Vector(out_vec_len, Matrix.ValType);

        var z: Matrix = undefined;
        if (combo == .transpose) {
            try z.init(theta.rows, inputs.rows, std.heap.page_allocator);
        } else {
            try z.init(inputs.rows, theta.rows, std.heap.page_allocator);
        }

        var row: usize = 0;
        var z_vec: OutVecType = undefined;

        while (row < inputs.rows) : (row += 1) {
            const inp_vec: InpVecType = inputs.getRow(row)[0..inp_vec_len].*;

            var col: usize = 0;
            while (col < out_vec_len) : (col += 1) {
                const param_vec: InpVecType = theta.getRow(col)[0..inp_vec_len].*;
                z_vec[col] = @exp(@reduce(.Add, inp_vec * param_vec));
            }

            const total_exp = @reduce(.Add, z_vec);
            z_vec /= @splat(out_vec_len, total_exp);
            z_vec[labels[row]] -= 1;

            col = 0;
            if (combo == .transpose) {
                while (col < out_vec_len) : (col += 1) z.set(col, row, z_vec[col]);
            } else {
                while (col < out_vec_len) : (col += 1) z.set(row, col, z_vec[col]);
            }
        }
        //
        return z;
    }
};

const Mnist = @import("mnist.zig").Mnist;

pub fn main() !void {
    // Khởi tạo dữ liệu và thông số
    const inp_vec_len: usize = 784; // = 28 * 28
    const fea_vec_len: usize = 128; // giảm số chiều đi còn khoảng 1/6
    const out_vec_len: usize = 10;

    var train: Mnist = undefined;
    defer train.deinit();
    train.init(.train, std.heap.page_allocator);
    // train.init(.t10k, std.heap.page_allocator);
    const pixels = try train.pixels();
    const labels = try train.labels();

    var images: Matrix = undefined;
    images.setup(pixels.len / inp_vec_len, inp_vec_len, std.heap.page_allocator);
    // images.setup(20000, inp_vec_len, std.heap.page_allocator);
    images.vals = pixels;
    images.shared_mem = true;

    var t10k: Mnist = undefined;
    defer t10k.deinit();
    t10k.init(.t10k, std.heap.page_allocator);
    const test_labels = try t10k.labels();
    const test_pixels = try t10k.pixels();

    var test_images: Matrix = undefined;
    test_images.setup(test_pixels.len / inp_vec_len, inp_vec_len, std.heap.page_allocator);
    test_images.vals = test_pixels;
    test_images.shared_mem = true;

    var softmax: Softmax = undefined;
    defer softmax.deinit();
    try softmax.init(inp_vec_len, fea_vec_len, out_vec_len, std.heap.page_allocator);

    // Softmax regression
    comptime var epoch: usize = 0;
    inline while (epoch < 5) : (epoch += 1) {
        try softmax.regression(epoch, images, labels[0..images.rows], //
            inp_vec_len, out_vec_len);
    }

    // Mạng nơ-ron 2 lớp
    epoch = 0;
    inline while (epoch < 10) : (epoch += 1) {
        try softmax.neural_network_regression(epoch, images, labels[0..images.rows], //
            inp_vec_len, fea_vec_len, out_vec_len);
    }

    try softmax.regression_validate(test_images, test_labels, inp_vec_len);

    try softmax.neural_network_validate(test_images, test_labels, inp_vec_len, fea_vec_len);
}
