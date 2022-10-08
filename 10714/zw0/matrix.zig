const std = @import("std");
const RndGen = std.rand.DefaultPrng;

pub const Matrix = struct {
    pub const ValType = f64;
    const padding: usize = 32; // = 512 / 16 bit, for vectorization

    pub const Combo = enum {
        no_combo,
        transpose,
        derivate,
    };

    rows: usize,
    cols: usize,
    size: usize,
    vals: []ValType,
    shared_mem: bool,
    allocator: std.mem.Allocator,

    const Self = @This();

    pub fn deinit(self: *Self) void {
        if (self.shared_mem) return;

        if (self.size > 0) {
            self.allocator.free(self.vals);
            self.size = 0;
        }
    }
    pub fn init(self: *Self, rows: usize, cols: usize, alloc: std.mem.Allocator) !void {
        self.setup(rows, cols, alloc);
        self.vals = try self.allocator.alloc(ValType, self.size + padding);
        std.mem.set(ValType, self.vals[0..], 0);
    }
    pub fn setup(self: *Self, rows: usize, cols: usize, alloc: std.mem.Allocator) void {
        std.debug.assert(rows > 0);
        std.debug.assert(cols > 0);
        //
        self.rows = rows;
        self.cols = cols;
        self.size = rows * cols;
        self.allocator = alloc;
        self.shared_mem = false;
    }
    pub fn dup(self: Self) !Self {
        var clone: Self = undefined;
        clone.setup(self.rows, self.cols, self.allocator);
        clone.vals = try clone.allocator.alloc(ValType, clone.size + padding);
        clone.setVals(self.getVals());
        return clone;
    }
    pub fn rnd(self: *Self) void {
        var r = RndGen.init(0);
        for (self.getVals()) |*val| {
            const x = r.random().float(f32) / @sqrt(@intToFloat(ValType, self.rows));
            val.* = @floatCast(ValType, x);
        }
    }

    pub inline fn toIdx(self: Self, row: usize, col: usize) usize {
        std.debug.assert(row < self.rows);
        std.debug.assert(col < self.cols);
        const idx = row * self.cols + col;
        std.debug.assert(idx < self.size);
        return idx;
    }
    pub inline fn idx2row(self: Self, idx: usize) usize {
        return idx / self.cols;
    }
    pub inline fn idx2col(self: Self, idx: usize) usize {
        return idx % self.cols;
    }
    pub inline fn set(self: Self, row: usize, col: usize, val: ValType) void {
        self.vals[self.toIdx(row, col)] = val;
    }
    pub inline fn get(self: Self, row: usize, col: usize) ValType {
        return self.vals[self.toIdx(row, col)];
    }

    pub inline fn getVals(self: Self) []ValType {
        return self.vals[0..self.size];
    }
    pub fn setVals(self: *Self, vals: []const ValType) void {
        std.debug.assert(vals.len == self.size);
        std.mem.copy(ValType, self.getVals(), vals);
    }

    pub inline fn getRows(self: Self, begin_row: usize, end_row: usize) []ValType {
        return self.vals[begin_row * self.cols .. end_row * self.cols];
    }

    pub inline fn getRow(self: Self, row: usize) []const ValType {
        return self.getRows(row, row + 1);
    }

    pub fn extractRows(self: Matrix, begin: usize, end: usize) Matrix {
        var m: Matrix = undefined;
        m.setup(end - begin, self.cols, self.allocator);
        m.shared_mem = true;
        m.vals = self.getRows(begin, end);
        return m;
    }

    const ValVecType = std.meta.Vector(padding, ValType);
    const ZEROS_VEC: ValVecType = @splat(padding, @as(ValType, 0));
    const ONES_VEC: ValVecType = @splat(padding, @as(ValType, 1));
    //
    pub fn relu(self: *Self, comptime combo: Combo) void {
        std.debug.assert(combo == .no_combo or combo == .derivate);

        const steps: usize = (self.size / padding) + 1;
        var step: usize = 0;
        var finals: ValVecType = undefined;

        while (step < steps) : (step += 1) {
            //
            const offset = step * padding;
            const vals_vec: ValVecType = self.vals[offset..][0..padding].*;
            const positives = vals_vec > ZEROS_VEC;
            //
            finals = if (combo == .derivate)
                @select(ValType, positives, ONES_VEC, ZEROS_VEC)
            else
                @select(ValType, positives, vals_vec, ZEROS_VEC);
            //
            comptime var idx: usize = 0;
            inline while (idx < padding) : (idx += 1) {
                self.vals[offset + idx] = finals[idx];
            }
        }
    }

    pub fn dotmul(self: Self, b: Self) void {
        std.debug.assert(self.cols == b.cols);
        std.debug.assert(self.rows == b.rows);
        std.debug.assert(self.size == b.size);

        const steps: usize = (self.size / padding) + 1;
        var step: usize = 0;
        var finals: ValVecType = undefined;
        //
        while (step < steps) : (step += 1) {
            //
            const offset = step * padding;
            const a_vals_vec: ValVecType = self.vals[offset..][0..padding].*;
            const b_vals_vec: ValVecType = b.vals[offset..][0..padding].*;
            finals = a_vals_vec * b_vals_vec;
            //
            comptime var idx: usize = 0;
            inline while (idx < padding) : (idx += 1) {
                self.vals[offset + idx] = finals[idx];
            }
        }
    }

    pub fn matmul(self: Self, b: Self, comptime vec_size: usize, comptime combo: Combo) !Self {
        // điều kiện để nhân ma trận là hàng = cột
        std.debug.assert(self.cols == vec_size);
        std.debug.assert(combo == .no_combo or combo == .transpose);

        var bt: Matrix = undefined;
        const do_transpose = (self.cols == b.rows);
        if (do_transpose) {
            bt = try b.transpose();
        } else if (self.cols == b.cols) {
            bt = b;
        } else unreachable;

        var result: Self = undefined;
        const VecType = std.meta.Vector(vec_size, ValType);

        if (combo == .transpose) {
            try result.init(bt.rows, self.rows, self.allocator);

            var row: usize = 0;
            while (row < self.rows) : (row += 1) {
                const v1: VecType = self.getRow(row)[0..vec_size].*;

                var col: usize = 0;
                while (col < bt.rows) : (col += 1) {
                    const v2: VecType = bt.getRow(col)[0..vec_size].*;
                    result.set(col, row, @reduce(.Add, v1 * v2));
                }
            }
        } else {
            try result.init(self.rows, bt.rows, self.allocator);

            var row: usize = 0;
            while (row < self.rows) : (row += 1) {
                const v1: VecType = self.getRow(row)[0..vec_size].*;

                var col: usize = 0;
                while (col < bt.rows) : (col += 1) {
                    const v2: VecType = bt.getRow(col)[0..vec_size].*;
                    result.set(row, col, @reduce(.Add, v1 * v2));
                }
            }
        }
        //
        if (do_transpose) {
            bt.deinit();
        }
        return result;
    }

    pub fn transpose(self: Self) !Self {
        var t: Self = undefined;
        try t.init(self.cols, self.rows, self.allocator);
        for (self.getVals()) |val, idx| {
            const row = self.idx2col(idx);
            const col = self.idx2row(idx);
            t.set(row, col, val);
        }
        return t;
    }

    pub fn eql(self: Self, b: Self) bool {
        if (self.cols != b.cols or self.rows != b.rows) return false;
        for (b.getVals()) |val, idx| {
            if (val != self.vals[idx]) return false;
        }
        return true;
    }

    pub fn minus(self: *Self, b: Self, alpha: ValType) void {
        std.debug.assert(self.rows == b.rows);
        std.debug.assert(self.cols == b.cols);
        //
        for (self.getVals()) |*val, idx| {
            val.* -= b.vals[idx] * alpha;
        }
    }

    pub fn print(self: Self) void {
        std.debug.print("\nMatrix {d} x {d}:\n", .{ self.rows, self.cols });
        for (self.getVals()) |c, idx| {
            std.debug.print("{d: >1.2} ", .{c});
            if (idx % self.cols == self.cols - 1) std.debug.print("\n", .{});
        }
    }
};

test "dotmul" {
    var a: Matrix = undefined;
    defer a.deinit();
    try a.init(4, 3, std.testing.allocator);
    const a_vals: []const Matrix.ValType = &.{ -1, -2, -3, 0, 0, 0, 1, 1, 1, 2, 2, 2 };
    a.setVals(a_vals);

    var b: Matrix = undefined;
    defer b.deinit();
    try b.init(4, 3, std.testing.allocator);
    a.dotmul(b);
    for (a.getVals()) |val| {
        try std.testing.expectEqual(val, 0);
    }

    const b_vals: []const Matrix.ValType = &.{ 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 };
    b.setVals(b_vals);
    a.setVals(a_vals);
    a.dotmul(b);
    try std.testing.expect(std.mem.eql(Matrix.ValType, a.getVals(), a_vals));

    const b_vals1: []const Matrix.ValType = &.{ -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 };
    b.setVals(b_vals1);
    a.setVals(a_vals);
    a.dotmul(b);
    for (a.getVals()) |val, idx| {
        try std.testing.expectEqual(val, -a_vals[idx]);
    }

    const b_vals2: []const Matrix.ValType = &.{ 6, 34, 2, 8, 2, 1, 3, 6, 3, 2, 1, 2 };
    b.setVals(b_vals2);
    a.setVals(a_vals);
    a.dotmul(b);
    for (a.getVals()) |val, idx| {
        try std.testing.expectEqual(val, a_vals[idx] * b_vals2[idx]);
    }
}

test "relu" {
    var a: Matrix = undefined;
    defer a.deinit();
    try a.init(4, 3, std.testing.allocator);
    const a_vals: []const Matrix.ValType = &.{ -1, -2, -3, 0, 0, 0, 1, 1, 1, 2, 2, 2 };
    a.setVals(a_vals);

    var b = try a.dup();
    defer b.deinit();

    try std.testing.expectEqual(a.vals.len, 12 + 32); // padding 32 elems
    try std.testing.expect(std.mem.eql(Matrix.ValType, a.getVals(), a_vals));

    const a_relus: []const Matrix.ValType = &.{ 0, 0, 0, 0, 0, 0, 1, 1, 1, 2, 2, 2 };
    a.relu(.no_combo);
    try std.testing.expect(std.mem.eql(Matrix.ValType, a.getVals(), a_relus));
    try std.testing.expect(std.mem.eql(Matrix.ValType, b.getVals(), a_vals));

    const a_relu_devirates: []const Matrix.ValType = &.{ 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1 };
    b.relu(.derivate);
    try std.testing.expect(std.mem.eql(Matrix.ValType, b.getVals(), a_relu_devirates));
}

test "matmul" {
    var a: Matrix = undefined;
    defer a.deinit();
    try a.init(2, 3, std.testing.allocator);

    var b: Matrix = undefined;
    defer b.deinit();
    try b.init(3, 2, std.testing.allocator);

    const a_vals: []const Matrix.ValType = &.{ 1, 2, 3, 4, 5, 6 };
    const b_vals: []const Matrix.ValType = &.{ 1, 0, 1, 0, 1, 0 };
    const mm_vals: []const Matrix.ValType = &.{ 6, 0, 15, 0 };

    a.setVals(a_vals);
    b.setVals(b_vals);

    try std.testing.expect(std.mem.eql(Matrix.ValType, a.getVals(), a_vals));
    try std.testing.expect(std.mem.eql(Matrix.ValType, b.getVals(), b_vals));

    var bt = try b.transpose();
    defer bt.deinit();

    // std.debug.print("\n{any}\n{any}\n", .{ b.getVals(), bt.getVals() });
    try std.testing.expectEqual(b.get(2, 1), bt.get(1, 2));
    try std.testing.expectEqual(b.get(1, 0), bt.get(0, 1));

    const bt_vals: []const Matrix.ValType = &.{ 1, 1, 1, 0, 0, 0 };
    try std.testing.expect(std.mem.eql(Matrix.ValType, bt.getVals(), bt_vals));

    var mm = try a.matmul(b, 3, .no_combo);
    defer mm.deinit();
    try std.testing.expect(std.mem.eql(Matrix.ValType, mm.getVals(), mm_vals));

    var m1 = try a.matmul(bt, 3, .no_combo);
    defer m1.deinit();
    try std.testing.expect(std.mem.eql(Matrix.ValType, m1.getVals(), mm_vals));
}

test "extractRows" {
    var m: Matrix = undefined;
    defer m.deinit();
    try m.init(3, 2, std.testing.allocator);
    const m_vals: []const Matrix.ValType = &.{ 0, 0, 1, 1, 2, 2 };
    m.setVals(m_vals);

    var x = m.extractRows(1, 3);
    defer x.deinit();

    try std.testing.expect(std.mem.eql(Matrix.ValType, x.getVals(), m_vals[2..]));
    try std.testing.expectEqual(x.cols, 2);
    try std.testing.expectEqual(x.rows, 2);
    try std.testing.expectEqual(x.get(0, 0), 1);
    try std.testing.expectEqual(x.get(0, 1), 1);
    try std.testing.expectEqual(x.get(1, 0), 2);
    try std.testing.expectEqual(x.get(1, 1), 2);
}

test "matrix" {
    var m: Matrix = undefined;
    defer m.deinit();
    try m.init(2, 3, std.testing.allocator);
    try std.testing.expectEqual(m.rows, 2);
    try std.testing.expectEqual(m.cols, 3);

    m.set(0, 0, 0);
    m.set(0, 1, 1);
    m.set(0, 2, 2);
    m.set(1, 0, 3);
    m.set(1, 1, 4);
    m.set(1, 2, 5);

    var row: usize = 0;
    var idx: usize = 0;

    while (row < m.rows) : (row += 1) {
        var col: usize = 0;
        while (col < m.cols) : (col += 1) {
            const val = @intToFloat(Matrix.ValType, idx);
            // std.debug.print("\n{d} {d}, {d} {d}", .{ row, col, idx, val });
            try std.testing.expectEqual(idx, m.toIdx(row, col));
            try std.testing.expectEqual(m.get(row, col), val);
            try std.testing.expectEqual(m.getVals()[idx], val);
            idx += 1;
        }
    }

    try std.testing.expectEqual(idx, m.size);
    // std.debug.print("\n\nmatrix {d} x {d}", .{ m.rows, m.cols });

    for (m.getVals()) |val, i| {
        // std.debug.print("\n{d} {d}, {d} {d}", .{ m.idx2row(i), m.idx2col(i), i, val });
        try std.testing.expectEqual(@intToFloat(Matrix.ValType, i), val);
        try std.testing.expectEqual(m.get(m.idx2row(i), m.idx2col(i)), val);
    }
    // std.debug.print("\n", .{});

    var a = try m.transpose();
    defer a.deinit();

    var b = try a.transpose();
    defer b.deinit();

    var c = try b.transpose();
    defer c.deinit();

    try std.testing.expect(m.eql(m));
    try std.testing.expect(m.eql(b));
    try std.testing.expect(a.eql(c));
}

test "rnd" {
    var a: Matrix = undefined;
    defer a.deinit();
    try a.init(5, 6, std.testing.allocator);
    for (a.getVals()) |val| try std.testing.expectEqual(val, 0);
    a.rnd();
    a.print();
    for (a.getVals()) |val| try std.testing.expect(val != 0);
}
