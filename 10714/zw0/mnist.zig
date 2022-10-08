const std = @import("std");
const Matrix = @import("matrix.zig").Matrix;

const INPUT_FILE_MAX_SIZE = 50_000_000;

fn readGzipFile(filename: []const u8, alloc: std.mem.Allocator) ![]const u8 {
    std.log.info("Reading: {s} \n", .{filename});
    var file = try std.fs.cwd().openFile(filename, .{});
    defer file.close();
    var in_stream = std.io.bufferedReader(file.reader());
    // Read and decompress the whole file
    var gzip_stream = try std.compress.gzip.gzipStream(alloc, in_stream.reader());
    defer gzip_stream.deinit();
    return try gzip_stream.reader().readAllAlloc(alloc, INPUT_FILE_MAX_SIZE);
}

pub const Mnist = struct {
    const img_width = 28;
    const img_height = 28;
    const img_size = img_width * img_height;

    const DataSet = enum {
        train,
        t10k,
    };

    pixels_filename: []const u8,
    labels_filename: []const u8,

    allocator: std.mem.Allocator,
    pixels_buf: ?[]Matrix.ValType,
    labels_buf: ?[]const u8,

    pub fn deinit(self: *Mnist) void {
        if (self.pixels_buf != null) {
            self.allocator.free(self.pixels_buf.?);
            self.pixels_buf = null;
        }
        if (self.labels_buf != null) {
            self.allocator.free(self.labels_buf.?);
            self.labels_buf = null;
        }
    }

    pub fn init(self: *Mnist, data_set: DataSet, alloc: std.mem.Allocator) void {
        self.allocator = alloc;
        self.pixels_buf = null;
        self.labels_buf = null;
        switch (data_set) {
            .train => {
                self.pixels_filename = "../data/train-images-idx3-ubyte.gz";
                self.labels_filename = "../data/train-labels-idx1-ubyte.gz";
            },

            .t10k => {
                self.pixels_filename = "../hw0/data/t10k-images-idx3-ubyte.gz";
                self.labels_filename = "../hw0/data/t10k-labels-idx1-ubyte.gz";
            },
        }
    }

    pub fn pixels(self: *Mnist) ![]Matrix.ValType {
        if (self.pixels_buf == null) {
            var buf = try readGzipFile(self.pixels_filename, self.allocator);
            defer self.allocator.free(buf);
            const slice = buf[16..]; // bỏ qua 16 bytes đấu
            var pixels_buf = try self.allocator.alloc(Matrix.ValType, slice.len);
            for (pixels_buf[0..]) |*val_ptr, idx| {
                val_ptr.* = @intToFloat(Matrix.ValType, slice[idx]) / @as(Matrix.ValType, 255);
            }
            self.pixels_buf = pixels_buf;
        }
        return self.pixels_buf.?;
    }

    pub fn labels(self: *Mnist) ![]const u8 {
        if (self.labels_buf == null)
            self.labels_buf = try readGzipFile(self.labels_filename, self.allocator);
        return self.labels_buf.?[8..]; // bỏ qua 8 bytes đầu
    }
};

test "mnist" {
    var t10k: Mnist = undefined;
    defer t10k.deinit();
    t10k.init(.t10k, std.testing.allocator);

    const pixels = try t10k.pixels();
    std.debug.print("\n{s}\n", .{"pixels:"});
    for (pixels[0 .. 28 * 28]) |c, idx| {
        std.debug.print("{d:>1.0} ", .{c});
        if (idx % 28 == 27) std.debug.print("\n", .{});
    }

    const labels = try t10k.labels();
    std.debug.print("\n{s}", .{"labels:"});
    for (labels[0..10]) |c| {
        std.debug.print("{d} ", .{c});
    }
    std.debug.print("\n", .{});

    std.debug.print("\n{s}\n", .{"pixels:"});
    for (pixels[28 * 28 .. 28 * 28 + 28 * 28]) |c, idx| {
        std.debug.print("{d:>1.0} ", .{c});
        if (idx % 28 == 27) std.debug.print("\n", .{});
    }
}
