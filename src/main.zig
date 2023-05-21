const std = @import("std");
const nn = @import("network.zig");

pub fn main() !void {
    var init_weights = [_]f32{0} ** 10;
    var network = nn.NetworkLayer{
        .weights = &init_weights,
        .bias = 0.5,
    };
    var size = network.size();
    std.debug.print("size: {}\n", .{size});
}

test "simple test" {
    var list = std.ArrayList(i32).init(std.testing.allocator);
    defer list.deinit(); // try commenting this out and see if zig detects the memory leak!
    try list.append(42);
    try std.testing.expectEqual(@as(i32, 42), list.pop());
}
