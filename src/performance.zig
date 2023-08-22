const std = @import("std");

pub const Stopwatch = struct {
    last_ts: i128,

    pub fn start(self: *Stopwatch) void {
        self.last_ts = std.time.nanoTimestamp();
    }

    pub fn report(self: *Stopwatch, label: []const u8) void {
        const elapsed = std.time.nanoTimestamp() - self.last_ts;
        std.debug.print("elapsed: {}ns ({s})\n", .{ elapsed, label });
        self.last_ts = std.time.nanoTimestamp();
    }
};
