const std = @import("std");

pub const Stopwatch = struct {
    last_ts: i128,
    elapsed: i128,

    pub fn start(self: *Stopwatch) void {
        self.last_ts = std.time.nanoTimestamp();
    }

    pub fn report(self: *Stopwatch, label: []const u8) void {
        std.debug.print("elapsed: {}ns ({s})\n", .{ self.elapsed, label });
    }

    pub fn stop(self: *Stopwatch) void {
        const val = std.time.nanoTimestamp() - self.last_ts;
        self.elapsed = val;
    }
};
