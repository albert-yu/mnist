/// DataPoint struct of arrays
pub const DataPointSOA = struct {
    x: []f64,
    x_chunk_size: usize,
    y: []f64,
    y_chunk_size: usize,

    const Self = @This();

    pub fn len(self: Self) usize {
        return self.x.len / self.x_chunk_size;
    }

    pub fn x_at(self: Self, i: usize) []f64 {
        return self.x[i .. i + self.x_chunk_size];
    }

    pub fn y_at(self: Self, i: usize) []f64 {
        return self.y[i .. i + self.y_chunk_size];
    }
};
