/// Computes inner (dot) product.
///
/// Input vectors assumed to be of equal length.
pub fn inner_product(vec1: []f32, vec2: []f32) f32 {
    var result: f32 = 0;
    for (vec1) |val, i| {
        const other_val = vec2[i];
        result += val * other_val;
    }
    return result;
}

pub fn sum(vec1: []f32, vec2: []f32, out: []f32) void {
    for (vec1) |val, i| {
        const other_val = vec2[i];
        out[i] = val + other_val;
    }
}

pub fn accumulate(acc: []f32, addend: []f32) void {
    sum(acc, addend, acc);
}

pub const Matrix = struct {
    data: []f32,
    rows: usize,
    cols: usize,

    /// Number of elements in this matrix
    pub fn size(self: Matrix) usize {
        return self.data.len;
    }

    /// Applies the matrix as a linear transformation
    /// to the vector (left multiplication),
    /// assuming correct dimensions.
    /// Writes the result to out
    pub fn apply(self: Matrix, vec: []f32, out: []f32) void {
        var i: usize = 0;
        while (i < self.rows) : (i += 1) {
            var acc: f32 = 0;
            var j: usize = 0;
            while (j < self.cols) : (j += 1) {
                var value = self.at(i, j);
                var vec_value = vec[j];
                acc += value * vec_value;
            }
            out[i] = acc;
        }
    }

    /// Returns the value at the given indices.
    ///
    /// Parameters:
    ///   i - 0-based row index
    ///   j - 0-based column index
    pub fn at(self: Matrix, i: usize, j: usize) f32 {
        var index = i * self.cols + j;
        return self.data[index];
    }
};
