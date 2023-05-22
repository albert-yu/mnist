/// Computes inner (dot) product.
///
/// Input vectors assumed to be of equal length.
pub fn inner_product(vec1: []f32, vec2: []f32) f32 {
    var result: f32 = 0;
    for (vec1) |weight, i| {
        const other_val = vec2[i];
        result += weight * other_val;
    }
    return result;
}

pub const Matrix = struct {
    data: []f32,
    rows: usize,
    cols: usize,

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
                var index = i * self.cols + j;
                var value = self.data[index];
                var vec_value = vec[j];
                acc += value * vec_value;
            }
            out[i] = acc;
        }
    }
};
