pub const NetworkLayer = struct {
    weights: []f32,
    bias: f32,

    pub fn size(self: NetworkLayer) usize {
        return self.weights.len;
    }

    /// w * x + b
    pub fn multiply_add(self: NetworkLayer, x: []f32) f32 {
        var result: f32 = 0;
        for (self.weights) |weight, i| {
            const other_val = x[i];
            result += weight * other_val;
        }
        return result + self.bias;
    }
};
