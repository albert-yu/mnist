pub const NetworkLayer = struct {
    weights: []f32,
    biases: []f32,

    pub fn size(self: NetworkLayer) usize {
        return self.weights.len;
    }
};
