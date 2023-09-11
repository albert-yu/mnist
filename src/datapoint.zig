/// DataPoint struct of arrays
pub const DataPointSOA = struct {
    x: []f64,
    x_chunk_size: usize,
    y: []f64,
    y_chunk_size: usize,
};
