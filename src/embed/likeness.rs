pub const LIKENESS_EMB_DIM: usize = 128;

/// Convert relative preference likeness scores (0.0 to 1.0)
/// into an embedding vector
pub fn likeness_to_vector(general_likeness: f32) -> [f32; LIKENESS_EMB_DIM] {
    [general_likeness; LIKENESS_EMB_DIM]
}
