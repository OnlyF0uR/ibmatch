pub const LIKENESS_EMB_DIM: usize = 128;

/// Convert relative preference likeness scores (0.0 to 1.0)
/// into an embedding vector
pub fn likeness_to_vector(general_likeness: f32) -> [f32; LIKENESS_EMB_DIM] {
    let mut vec = [0f32; LIKENESS_EMB_DIM];
    for i in 0..LIKENESS_EMB_DIM {
        vec[i] = general_likeness;
    }
    vec
}
