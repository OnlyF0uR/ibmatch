use crate::errors::MatchError;

mod interests;
mod likeness;
mod text;

pub use interests::INTEREST_EMB_DIM;
pub use interests::interests_to_vector;
use likeness::LIKENESS_EMB_DIM;
pub use likeness::likeness_to_vector;
pub use text::TEXT_EMB_DIM;
pub use text::text_to_embedding;

pub fn calculate_persistent_embeddings(
    raw_interests: &[u32],
    raw_biography: &str,
) -> Result<([f32; TEXT_EMB_DIM], [f32; INTEREST_EMB_DIM]), MatchError> {
    let text_embed = text_to_embedding(raw_biography);
    let interest_embed = interests_to_vector(raw_interests);

    Ok((text_embed, interest_embed))
}

const EMBEDDING_WEIGHTS: (f32, f32, f32) = (0.25, 0.35, 0.675);

pub fn combine_embeddings(
    text_embed: &[f32; TEXT_EMB_DIM],
    interest_embed: &[f32; INTEREST_EMB_DIM],
    likeness_embed: &[f32; LIKENESS_EMB_DIM],
) -> Result<Vec<f32>, MatchError> {
    let (w_text, w_interest, w_likeness) = EMBEDDING_WEIGHTS;

    // Concatenate weighted embeddings
    let mut combined = Vec::with_capacity(TEXT_EMB_DIM + INTEREST_EMB_DIM + LIKENESS_EMB_DIM);

    // Add weighted text embeddings
    for &val in text_embed.iter() {
        combined.push(val * w_text);
    }

    // Add weighted interest embeddings
    for &val in interest_embed.iter() {
        combined.push(val * w_interest);
    }

    // Add weighted likeness embeddings
    for &val in likeness_embed.iter() {
        combined.push(val * w_likeness);
    }

    // Normalize the combined vector
    let norm = combined.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-8);
    for v in &mut combined {
        *v /= norm;
    }

    Ok(combined)
}

pub fn initialize() -> Result<(), MatchError> {
    // Initialize text embeddings
    text::initialize_embeddings("data/glove.6B.50d.txt")?;
    Ok(())
}
