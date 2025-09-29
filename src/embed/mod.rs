use crate::{
    embed::{interests::interests_to_vector, text::text_to_embedding},
    errors::MatchError,
};

mod interests;
mod text;

pub use interests::INTEREST_EMB_DIM;
pub use text::TEXT_EMB_DIM;

// final_vec = α * text + β * interest + γ * aggregated_image
pub fn calculate_embeddings(
    raw_interests: &[u32],
    raw_biography: &str,
) -> Result<([f32; TEXT_EMB_DIM], [f32; INTEREST_EMB_DIM]), MatchError> {
    let text_embed = text_to_embedding(raw_biography);
    let interest_embed = interests_to_vector(&raw_interests);

    Ok((text_embed, interest_embed))
}

pub fn combine_embeddings(
    text_embed: &[f32],
    interest_embed: &[f32],
    weights: (f32, f32),
) -> Result<Vec<f32>, MatchError> {
    let (w_text, w_interest) = weights;

    let mut combined = vec![0f32; text_embed.len()];
    for i in 0..text_embed.len() {
        combined[i] = text_embed[i] * w_text + interest_embed[i] * w_interest;
    }

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
