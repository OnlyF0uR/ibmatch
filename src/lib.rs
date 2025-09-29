use crate::errors::MatchError;

mod data;
mod embed;
pub mod errors;

// Exports for external use
pub use data::user::UserProfile;
// pub use embed::calculate_embeddings;
// pub use embed::combine_embeddings;

pub fn initialize() -> Result<(), MatchError> {
    data::initialize()?;
    embed::initialize()?;

    Ok(())
}
