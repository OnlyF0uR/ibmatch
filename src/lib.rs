use crate::errors::MatchError;

mod data;
mod embed;
pub mod errors;

// Exports for external use
pub use data::UserProfile;
pub use data::get_rocks_db;

pub fn initialize() -> Result<(), MatchError> {
    data::initialize()?;
    embed::initialize()?;

    Ok(())
}
