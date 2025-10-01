use crate::errors::MatchError;

mod data;
mod embed;
pub mod errors;

// Exports for external use
pub use data::DisplayImage;
pub use data::DisplayMeta;
pub use data::Preferences;
pub use data::UserProfile;
pub use data::delete_user;
pub use data::get_daily_swipe_statistics;
pub use data::get_rocks_db;

pub fn initialize() -> Result<(), MatchError> {
    data::initialize()?;
    embed::initialize()?;

    Ok(())
}
