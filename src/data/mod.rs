use crate::errors::MatchError;

mod db;
mod hnsw;
mod user;

pub use db::get_rocks_db;
pub use user::DisplayImage;
pub use user::DisplayMeta;
pub use user::Preferences;
pub use user::UserProfile;
pub use user::delete_user;
pub use user::get_daily_swipe_statistics;

pub fn initialize() -> Result<(), MatchError> {
    let db = get_rocks_db();
    let hnsw = hnsw::get_hnsw_index();

    // Bulk load all user profiles from the db into hnsw
    let _ = user::bulk_load(&db, &hnsw)?;
    Ok(())
}
