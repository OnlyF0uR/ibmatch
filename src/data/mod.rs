use crate::errors::MatchError;

mod db;
mod hnsw;
mod user;

pub use db::get_rocks_db;
pub use user::UserProfile;

pub fn initialize() -> Result<(), MatchError> {
    let db = get_rocks_db();
    let hnsw = hnsw::get_hnsw_index();

    // Bulk load all user profiles from the db into hnsw
    let _ = user::bulk_load(&db, &hnsw)?;
    Ok(())
}
