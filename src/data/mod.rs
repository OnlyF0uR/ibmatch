use crate::errors::MatchError;

mod db;
mod hnsw;

pub mod user;

pub fn initialize() -> Result<(), MatchError> {
    let db = db::get_db();
    let hnsw = hnsw::get_hnsw_index();

    // Bulk load all user profiles from the db into hnsw
    let _ = user::bulk_load(&db, &hnsw)?;
    Ok(())
}
