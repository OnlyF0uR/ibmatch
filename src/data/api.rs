use std::sync::Arc;

use rocksdb::DB;

use crate::errors::MatchError;

/// Mark a user as seen by another user
/// This ensures that will not be shown again in searches
pub fn mark_seen(db: &Arc<DB>, user_id: u32, seen_user_id: u32) -> Result<(), MatchError> {
    let key = format!("seen:{}:{}", user_id, seen_user_id);
    db.put(key.as_bytes(), b"1")?;

    Ok(())
}
