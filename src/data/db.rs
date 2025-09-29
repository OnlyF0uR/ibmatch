use once_cell::sync::Lazy;
use rocksdb::{DB, Options};
use std::sync::Arc;

// Configure RocksDB options
fn rocksdb_options() -> Options {
    let mut opts = Options::default();
    opts.create_if_missing(true);
    opts
}

// Global RocksDB instance wrapped in Arc
static DB_INSTANCE: Lazy<Arc<DB>> = Lazy::new(|| {
    let db = DB::open(&rocksdb_options(), "users_db").expect("Failed to open RocksDB");
    Arc::new(db)
});

// Helper function to access the DB
pub fn get_db() -> Arc<DB> {
    DB_INSTANCE.clone()
}

// Write a test that instantiates the db in a temprorary directory
#[cfg(test)]
mod tests {
    use super::*;
    use tempdir::TempDir;

    #[test]
    #[ignore]
    fn test_rocksdb_open() {
        let temp_dir = TempDir::new("test_db").expect("Failed to create temp dir");
        let db_path = temp_dir.path().to_str().unwrap();

        let opts = rocksdb_options();
        let db = DB::open(&opts, db_path).expect("Failed to open RocksDB in temp dir");

        // Test putting and getting a value
        db.put(b"key1", b"value1").expect("Failed to put value");
        let value = db.get(b"key1").expect("Failed to get value").unwrap();
        assert_eq!(value.as_slice(), b"value1");
    }
}
