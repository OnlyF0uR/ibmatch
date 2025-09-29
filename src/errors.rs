use std::{io, sync::PoisonError};

#[derive(Debug)]
pub enum MatchError {
    UserEncodeError(bincode::error::EncodeError),
    UserDecodeError(bincode::error::DecodeError),
    RocksDBError(rocksdb::Error),
    UserNotFound,
    HnswLockError(String),
    EmbeddingLoadingError(io::Error),
}

impl std::fmt::Display for MatchError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            MatchError::UserEncodeError(e) => write!(f, "User encoding error: {}", e),
            MatchError::UserDecodeError(e) => write!(f, "User decoding error: {}", e),
            MatchError::RocksDBError(e) => write!(f, "RocksDB error: {}", e),
            MatchError::UserNotFound => write!(f, "User not found"),
            MatchError::HnswLockError(e) => write!(f, "HNSW lock error: {}", e),
            MatchError::EmbeddingLoadingError(e) => write!(f, "Embedding loading error: {}", e),
        }
    }
}

impl std::error::Error for MatchError {}

impl From<bincode::error::EncodeError> for MatchError {
    fn from(err: bincode::error::EncodeError) -> Self {
        MatchError::UserEncodeError(err)
    }
}

impl From<bincode::error::DecodeError> for MatchError {
    fn from(err: bincode::error::DecodeError) -> Self {
        MatchError::UserDecodeError(err)
    }
}

impl From<rocksdb::Error> for MatchError {
    fn from(err: rocksdb::Error) -> Self {
        MatchError::RocksDBError(err)
    }
}

impl<T> From<PoisonError<T>> for MatchError {
    fn from(err: PoisonError<T>) -> Self {
        MatchError::HnswLockError(err.to_string())
    }
}

impl From<io::Error> for MatchError {
    fn from(err: io::Error) -> Self {
        MatchError::EmbeddingLoadingError(err)
    }
}
