use std::sync::Arc;

use bincode::{
    Decode, Encode,
    config::{self},
};
use hnsw_rs::hnsw::FilterT;
use rocksdb::DB;

use crate::{
    data::hnsw,
    embed::{self, INTEREST_EMB_DIM},
};
use crate::{embed::TEXT_EMB_DIM, errors::MatchError};

// Likeness, rating, distance
const FILTER_LIKENESS_WEIGHT: f32 = 0.5;
const FILTER_RATING_WEIGHT: f32 = 0.3;
const FILTER_DISTANCE_WEIGHT: f32 = 0.2;

/// Filter implementation for HNSW search
struct UserFilter {
    allowed_ids: std::collections::HashSet<usize>,
}

impl FilterT for UserFilter {
    fn hnsw_filter(&self, id: &hnsw_rs::prelude::DataId) -> bool {
        self.allowed_ids.contains(&id)
    }
}

/// Represents a user's preferences for matching
#[derive(Debug, Encode, Decode, Clone)]
pub struct Preferences {
    pub gender: Vec<u8>, // Gender indeces the user is interested in (0=M, 1=F, 2=O)
    pub age_range: [u8; 2], // min and max age
    pub distance_km: u32, // maximum distance
    pub min_height_cm: u16, // maximum height of a person
}

/// Metadata for internal tracking
#[derive(Debug, Encode, Decode)]
pub struct Meta {
    pub last_seen: u64, // Unix timestamp
    pub plan: u8,       // 0=free, 1=premium
    pub banned: bool,
}

/// Full user profile with multiplier for boosting
#[derive(Debug, Encode, Decode)]
pub struct UserProfile {
    pub user_id: u32,
    pub age: u8,
    pub gender: u8,     // Gender index (0=M, 1=F, 2=O)
    pub height_cm: u16, // Height in cm

    pub likeness_score: f32,   // Overall likeness score (0.0 to 1.0)
    pub preference_score: f32, // Overall preference score (0.0 to 1.0)
    pub norm_rating: f32,      // normalized rating score (0.0 to 1.0)

    pub likeness_updates: u32,   // Number of updates to likeness score
    pub preference_updates: u32, // Number of updates to preference score

    pub location: [f64; 2],       // lat/lon
    pub preferences: Preferences, // Matching preference / filters
    pub meta: Meta,
    pub multiplier: f32, // exposure / scoring multiplier, like boosters
    pub multiplier_expiry: Option<u64>, // Unix timestamp when multiplier expires

    // Embeddings
    pub text_embedding: [f32; TEXT_EMB_DIM], // 50-d embedding for bio/interests
    pub interest_embeddings: [f32; INTEREST_EMB_DIM],
}

impl UserProfile {
    /// Function to create a new user (we shall infer embeddings)
    pub fn create_new(
        db: &Arc<DB>,
        user_id: u32,
        age: u8,
        gender: u8,
        height_cm: u16,
        location: [f64; 2],
        preferences: Preferences,
        raw_intersts: &[u32],
        raw_biography: &str,
    ) -> Result<Self, MatchError> {
        // 1. Calculate text embedding
        let (t_embed, i_embed) =
            embed::calculate_persistent_embeddings(raw_intersts, raw_biography)?;

        // Get current unix timestamp
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();

        // 2. Create user profile with default values
        let user = UserProfile {
            user_id,
            age,
            gender,
            height_cm,
            likeness_score: 0.5,   // Neutral initial likeness score
            preference_score: 0.5, // Neutral initial preference score
            norm_rating: 0.5,
            likeness_updates: 0,
            preference_updates: 0,
            location,
            preferences,
            meta: Meta {
                last_seen: now,
                plan: 0,
                banned: false,
            },
            multiplier: 1.0,
            multiplier_expiry: None,
            text_embedding: t_embed,
            interest_embeddings: i_embed,
        };

        // 3. Store the user in the DB and HNSW
        let key = format!("user:{}", user.user_id);
        let value = user.encode()?;

        db.put(key.as_bytes(), &value)?;

        let likeness_embedding = embed::likeness_to_vector(user.likeness_score);
        let combined = embed::combine_embeddings(
            &user.text_embedding,
            &user.interest_embeddings,
            &likeness_embedding,
        )?;

        // Add the user into the HNSW index
        let hnsw = hnsw::get_hnsw_index();
        let hnsw_write = hnsw.write()?;
        hnsw_write.insert((&combined, user.user_id as usize));

        // Add user to gender mapping
        hnsw::add_user_to_gender(user.user_id as usize, user.gender);

        Ok(user)
    }

    /// Load a user
    pub fn load_user(db: &Arc<DB>, user_id: u32) -> Result<Self, MatchError> {
        let key = format!("user:{}", user_id);

        let value = db.get(key.as_bytes())?;
        let mut parsed = match value {
            Some(v) => UserProfile::decode(&v)?,
            None => return Err(MatchError::UserNotFound),
        };

        // Set active time
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();

        parsed.meta.last_seen = now;

        // Also restore in DB
        let value = parsed.encode()?;
        db.put(key.as_bytes(), &value)?;

        // Combine the relevant embeddings
        let likeness_embedding = embed::likeness_to_vector(parsed.likeness_score);
        let combined = embed::combine_embeddings(
            &parsed.text_embedding,
            &parsed.interest_embeddings,
            &likeness_embedding,
        )?;

        // Add to HNSW to maintain accurate representation (will override if already exists)
        let hnsw = hnsw::get_hnsw_index();
        let hnsw_write = hnsw.write()?;
        hnsw_write.insert((&combined, user_id as usize));

        // Add to gender mapping
        hnsw::add_user_to_gender(user_id as usize, parsed.gender);

        Ok(parsed)
    }

    /// Search
    /// Search for potential matches based on the user's preferences and embeddings.
    /// This function performs a nearest neighbor search in the HNSW index,
    /// applies pre-filters, post-filters, and returns the top_k matches with their scores.
    pub fn search(
        &self,
        db: &Arc<DB>,
        top_k: usize,
    ) -> Result<Vec<(f32, UserProfile)>, MatchError> {
        // It is vital that we substitute the likeness embedding with the preference embedding
        // because we search towards likeness in accordance with the preferences of this user
        let preferece_embedding = embed::likeness_to_vector(self.preference_score);

        let combined = embed::combine_embeddings(
            &self.text_embedding,
            &self.interest_embeddings,
            &preferece_embedding,
        )?;

        // Get the allowed gender ids
        // these are the ids of users of which we can prefilter on gender
        let allowed_ids =
            hnsw::get_allowed_ids(db, self.user_id as usize, &self.preferences.gender);

        let hnsw = hnsw::get_hnsw_index();
        let hnsw_read = hnsw.read()?;

        let filter = UserFilter { allowed_ids };
        let search = hnsw_read.search_filter(&combined, top_k * 5, 16, Some(&filter));

        // Here we have the IDs of the candidates
        // let mut users = Vec::new();
        let mut users: Vec<(f32, UserProfile)> = Vec::with_capacity(search.len());

        // Don't flood the results with people that liked us
        let max_liked_by = (top_k as f32 * 0.25).ceil() as usize;
        // These are the user ids that swiped us, without us swiping them
        let liked_by = self.get_potential_matches(db, max_liked_by)?;
        for n in &liked_by {
            let candidate = UserProfile::load_user(&db, *n)?;

            let (passed, score) = self.post_filter(&candidate, 0);
            if passed {
                users.push((score, candidate));
            }
        }

        for n in &search {
            let candidate = UserProfile::load_user(&db, n.d_id as u32)?;

            let (passed, score) = self.post_filter(&candidate, 0);
            if passed {
                users.push((score, candidate));
            }
        }

        // Now if users.len is smaller than top_k we must relax the filters
        if users.len() < top_k {
            for n in &search {
                let candidate = UserProfile::load_user(&db, n.d_id as u32)?;

                let (passed, score) = self.post_filter(&candidate, 1);
                if passed {
                    users.push((score, candidate));
                }
            }
        }

        // If still not enough, relax even more
        if users.len() < top_k {
            for n in &search {
                let candidate = UserProfile::load_user(&db, n.d_id as u32)?;

                let (passed, score) = self.post_filter(&candidate, 2);
                if passed {
                    users.push((score, candidate));
                }
            }
        }

        // Now we must sort on the users score
        users.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
        // Finally truncate to top_k
        users.truncate(top_k);

        Ok(users)
    }

    /// Update user preferences
    /// This updates the user's preferences and persists them to the database.
    pub fn update_preferences(
        &mut self,
        db: &Arc<DB>,
        new_preferences: &Preferences,
    ) -> Result<(), MatchError> {
        self.preferences = new_preferences.clone();
        self.update_last_seen();

        let key = format!("user:{}", self.user_id);
        let value = self.encode()?;
        db.put(key.as_bytes(), &value)?;

        Ok(())
    }

    /// Process a swipe between two users
    /// This function updates both users: the current user's preference score and the target user's likeness score.
    /// It also registers the swipe in the database to exclude the user from showing in future searches.
    pub fn process_swipe(
        &mut self,
        db: &Arc<DB>,
        target_user_id: u32,
        positive: bool,
    ) -> Result<bool, MatchError> {
        // Load the target user to get their likeness score and update it
        let mut target_user = UserProfile::load_user(db, target_user_id)?;

        // Update current user's preference score based on target's likeness
        let impact_factor = 1.0 / (1.0 + self.preference_updates as f32 * 0.1);
        let likeness_factor = if positive {
            target_user.likeness_score // Positive swipe: move towards their likeness
        } else {
            1.0 - target_user.likeness_score // Negative swipe: move away from their likeness
        };

        let target_preference = likeness_factor;
        let change = (target_preference - self.preference_score) * 0.1 * impact_factor;
        self.preference_score = (self.preference_score + change).clamp(0.0, 1.0);
        self.preference_updates += 1;

        // Update target user's likeness score
        let target_impact_factor = 1.0 / (1.0 + target_user.likeness_updates as f32 * 0.1);
        let base_change = if positive { 0.1 } else { -0.05 };
        let likeness_change = base_change * target_impact_factor;
        target_user.likeness_score = (target_user.likeness_score + likeness_change).clamp(0.0, 1.0);
        target_user.likeness_updates += 1;

        // Update timestamp only for the active user (the one swiping)
        self.update_last_seen();

        // Register the outgoing swipe in RocksDB (format: "swipe:<user_id>:<target_user_id>")
        let swipe_key = format!("swipe:{}:{}", self.user_id, target_user_id);
        let swipe_value = if positive { b"1" } else { b"0" };
        db.put(swipe_key.as_bytes(), swipe_value)?;
        // This one is so we can iterate for matches quicker
        let swipe_in_key = format!("swipe-in:{}:{}", target_user_id, self.user_id);
        db.put(swipe_in_key.as_bytes(), swipe_value)?;

        // Save both users to database
        let key = format!("user:{}", self.user_id);
        let value = self.encode()?;
        db.put(key.as_bytes(), &value)?;

        let target_key = format!("user:{}", target_user_id);
        let target_value = target_user.encode()?;
        db.put(target_key.as_bytes(), &target_value)?;

        // Update HNSW index for target user with new likeness embedding
        let target_likeness_embedding = embed::likeness_to_vector(target_user.likeness_score);
        let target_combined = embed::combine_embeddings(
            &target_user.text_embedding,
            &target_user.interest_embeddings,
            &target_likeness_embedding,
        )?;

        let hnsw = hnsw::get_hnsw_index();
        let hnsw_write = hnsw.write()?;
        hnsw_write.insert((&target_combined, target_user_id as usize));

        let is_match = self.is_liked_by(db, target_user_id)?;
        Ok(is_match)
    }

    fn is_liked_by(&self, db: &Arc<DB>, other_user_id: u32) -> Result<bool, MatchError> {
        // Because we have both user ids we can reconstruct the original swipe key
        let swipe_key = format!("swipe:{}:{}", other_user_id, self.user_id);
        let value = db.get(swipe_key.as_bytes())?;
        if let Some(v) = value {
            if let Ok(swipe_value) = std::str::from_utf8(&v) {
                return Ok(swipe_value == "1");
            }
        }
        Ok(false)
    }

    fn get_potential_matches(&self, db: &Arc<DB>, max_n: usize) -> Result<Vec<u32>, MatchError> {
        let mut liked_by = Vec::new();
        let prefix = format!("swipe-in:{}:", self.user_id);
        let iter = db.prefix_iterator(prefix.as_bytes());

        // Ensure that we do not collect more than needed
        for item in iter {
            let (key, value) = item?;
            if let Ok(key_str) = std::str::from_utf8(&key) {
                if key_str.starts_with(&prefix) {
                    if let Some(id_str) = key_str.strip_prefix(&prefix) {
                        if let Ok(user_id) = id_str.parse::<u32>() {
                            // Check if the swipe was positive
                            if let Ok(swipe_value) = std::str::from_utf8(&value) {
                                // Also check we have not swiped them back already
                                let already_swiped = self.is_liked_by(db, user_id)?;
                                if swipe_value == "1" && !already_swiped {
                                    liked_by.push(user_id);
                                    if liked_by.len() >= max_n {
                                        break;
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        Ok(liked_by)
    }

    /// Update last seen timestamp to current time
    /// This updates the `last_seen` field in the user's metadata to the current
    /// Unix timestamp. Note that this function only updates the field in memory;
    /// saving to the database should be handled externally.
    fn update_last_seen(&mut self) {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();

        self.meta.last_seen = now;
    }

    fn post_filter(&self, candidate: &UserProfile, strictness_level: u8) -> (bool, f32) {
        let mut min_age = self.preferences.age_range[0];
        let mut max_age = self.preferences.age_range[1];
        let mut max_distance = self.preferences.distance_km;
        let min_height = self.preferences.min_height_cm;
        let distance = self.distance_in_km(candidate);

        if strictness_level == 0 {
            // Age range
            if candidate.age < min_age || candidate.age > max_age {
                return (false, 0.0);
            }

            if distance > max_distance as f64 {
                return (false, 0.0);
            }
        } else if strictness_level == 1 {
            // Age range
            if candidate.age < min_age || candidate.age > max_age {
                return (false, 0.0);
            }

            // Update max distance to be more lenient
            max_distance = (max_distance as f32 * 1.5) as u32;

            if distance > max_distance as f64 {
                return (false, 0.0);
            }
        } else {
            // Update age range to be more lenient
            // Age range min of 18 max of 99, expand by 5 years on both sides
            min_age = if min_age > 18 { min_age - 5 } else { 18 };
            max_age = if max_age < 99 { max_age + 5 } else { 99 };

            if candidate.age < min_age || candidate.age > max_age {
                return (false, 0.0);
            }

            // Update max distance to be more lenient
            max_distance *= 2;

            if distance > max_distance as f64 {
                return (false, 0.0);
            }
        }

        // Ensure that the candidate is not smaller than min height
        if candidate.height_cm < min_height {
            return (false, 0.0);
        }

        // Now if the user has not been seen in the last 30 days we remove them
        // from hnsw, same goes for if he is banned, and we return false
        if candidate.meta.banned
            || (std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs()
                - candidate.meta.last_seen)
                > 2_592_000
        {
            // IDEA: Remove from HNSW?
            return (false, 0.0);
        }

        let mut multiplier = candidate.multiplier;
        if let Some(expiry) = candidate.multiplier_expiry {
            let now = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs();
            if now >= expiry {
                multiplier = 1.0; // Consider multiplier to be 1.0 if expired
            }
        }

        let likeness = candidate.likeness_score;
        let rating = candidate.norm_rating;
        let closeness = (1.0 / (1.0 + distance)).clamp(0.0, 1.0) as f32;

        let score = (likeness * FILTER_LIKENESS_WEIGHT
            + rating * FILTER_RATING_WEIGHT
            + closeness * FILTER_DISTANCE_WEIGHT)
            * multiplier;

        (true, score)
    }

    fn distance_in_km(&self, other: &UserProfile) -> f64 {
        let lat1 = self.location[0].to_radians();
        let lon1 = self.location[1].to_radians();
        let lat2 = other.location[0].to_radians();
        let lon2 = other.location[1].to_radians();

        let dlat = lat2 - lat1;
        let dlon = lon2 - lon1;

        let a = (dlat / 2.0).sin().powi(2) + lat1.cos() * lat2.cos() * (dlon / 2.0).sin().powi(2);
        let c = 2.0 * a.sqrt().asin();

        // Earth's radius in kilometers
        let earth_radius_km = 6371.0;
        earth_radius_km * c
    }

    /// Encode the UserProfile to a byte vector using bincode
    fn encode(&self) -> Result<Vec<u8>, MatchError> {
        let config = config::standard();
        let encoded: Vec<u8> = bincode::encode_to_vec(self, config)?;

        Ok(encoded)
    }

    /// Decode a UserProfile from a byte slice
    fn decode(bytes: &[u8]) -> Result<Self, MatchError> {
        let config = config::standard();
        let (decoded, _): (Self, _) = bincode::decode_from_slice(bytes, config)?;

        Ok(decoded)
    }

    pub fn normalize_rating(mut rating: f32) -> f32 {
        if rating < 1.0 {
            rating = 1.0;
        } else if rating > 10.0 {
            rating = 10.0;
        }

        (rating as f32 - 1.0) / 9.0
    }
}

pub fn bulk_load(
    db: &Arc<DB>,
    hnsw: &Arc<
        std::sync::RwLock<hnsw_rs::prelude::Hnsw<'static, f32, hnsw_rs::prelude::DistCosine>>,
    >,
) -> Result<usize, MatchError> {
    let iter = db.iterator(rocksdb::IteratorMode::Start);
    let mut loaded_count = 0;

    for item in iter {
        let (key, value) = item?;

        // Check if this is a user key (format: "user:{id}")
        if let Ok(key_str) = std::str::from_utf8(&key) {
            if key_str.starts_with("user:") {
                // Extract user ID from key
                if let Some(id_str) = key_str.strip_prefix("user:") {
                    if let Ok(user_id) = id_str.parse::<usize>() {
                        // Decode user profile
                        let user_profile = UserProfile::decode(&value)?;
                        // Only load users that are not banned and have been online within
                        // the last 30 days (2592000 seconds)
                        if user_profile.meta.banned
                            || (std::time::SystemTime::now()
                                .duration_since(std::time::UNIX_EPOCH)
                                .unwrap_or_default()
                                .as_secs()
                                - user_profile.meta.last_seen)
                                > 2_592_000
                        {
                            continue;
                        }

                        let likeness_embedding =
                            embed::likeness_to_vector(user_profile.likeness_score);
                        let combined = embed::combine_embeddings(
                            &user_profile.text_embedding,
                            &user_profile.interest_embeddings,
                            &likeness_embedding,
                        )?;

                        // Insert text embedding into HNSW index
                        let hnsw_write = hnsw.write()?;
                        hnsw_write.insert((&combined, user_id));

                        // Add user to gender mapping
                        hnsw::add_user_to_gender(user_id, user_profile.gender);

                        loaded_count += 1;
                    }
                }
            }
        }
    }

    Ok(loaded_count)
}

// Test
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_encode_decode() {
        let user = UserProfile {
            user_id: 1,
            age: 24,
            gender: 1,
            height_cm: 190,
            likeness_score: 0.5,
            preference_score: 0.5,
            norm_rating: 0.75,
            likeness_updates: 1,
            preference_updates: 5,
            location: [52.3702, 4.895],
            preferences: Preferences {
                gender: vec![0],
                age_range: [22, 28],
                distance_km: 50,
                min_height_cm: 200,
            },
            meta: Meta {
                last_seen: 1_695_900_000,
                plan: 1,
                banned: false,
            },
            multiplier: 1.2,
            multiplier_expiry: None,
            text_embedding: [0.1; TEXT_EMB_DIM],
            interest_embeddings: [0.2; INTEREST_EMB_DIM],
        };

        // Test encoding
        let encoded = user.encode().expect("Failed to encode user");
        assert!(!encoded.is_empty());

        // Test decoding
        let decoded = UserProfile::decode(&encoded).expect("Failed to decode user");

        // Verify all fields match
        assert_eq!(decoded.user_id, user.user_id);
        assert_eq!(decoded.age, user.age);
        assert_eq!(decoded.gender, user.gender);
        assert_eq!(decoded.height_cm, user.height_cm);
        assert_eq!(decoded.likeness_score, user.likeness_score);
        assert_eq!(decoded.preference_score, user.preference_score);
        assert_eq!(decoded.norm_rating, user.norm_rating);
        assert_eq!(decoded.likeness_updates, user.likeness_updates);
        assert_eq!(decoded.preference_updates, user.preference_updates);
        assert_eq!(decoded.location, user.location);
        assert_eq!(decoded.preferences.gender, user.preferences.gender);
        assert_eq!(decoded.preferences.age_range, user.preferences.age_range);
        assert_eq!(
            decoded.preferences.distance_km,
            user.preferences.distance_km
        );
        assert_eq!(
            decoded.preferences.min_height_cm,
            user.preferences.min_height_cm
        );
        assert_eq!(decoded.meta.last_seen, user.meta.last_seen);
        assert_eq!(decoded.meta.plan, user.meta.plan);
        assert_eq!(decoded.meta.banned, user.meta.banned);
        assert_eq!(decoded.multiplier, user.multiplier);
        assert_eq!(decoded.multiplier_expiry, user.multiplier_expiry);
        assert_eq!(decoded.text_embedding, user.text_embedding);
        assert_eq!(decoded.interest_embeddings, user.interest_embeddings);
    }

    #[test]
    fn test_multiple_gender_preferences() {
        let user = UserProfile {
            user_id: 1,
            age: 30,
            gender: 2,
            height_cm: 175,
            likeness_score: 0.5,
            preference_score: 0.5,
            norm_rating: 0.85,
            likeness_updates: 2,
            preference_updates: 3,
            location: [40.7128, -74.0060],
            preferences: Preferences {
                gender: vec![0, 1, 2],
                age_range: [18, 65],
                distance_km: 100,
                min_height_cm: 210,
            },
            meta: Meta {
                last_seen: 1_700_000_000,
                plan: 0,
                banned: false,
            },
            multiplier: 0.8,
            multiplier_expiry: None,
            text_embedding: [0.4; TEXT_EMB_DIM],
            interest_embeddings: [0.5; INTEREST_EMB_DIM],
        };

        let encoded = user.encode().expect("Failed to encode user");
        let decoded = UserProfile::decode(&encoded).expect("Failed to decode user");

        assert_eq!(decoded.age, user.age);
        assert_eq!(decoded.preferences.gender.len(), 3);
        assert_eq!(decoded.preferences.gender, user.preferences.gender);
        assert_eq!(decoded.meta.plan, user.meta.plan);
        assert_eq!(decoded.meta.banned, user.meta.banned);
        assert_eq!(decoded.text_embedding, user.text_embedding);
        assert_eq!(decoded.interest_embeddings, user.interest_embeddings);
    }

    #[test]
    fn test_empty_gender_preferences() {
        let user = UserProfile {
            user_id: 1,
            age: 25,
            gender: 0,
            height_cm: 180,
            likeness_score: 0.5,
            preference_score: 0.5,
            norm_rating: 0.5,
            likeness_updates: 0,
            preference_updates: 0,
            location: [0.0, 0.0],
            preferences: Preferences {
                gender: vec![],
                age_range: [18, 99],
                distance_km: 1,
                min_height_cm: 250,
            },
            meta: Meta {
                last_seen: 0,
                plan: 0,
                banned: true,
            },
            multiplier: 0.1,
            multiplier_expiry: None,
            text_embedding: [0.0; TEXT_EMB_DIM],
            interest_embeddings: [0.0; INTEREST_EMB_DIM],
        };

        let encoded = user.encode().expect("Failed to encode user");
        let decoded = UserProfile::decode(&encoded).expect("Failed to decode user");

        assert_eq!(decoded.preferences.gender.len(), 0);
        assert!(decoded.meta.banned);
    }

    #[test]
    fn test_boundary_values() {
        let user = UserProfile {
            user_id: u32::MAX,
            age: 255,            // Max u8
            gender: 0,           // Empty string
            height_cm: u16::MAX, // Max u16
            likeness_score: 1.0,
            preference_score: 0.0,
            norm_rating: 0.0, // Min normalized rating
            likeness_updates: u32::MAX,
            preference_updates: u32::MAX,
            location: [-90.0, -180.0], // Min lat/lon
            preferences: Preferences {
                gender: vec![0],
                age_range: [0, 255],     // Full u8 range
                distance_km: u32::MAX,   // Max distance
                min_height_cm: u16::MAX, // Max height
            },
            meta: Meta {
                last_seen: u64::MAX,
                plan: 1,
                banned: false,
            },
            multiplier: f32::MAX,
            multiplier_expiry: Some(u64::MAX),
            text_embedding: [f32::MAX; TEXT_EMB_DIM],
            interest_embeddings: [f32::MIN; INTEREST_EMB_DIM],
        };

        let encoded = user.encode().expect("Failed to encode user");
        let decoded = UserProfile::decode(&encoded).expect("Failed to decode user");

        assert_eq!(decoded.user_id, user.user_id);
        assert_eq!(decoded.age, 255);
        assert_eq!(decoded.gender, 0);
        assert_eq!(decoded.height_cm, u16::MAX);
        assert_eq!(decoded.likeness_score, 1.0);
        assert_eq!(decoded.preference_score, 0.0);
        assert_eq!(decoded.norm_rating, 0.0);
        assert_eq!(decoded.likeness_updates, u32::MAX);
        assert_eq!(decoded.preference_updates, u32::MAX);
        assert_eq!(decoded.preferences.distance_km, u32::MAX);
        assert_eq!(decoded.preferences.min_height_cm, u16::MAX);
        assert_eq!(decoded.meta.last_seen, u64::MAX);
        assert_eq!(decoded.multiplier, f32::MAX);
        assert_eq!(decoded.multiplier_expiry, Some(u64::MAX));
        assert_eq!(decoded.text_embedding, user.text_embedding);
        assert_eq!(decoded.interest_embeddings, user.interest_embeddings);
    }

    #[test]
    fn test_decode_invalid_data() {
        let invalid_data = vec![0xFF, 0xFF, 0xFF]; // Invalid bincode data
        let result = UserProfile::decode(&invalid_data);
        assert!(result.is_err());
    }

    #[test]
    fn test_decode_empty_data() {
        let empty_data = vec![];
        let result = UserProfile::decode(&empty_data);
        assert!(result.is_err());
    }
}
