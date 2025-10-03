use std::{collections::HashSet, sync::Arc};

use bincode::{
    Decode, Encode,
    config::{self},
};
use hnsw_rs::hnsw::FilterT;
use rand::Rng;
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
const FILTER_INTEREST_OVERLAP_WEIGHT: f32 = 0.08;

/// Filter implementation for HNSW search
struct UserFilter {
    allowed_ids: std::collections::HashSet<usize>,
}

impl FilterT for UserFilter {
    fn hnsw_filter(&self, id: &hnsw_rs::prelude::DataId) -> bool {
        self.allowed_ids.contains(id)
    }
}

/// Represents a user's preferences for matching
#[derive(Debug, Encode, Decode, Clone)]
pub struct Preferences {
    pub gender: Vec<u8>, // Gender indeces the user is interested in (0=M, 1=F, 2=O)
    pub age_range: [u8; 2], // min and max age
    pub distance_km: u32, // maximum distance
    pub min_height_cm: u16, // minimum height of a person
}

#[derive(Debug, Encode, Decode, Clone)]
pub struct DisplayImage {
    pub storage_id: String,
    pub photo_index: u8, // 0 is primary
    pub verified: bool,
}

#[derive(Debug, Encode, Decode, Clone)]
pub struct DisplayMeta {
    pub name: String,
    pub bio: String,
    pub interests: Vec<u32>,         // Interest IDs
    pub primary_interests: Vec<u32>, // Primary Interest IDs
    pub images: Vec<DisplayImage>,   // Storage IDs for user images
    pub location_name: String,
    pub looking_for: Option<u16>,
    pub institution_name: Option<String>, // Name of the company or institution
    pub institution_title: Option<String>, // Title or role at the institution
}

/// Metadata for internal tracking
#[derive(Debug, Encode, Decode)]
pub struct Meta {
    pub last_seen: u64, // Unix timestamp
    pub banned: bool,
    pub incognito: bool,
    pub swipe_streak: u32,          // Number of days with at least one swipe
    pub longest_swipe_streak: u32,  // Longest swipe streak ever achieved
    pub last_swipe_day: u32,        // The last day (as days since epoch) the user swiped
    pub plan: u8,                   // 0=free, 1=premium
    pub swipes_until_rateable: u32, // Number of swipes until user can be rated, 0 means ready for rating
    pub is_demo_account: bool, // Is this a demo/test user (distance is not factored in, when they show up as candidates)
}

/// Full user profile with multiplier for boosting
#[derive(Debug, Encode, Decode)]
pub struct UserProfile {
    pub user_id: u32,
    pub age: u8,
    pub gender: u8,             // Gender index (0=M, 1=F, 2=O)
    pub height_cm: Option<u16>, // Height in cm

    pub likeness_score: f32,      // Overall likeness score (0.0 to 1.0)
    pub preference_score: f32,    // Overall preference score (0.0 to 1.0)
    pub norm_rating: f32,         // normalized rating score (0.0 to 1.0)
    pub norm_rating_updates: u32, // Number of updates to normalized rating

    pub likeness_updates: u32,   // Number of updates to likeness score
    pub preference_updates: u32, // Number of updates to preference score

    pub location: [f64; 2],       // lat/lon
    pub preferences: Preferences, // Matching preference / filters

    pub meta: Meta,
    pub display_meta: DisplayMeta,

    pub multiplier: f32, // exposure / scoring multiplier, like boosters
    pub multiplier_expiry: Option<u64>, // Unix timestamp when multiplier expires

    // Embeddings
    pub text_embedding: [f32; TEXT_EMB_DIM], // 50-d embedding for bio/interests
    pub interest_embeddings: [f32; INTEREST_EMB_DIM], // 16-d embedding for interests
}

impl UserProfile {
    /// Function to create a new user (we shall infer embeddings)
    #[allow(clippy::too_many_arguments)]
    pub fn create_new(
        db: &Arc<DB>,
        user_id: u32,
        age: u8,
        gender: u8,
        height_cm: Option<u16>,
        location: [f64; 2],
        plan: u8,
        is_demo_account: bool,
        preferences: Preferences,
        display_meta: DisplayMeta,
    ) -> Result<Self, MatchError> {
        // 1. Calculate text embedding
        let (t_embed, i_embed) =
            embed::calculate_persistent_embeddings(&display_meta.interests, &display_meta.bio)?;

        // Get current unix timestamp
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();

        // Generate initial random swipes_until_rateable between 10-50
        let mut rng = rand::rng();
        let swipes_until_rateable = rng.random_range(10..=50);

        // 2. Create user profile with default values
        let user = UserProfile {
            user_id,
            age,
            gender,
            height_cm,
            likeness_score: 0.5,   // Neutral initial likeness score
            preference_score: 0.5, // Neutral initial preference score
            norm_rating: 0.5,
            norm_rating_updates: 0,
            likeness_updates: 0,
            preference_updates: 0,
            location,
            preferences,
            meta: Meta {
                last_seen: now,
                banned: false,
                incognito: false,
                swipe_streak: 0,
                longest_swipe_streak: 0,
                last_swipe_day: 0,
                plan,
                swipes_until_rateable,
                is_demo_account,
            },
            display_meta,
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

    pub fn has_swiped_today(&self) -> bool {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();
        let today_days = (now / 86400) as u32; // days since epoch
        self.meta.last_swipe_day == today_days
    }

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

        // IDEA: See if we have some users in cache still from previous top_k overfetches.
        // that way we don't need to hit hnsw every time, if we still have some left over
        // candidates from last time.

        // Collect search results and then DROP the lock before calling load_user
        let search_results = {
            let hnsw = hnsw::get_hnsw_index();
            let hnsw_read = hnsw.read()?;
            let filter = UserFilter { allowed_ids };
            hnsw_read.search_filter(&combined, top_k * 5, 16, Some(&filter))
        }; // Read lock is dropped here!

        // Here we have the IDs of the candidates
        let mut users: Vec<(f32, UserProfile)> = Vec::with_capacity(search_results.len());
        let mut added_user_ids: HashSet<u32> = HashSet::with_capacity(search_results.len());

        // Don't flood the results with people that liked us
        let max_liked_by = (top_k as f32 * 0.25).ceil() as usize;
        // These are the user ids that swiped us, without us swiping them
        let (liked_by, _total_likes) = self.get_potential_matches(db, 0, max_liked_by)?;
        for (user_id, _timestamp) in &liked_by {
            let candidate = match UserProfile::load_user(db, *user_id) {
                Ok(u) => u,
                Err(e) => {
                    if matches!(e, MatchError::UserNotFound) {
                        continue;
                    } else {
                        return Err(e);
                    }
                } // Skip if user not found
            };

            let (passed, score) = self.post_filter(&candidate, 0);
            if passed {
                added_user_ids.insert(candidate.user_id);
                users.push((score, candidate));
            }
        }

        for level in 0..3 {
            for n in &search_results {
                let user_id = n.d_id as u32;
                // Skip if already added
                if added_user_ids.contains(&user_id) {
                    continue;
                }

                let candidate = match UserProfile::load_user(db, user_id) {
                    Ok(u) => u,
                    Err(e) => {
                        if matches!(e, MatchError::UserNotFound) {
                            continue;
                        } else {
                            return Err(e);
                        }
                    } // Skip if user not found
                };

                let (passed, score) = self.post_filter(&candidate, level);
                if passed {
                    added_user_ids.insert(candidate.user_id);
                    users.push((score, candidate));
                }
            }

            // If we have enough users, we don't need to
            // escalate to the next level
            if users.len() >= top_k {
                break;
            }
        }

        // Now we must sort on the users score
        users.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));

        // IDEA: Instead of truncating, we could add the remaining to the cache
        // and serve from cache next time
        users.truncate(top_k);

        Ok(users)
    }

    /// Get potential matches with pagination
    /// This function retrieves a paginated list of user IDs and timestamps who have swiped right (liked) the current user
    /// but whom the current user has not yet swiped on. Results are ordered by newest likes first.
    ///
    /// * `offset` - Number of results to skip (0 for first page)
    /// * `chunk_size` - Number of results to return per page
    /// Returns (Vec<(user_id, timestamp)>, total_count) where total_count is the total number of potential matches
    pub fn get_potential_matches(
        &self,
        db: &Arc<DB>,
        offset: usize,
        chunk_size: usize,
    ) -> Result<(Vec<(u32, u64)>, u64), MatchError> {
        let mut liked_by_with_timestamp = Vec::new();
        let prefix = format!("swipe-in:{}:", self.user_id);
        let iter = db.prefix_iterator(prefix.as_bytes());

        // Collect all potential matches with their timestamps
        for item in iter {
            let (key, value) = item?;
            if let Ok(key_str) = std::str::from_utf8(&key)
                && key_str.starts_with(&prefix)
                && let Some(id_str) = key_str.strip_prefix(&prefix)
                && let Ok(user_id) = id_str.parse::<u32>()
            {
                // Parse the swipe value and timestamp (format: "1:1234567890" or "0:1234567890")
                if let Ok(swipe_data) = std::str::from_utf8(&value) {
                    let parts: Vec<&str> = swipe_data.split(':').collect();
                    if parts.len() == 2 {
                        let swipe_value = parts[0];
                        let timestamp = parts[1].parse::<u64>().unwrap_or(0);

                        // Also check we have not swiped them back already
                        let already_swiped = self.is_liked_by(db, user_id)?;
                        if swipe_value == "1" && !already_swiped {
                            liked_by_with_timestamp.push((user_id, timestamp));
                        }
                    }
                }
            }
        }

        // Store total count before pagination
        let total_count = liked_by_with_timestamp.len() as u64;

        // Sort by timestamp (newest first)
        liked_by_with_timestamp.sort_by(|a, b| b.1.cmp(&a.1));

        // Apply pagination
        let start_idx = offset * chunk_size;

        let liked_by: Vec<(u32, u64)> = liked_by_with_timestamp
            .into_iter()
            .skip(start_idx)
            .take(chunk_size)
            .collect();

        Ok((liked_by, total_count))
    }

    /// Update bio
    pub fn update_bio(&mut self, db: &Arc<DB>, new_bio: &str) -> Result<(), MatchError> {
        self.display_meta.bio = new_bio.to_string();
        self.update_last_seen();

        // Recalculate text embeddings
        self.text_embedding = embed::text_to_embedding(new_bio);

        let key = format!("user:{}", self.user_id);
        let value = self.encode()?;
        db.put(key.as_bytes(), &value)?;

        Ok(())
    }

    /// Update (primary) interests
    pub fn update_interests(
        &mut self,
        db: &Arc<DB>,
        new_interests: &[u32],
        new_primary_interests: &[u32],
    ) -> Result<(), MatchError> {
        let old_interests = self.display_meta.interests.clone();

        self.display_meta.interests = new_interests.to_vec();
        self.display_meta.primary_interests = new_primary_interests.to_vec();
        self.update_last_seen();

        // Recalculate interest embeddings, but only if interests changed, so not if only primary changed
        if self.display_meta.interests != old_interests {
            self.text_embedding = embed::text_to_embedding(&self.display_meta.bio);
        }

        let key = format!("user:{}", self.user_id);
        let value = self.encode()?;
        db.put(key.as_bytes(), &value)?;

        Ok(())
    }

    /// Update height
    /// This updates the user's height and persists it to the database.
    pub fn update_height(
        &mut self,
        db: &Arc<DB>,
        new_height_cm: Option<u16>,
    ) -> Result<(), MatchError> {
        self.height_cm = new_height_cm;
        self.update_last_seen();

        let key = format!("user:{}", self.user_id);
        let value = self.encode()?;
        db.put(key.as_bytes(), &value)?;

        Ok(())
    }

    /// Update images
    pub fn update_images(
        &mut self,
        db: &Arc<DB>,
        new_image_data: &[DisplayImage],
    ) -> Result<(), MatchError> {
        self.display_meta.images = new_image_data.to_vec();
        self.update_last_seen();

        let key = format!("user:{}", self.user_id);
        let value = self.encode()?;
        db.put(key.as_bytes(), &value)?;

        Ok(())
    }

    /// Update looking for
    pub fn update_looking_for(
        &mut self,
        db: &Arc<DB>,
        new_looking_for: Option<u16>,
    ) -> Result<(), MatchError> {
        self.display_meta.looking_for = new_looking_for;
        self.update_last_seen();

        let key = format!("user:{}", self.user_id);
        let value = self.encode()?;
        db.put(key.as_bytes(), &value)?;

        Ok(())
    }

    /// Update institution
    pub fn update_institution(
        &mut self,
        db: &Arc<DB>,
        institution_name: Option<String>,
        institution_title: Option<String>,
    ) -> Result<(), MatchError> {
        self.display_meta.institution_name = institution_name;
        self.display_meta.institution_title = institution_title;
        self.update_last_seen();

        let key = format!("user:{}", self.user_id);
        let value = self.encode()?;
        db.put(key.as_bytes(), &value)?;

        Ok(())
    }

    /// Update location
    /// This updates the user's location and persists it to the database.
    /// It also updates the last seen timestamp and the location_name in display metadata.
    pub fn update_location(
        &mut self,
        db: &Arc<DB>,
        latitude: f64,
        longitude: f64,
        location_name: String,
    ) -> Result<(), MatchError> {
        self.location = [latitude, longitude];
        self.display_meta.location_name = location_name;
        self.update_last_seen();

        let key = format!("user:{}", self.user_id);
        let value = self.encode()?;
        db.put(key.as_bytes(), &value)?;

        Ok(())
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

    /// Apply a ban to the user
    /// This function bans the user, preventing them from appearing in searches.
    pub fn apply_ban(&mut self, db: &Arc<DB>) -> Result<(), MatchError> {
        self.meta.banned = true;

        let key = format!("user:{}", self.user_id);
        let value = self.encode()?;
        db.put(key.as_bytes(), &value)?;
        Ok(())
    }

    /// Set incognito mode
    /// This function sets the user's incognito mode status.
    pub fn set_incognito(&mut self, db: &Arc<DB>, incognito: bool) -> Result<(), MatchError> {
        self.meta.incognito = incognito;
        self.update_last_seen();

        let key = format!("user:{}", self.user_id);
        let value = self.encode()?;
        db.put(key.as_bytes(), &value)?;

        Ok(())
    }

    /// Apply multiplier (booster)
    /// This function applies a multiplier (booster) to the user's profile for increased exposure.
    pub fn apply_multiplier(
        &mut self,
        db: &Arc<DB>,
        multiplier: f32,
        duration_secs: u64,
    ) -> Result<(), MatchError> {
        self.multiplier = multiplier;
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();
        self.multiplier_expiry = Some(now + duration_secs);
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

        // Update target user's swipes_until_rateable
        if target_user.meta.swipes_until_rateable == 0 {
            // Generate new random value between 10-50
            let mut rng = rand::rng();
            target_user.meta.swipes_until_rateable = rng.random_range(10..=50);
        } else {
            // Decrement the counter
            target_user.meta.swipes_until_rateable -= 1;
        }

        // Update timestamp only for the active user (the one swiping)
        self.update_last_seen();

        // We must also update the swipe streak
        // Now if there was already a swipe within 24 hours we do not increment
        // if there was a swipe yesterday but not today we increment
        // if there was no swipe yesterday we reset the streak
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();
        let today_days = (now / 86400) as u32; // days since epoch
        if self.meta.last_swipe_day == today_days {
            // Already swiped today, do nothing
        } else if self.meta.last_swipe_day == today_days - 1 {
            // Swiped yesterday, increment streak
            self.meta.swipe_streak += 1;

            // Update longest streak if current streak is now longer
            if self.meta.swipe_streak > self.meta.longest_swipe_streak {
                self.meta.longest_swipe_streak = self.meta.swipe_streak;
            }

            self.meta.last_swipe_day = today_days;
        } else {
            // No swipe yesterday, reset streak
            self.meta.swipe_streak = 1;

            // Check if this single day streak should update longest (in case longest was 0)
            if self.meta.swipe_streak > self.meta.longest_swipe_streak {
                self.meta.longest_swipe_streak = self.meta.swipe_streak;
            }

            self.meta.last_swipe_day = today_days;
        }

        // Register the outgoing swipe in RocksDB with timestamp (format: "swipe:<user_id>:<target_user_id>")
        let swipe_key = format!("swipe:{}:{}", self.user_id, target_user_id);
        let swipe_value = format!("{}:{}", if positive { "1" } else { "0" }, now);
        db.put(swipe_key.as_bytes(), swipe_value.as_bytes())?;

        // This one is so we can iterate for matches quicker
        let swipe_in_key = format!("swipe-in:{}:{}", target_user_id, self.user_id);
        db.put(swipe_in_key.as_bytes(), swipe_value.as_bytes())?;

        // Save both users to database
        let key = format!("user:{}", self.user_id);
        let value = self.encode()?;
        db.put(key.as_bytes(), &value)?;

        let target_key = format!("user:{}", target_user_id);
        let target_value = target_user.encode()?;
        db.put(target_key.as_bytes(), &target_value)?;

        // We also should update the daily likes statistics
        update_daily_swipe_statistic(db, positive)?;

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

        let liked_back = self.is_liked_by(db, target_user_id)?;
        Ok(liked_back)
    }

    /// Process a new rating for the user
    /// This updates the user's normalized rating based on a new rating input (1-10 scale).
    /// It uses a simple moving average to update the normalized rating.
    /// Takes into account the number of previous updates to weight the new rating appropriately.
    pub fn apply_rating(&mut self, db: &Arc<DB>, new_rating: f32) -> Result<(), MatchError> {
        let clamped_rating = new_rating.clamp(1.0, 10.0);
        let normalized = normalize_rating(clamped_rating);

        // Calculate weight factor based on number of previous updates
        // More updates = less impact from new rating (stabilizes over time)
        let weight_factor = 1.0 / (1.0 + self.norm_rating_updates as f32 * 0.1);

        // Update normalized rating using weighted average
        let change = (normalized - self.norm_rating) * weight_factor;
        self.norm_rating = (self.norm_rating + change).clamp(0.0, 1.0);

        // Increment the number of rating updates
        self.norm_rating_updates += 1;

        // Save updated user to database
        let key = format!("user:{}", self.user_id);
        let value = self.encode()?;
        db.put(key.as_bytes(), &value)?;

        Ok(())
    }

    fn is_liked_by(&self, db: &Arc<DB>, other_user_id: u32) -> Result<bool, MatchError> {
        // Because we have both user ids we can reconstruct the original swipe key
        let swipe_key = format!("swipe:{}:{}", other_user_id, self.user_id);
        let value = db.get(swipe_key.as_bytes())?;
        if let Some(v) = value
            && let Ok(swipe_data) = std::str::from_utf8(&v)
        {
            // Parse the swipe value and timestamp (format: "1:1234567890" or "0:1234567890")
            let parts: Vec<&str> = swipe_data.split(':').collect();
            if parts.len() == 2 {
                let swipe_value = parts[0];
                return Ok(swipe_value == "1");
            }
        }
        Ok(false)
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

        if strictness_level == 0 && !candidate.meta.is_demo_account {
            // Age range
            if candidate.age < min_age || candidate.age > max_age {
                return (false, 0.0);
            }

            if distance > max_distance as f64 {
                return (false, 0.0);
            }
        } else if strictness_level == 1 && !candidate.meta.is_demo_account {
            // Age range
            if candidate.age < min_age || candidate.age > max_age {
                return (false, 0.0);
            }

            // Update max distance to be more lenient
            max_distance = (max_distance as f32 * 1.5) as u32;

            if distance > max_distance as f64 {
                return (false, 0.0);
            }
        } else if !candidate.meta.is_demo_account {
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
        // If the candidate has height info then we check against it
        if let Some(h) = candidate.height_cm
            && h < min_height
            && !candidate.meta.is_demo_account
        {
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

        // Calculate interest overlap using Jaccard similarity
        let self_interests: HashSet<u32> = self.display_meta.interests.iter().copied().collect();
        let candidate_interests: HashSet<u32> =
            candidate.display_meta.interests.iter().copied().collect();

        let intersection = self_interests.intersection(&candidate_interests).count();
        let union = self_interests.union(&candidate_interests).count();

        let interest_overlap = if union > 0 {
            intersection as f32 / union as f32
        } else {
            0.0
        };

        let score = (likeness * FILTER_LIKENESS_WEIGHT
            + rating * FILTER_RATING_WEIGHT
            + closeness * FILTER_DISTANCE_WEIGHT
            + interest_overlap * FILTER_INTEREST_OVERLAP_WEIGHT)
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
}

fn normalize_rating(mut rating: f32) -> f32 {
    rating = rating.clamp(1.0, 10.0);
    (rating - 1.0) / 9.0
}

/// Update swipe statistic, which is for each day
/// This function updates the global swipe statistics for all users.
/// It increments the swipe count for the current day in the database.
fn update_daily_swipe_statistic(db: &Arc<DB>, positive: bool) -> Result<(), MatchError> {
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();
    let today_days = (now / 86400) as u32; // days since epoch

    let stat_key = format!("stat:swipes:{}", today_days);
    let (mut positive_count, mut negative_count) = match db.get(stat_key.as_bytes())? {
        Some(v) => {
            let count_str = std::str::from_utf8(&v).unwrap_or("0:0");
            let parts: Vec<&str> = count_str.split(':').collect();
            let pos = parts.first().unwrap_or(&"0").parse::<u32>().unwrap_or(0);
            let neg = parts.get(1).unwrap_or(&"0").parse::<u32>().unwrap_or(0);
            (pos, neg)
        }
        None => (0, 0),
    };

    if positive {
        positive_count += 1;
    } else {
        negative_count += 1;
    }

    let new_value = format!("{}:{}", positive_count, negative_count);
    db.put(stat_key.as_bytes(), new_value.as_bytes())?;

    Ok(())
}

/// Get swipe statistics for the last `days` days
/// This function retrieves the swipe statistics for the last `days` days.
/// Returns Vec<(positive_count, negative_count)> where index 0 is today, index 1 is yesterday, etc.
pub fn get_daily_swipe_statistics(
    db: &Arc<DB>,
    days_ago: u32,
) -> Result<Vec<(u32, u32)>, MatchError> {
    let mut stats = Vec::with_capacity(days_ago as usize);
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();
    let today_days = (now / 86400) as u32; // days since epoch

    // Get the swipe statistics for each day from today back to days_ago
    for day_offset in 0..days_ago {
        let target_day = today_days - day_offset;
        let stat_key = format!("stat:swipes:{}", target_day);

        let (positive_count, negative_count) = match db.get(stat_key.as_bytes())? {
            Some(v) => {
                let count_str = std::str::from_utf8(&v).unwrap_or("0:0");
                let parts: Vec<&str> = count_str.split(':').collect();
                let pos = parts.first().unwrap_or(&"0").parse::<u32>().unwrap_or(0);
                let neg = parts.get(1).unwrap_or(&"0").parse::<u32>().unwrap_or(0);
                (pos, neg)
            }
            None => (0, 0),
        };

        stats.push((positive_count, negative_count));
    }

    Ok(stats)
}

/// Delete a user
/// This function removes the user from the database.
pub fn delete_user(db: &Arc<DB>, user_id: u32) -> Result<(), MatchError> {
    let key = format!("user:{}", user_id);
    db.delete(key.as_bytes())?;

    Ok(())
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
        if let Ok(key_str) = std::str::from_utf8(&key)
            && key_str.starts_with("user:")
        {
            // Extract user ID from key
            if let Some(id_str) = key_str.strip_prefix("user:")
                && let Ok(user_id) = id_str.parse::<usize>()
            {
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

                let likeness_embedding = embed::likeness_to_vector(user_profile.likeness_score);
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
            height_cm: Some(190),
            likeness_score: 0.5,
            preference_score: 0.5,
            norm_rating: 0.75,
            norm_rating_updates: 3,
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
                banned: false,
                incognito: false,
                swipe_streak: 0,
                longest_swipe_streak: 0,
                last_swipe_day: 0,
                plan: 0,
                swipes_until_rateable: 25,
                is_demo_account: false,
            },
            display_meta: DisplayMeta {
                name: "Alice".to_string(),
                bio: "Love hiking and outdoor adventures.".to_string(),
                interests: vec![1, 2, 3],
                primary_interests: vec![1, 3],
                images: vec![
                    DisplayImage {
                        storage_id: "img1".to_string(),
                        photo_index: 0,
                        verified: true,
                    },
                    DisplayImage {
                        storage_id: "img2".to_string(),
                        photo_index: 1,
                        verified: false,
                    },
                ],
                location_name: "Amsterdam".to_string(),
                looking_for: Some(1),
                institution_name: Some("University of Amsterdam".to_string()),
                institution_title: Some("Student".to_string()),
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
        assert_eq!(decoded.norm_rating_updates, user.norm_rating_updates);
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
        assert_eq!(decoded.meta.banned, user.meta.banned);
        assert_eq!(decoded.meta.incognito, user.meta.incognito);
        assert_eq!(decoded.meta.swipe_streak, user.meta.swipe_streak);
        assert_eq!(
            decoded.meta.longest_swipe_streak,
            user.meta.longest_swipe_streak
        );
        assert_eq!(decoded.meta.last_swipe_day, user.meta.last_swipe_day);
        assert_eq!(decoded.meta.plan, user.meta.plan);
        assert_eq!(decoded.display_meta.name, user.display_meta.name);
        assert_eq!(decoded.display_meta.bio, user.display_meta.bio);
        assert_eq!(decoded.display_meta.interests, user.display_meta.interests);
        assert_eq!(
            decoded.display_meta.primary_interests,
            user.display_meta.primary_interests
        );
        assert_eq!(
            decoded.display_meta.images.len(),
            user.display_meta.images.len()
        );
        assert_eq!(
            decoded.display_meta.location_name,
            user.display_meta.location_name
        );
        assert_eq!(
            decoded.display_meta.looking_for,
            user.display_meta.looking_for
        );
        assert_eq!(
            decoded.display_meta.institution_name,
            user.display_meta.institution_name
        );
        assert_eq!(
            decoded.display_meta.institution_title,
            user.display_meta.institution_title
        );
        assert_eq!(decoded.multiplier, user.multiplier);
        assert_eq!(decoded.multiplier_expiry, user.multiplier_expiry);
        assert_eq!(decoded.text_embedding, user.text_embedding);
        assert_eq!(decoded.interest_embeddings, user.interest_embeddings);
        assert_eq!(
            decoded.meta.swipes_until_rateable,
            user.meta.swipes_until_rateable
        );
    }

    #[test]
    fn test_multiple_gender_preferences() {
        let user = UserProfile {
            user_id: 1,
            age: 30,
            gender: 2,
            height_cm: Some(175),
            likeness_score: 0.5,
            preference_score: 0.5,
            norm_rating: 0.85,
            norm_rating_updates: 5,
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
                banned: false,
                incognito: false,
                swipe_streak: 0,
                longest_swipe_streak: 0,
                last_swipe_day: 0,
                plan: 0,
                swipes_until_rateable: 42,
                is_demo_account: false,
            },
            display_meta: DisplayMeta {
                name: "Bob".to_string(),
                bio: "Tech enthusiast and foodie.".to_string(),
                interests: vec![4, 5, 6],
                primary_interests: vec![5],
                images: vec![DisplayImage {
                    storage_id: "img3".to_string(),
                    photo_index: 0,
                    verified: true,
                }],
                location_name: "New York".to_string(),
                looking_for: None,
                institution_name: None,
                institution_title: None,
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
        assert_eq!(decoded.meta.banned, user.meta.banned);
        assert_eq!(decoded.display_meta.name, user.display_meta.name);
        assert_eq!(decoded.display_meta.bio, user.display_meta.bio);
        assert_eq!(decoded.text_embedding, user.text_embedding);
        assert_eq!(decoded.interest_embeddings, user.interest_embeddings);
    }

    #[test]
    fn test_empty_gender_preferences() {
        let user = UserProfile {
            user_id: 1,
            age: 25,
            gender: 0,
            height_cm: Some(180),
            likeness_score: 0.5,
            preference_score: 0.5,
            norm_rating: 0.5,
            norm_rating_updates: 0,
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
                banned: true,
                incognito: false,
                swipe_streak: 0,
                longest_swipe_streak: 0,
                last_swipe_day: 0,
                plan: 0,
                swipes_until_rateable: 15,
                is_demo_account: false,
            },
            display_meta: DisplayMeta {
                name: "".to_string(),
                bio: "".to_string(),
                interests: vec![],
                primary_interests: vec![],
                images: vec![],
                location_name: "".to_string(),
                looking_for: None,
                institution_name: None,
                institution_title: None,
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
            age: 255,                  // Max u8
            gender: 0,                 // Empty string
            height_cm: Some(u16::MAX), // Max u16
            likeness_score: 1.0,
            preference_score: 0.0,
            norm_rating: 0.0, // Min normalized rating
            norm_rating_updates: u32::MAX,
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
                banned: false,
                incognito: false,
                swipe_streak: 0,
                longest_swipe_streak: 0,
                last_swipe_day: 0,
                plan: 0,
                swipes_until_rateable: u32::MAX,
                is_demo_account: false,
            },
            display_meta: DisplayMeta {
                name: "".to_string(),
                bio: "".to_string(),
                interests: vec![],
                primary_interests: vec![],
                images: vec![],
                location_name: "".to_string(),
                looking_for: None,
                institution_name: None,
                institution_title: None,
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
        assert_eq!(decoded.height_cm, Some(u16::MAX));
        assert_eq!(decoded.likeness_score, 1.0);
        assert_eq!(decoded.preference_score, 0.0);
        assert_eq!(decoded.norm_rating, 0.0);
        assert_eq!(decoded.likeness_updates, u32::MAX);
        assert_eq!(decoded.preference_updates, u32::MAX);
        assert_eq!(decoded.preferences.distance_km, u32::MAX);
        assert_eq!(decoded.preferences.min_height_cm, u16::MAX);
        assert_eq!(decoded.meta.last_seen, u64::MAX);
        assert!(!decoded.meta.banned);
        assert!(!decoded.meta.incognito);
        assert_eq!(decoded.meta.swipe_streak, 0);
        assert_eq!(decoded.meta.longest_swipe_streak, 0);
        assert_eq!(decoded.meta.last_swipe_day, 0);
        assert_eq!(decoded.meta.plan, 0);
        assert_eq!(decoded.display_meta.name, "");
        assert_eq!(decoded.display_meta.bio, "");
        assert_eq!(decoded.display_meta.interests.len(), 0);
        assert_eq!(decoded.display_meta.primary_interests.len(), 0);
        assert_eq!(decoded.display_meta.images.len(), 0);
        assert_eq!(decoded.display_meta.location_name, "");
        assert_eq!(decoded.display_meta.looking_for, None);
        assert_eq!(decoded.display_meta.institution_name, None);
        assert_eq!(decoded.display_meta.institution_title, None);
        assert_eq!(decoded.multiplier, f32::MAX);
        assert_eq!(decoded.multiplier_expiry, Some(u64::MAX));
        assert_eq!(decoded.text_embedding, user.text_embedding);
        assert_eq!(decoded.interest_embeddings, user.interest_embeddings);
        assert_eq!(decoded.meta.swipes_until_rateable, u32::MAX);
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
