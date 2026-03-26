//! Collision detection algorithms and shapes.

mod aabb;
mod broadphase;
mod epa;
mod gjk;
mod shapes;

pub use aabb::Aabb;
pub use broadphase::{BroadPhase, SpatialHash, SweepAndPrune};
pub use epa::{epa, PenetrationInfo};
pub use gjk::{gjk_distance, gjk_intersection, Simplex};
pub use shapes::CollisionShape;

// Placeholder exports
pub struct ContactManifold;
