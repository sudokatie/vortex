//! Collision detection algorithms and shapes.

mod aabb;
mod broadphase;
mod gjk;
mod shapes;

pub use aabb::Aabb;
pub use broadphase::{BroadPhase, SpatialHash, SweepAndPrune};
pub use gjk::{gjk_distance, gjk_intersection, Simplex};
pub use shapes::CollisionShape;

// Placeholder exports
pub struct ContactManifold;
