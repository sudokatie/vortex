//! Math utilities for physics simulation.

mod transform;

pub use transform::Transform;

// Re-export glam types
pub use glam::{Mat3, Quat, Vec2, Vec3};
