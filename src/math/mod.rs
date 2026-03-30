//! Math utilities for physics simulation.
//!
//! This module provides vector, matrix, and quaternion types along with
//! physics-specific extensions. Types are re-exported from glam with
//! additional utility traits.

mod mat3;
mod quat;
mod transform;
mod vec2;
mod vec3;

pub use mat3::{Mat3, Mat3Ext};
pub use quat::{Quat, QuatExt};
pub use transform::Transform;
pub use vec2::{Vec2, Vec2Ext};
pub use vec3::{Vec3, Vec3Ext};
