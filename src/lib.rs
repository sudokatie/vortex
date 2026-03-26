//! Vortex - Physics engine with rigid body dynamics and collision detection.
//!
//! # Features
//!
//! - Rigid body simulation with forces, torques, and constraints
//! - GJK/EPA collision detection for convex shapes
//! - Sequential impulse constraint solver
//! - Distance, ball, and hinge joints
//! - Island-based sleeping
//!
//! # Example
//!
//! ```
//! use vortex::prelude::*;
//!
//! let mut world = PhysicsWorld::new();
//! world.set_gravity(Vec3::new(0.0, -9.81, 0.0));
//!
//! // Add a dynamic box
//! let body = RigidBody::new(
//!     CollisionShape::cube(Vec3::splat(0.5)),
//!     1.0,
//!     BodyType::Dynamic,
//! );
//! let handle = world.add_body(body);
//!
//! // Simulate
//! world.step(1.0 / 60.0);
//! ```

pub mod collision;
pub mod constraints;
pub mod dynamics;
pub mod math;
pub mod world;

pub mod prelude {
    //! Common imports for working with Vortex.

    pub use crate::collision::{Aabb, CollisionShape, ContactManifold};
    pub use crate::constraints::{BallJoint, DistanceJoint, HingeJoint, Joint};
    pub use crate::dynamics::{BodyType, Material, RigidBody};
    pub use crate::math::Transform;
    pub use crate::world::{BodyHandle, PhysicsWorld};

    pub use glam::{Mat3, Quat, Vec2, Vec3};
}

pub use world::PhysicsWorld;
