//! Vortex - A physics engine with rigid body dynamics and collision detection
//!
//! # Features
//!
//! - `dim3` (default): 3D physics with Vec3, Quat, full collision shapes
//! - `dim2`: 2D physics with Vec2, scalar rotation, simplified shapes
//!
//! # Example
//! ```
//! use vortex::prelude::*;
//!
//! let mut world = PhysicsWorld::new();
//! world.set_gravity(Vec3::new(0.0, -9.81, 0.0));
//!
//! // Add a dynamic body
//! let body = RigidBody::new(
//!     CollisionShape::sphere(0.5),
//!     1.0,
//!     BodyType::Dynamic
//! );
//! world.add_body(body);
//! ```

// Core modules (always available)
pub mod math;

// 3D physics modules (enabled with `dim3` feature, which is default)
#[cfg(feature = "dim3")]
pub mod collision;
#[cfg(feature = "dim3")]
pub mod constraints;
#[cfg(feature = "dim3")]
pub mod dynamics;
#[cfg(feature = "dim3")]
pub mod world;
#[cfg(feature = "dim3")]
pub mod fluid;

// 2D physics module (enabled with `dim2` feature)
#[cfg(feature = "dim2")]
pub mod dim2;

/// Prelude - commonly used types for 3D physics
#[cfg(feature = "dim3")]
pub mod prelude {
    pub use glam::{Vec3, Mat3, Quat};
    pub use crate::collision::{
        Aabb, BroadPhase, Bvh, CollisionShape, ContactManifold, ContactPoint,
        Face, SpatialHash, SweepAndPrune, gjk_intersection,
    };
    pub use crate::constraints::{
        BallJoint, ConstraintSolver, ContactConstraint, DistanceJoint,
        HingeJoint, Joint, JointLimit, JointMotor, PositionCorrection, SolverConfig,
    };
    pub use crate::dynamics::{
        BodyType, Buoyancy, Drag, ForceGenerator, ForceRegistry, Gravity,
        IntegratorType, IntegrationState, IntegrationInput, integrate_state,
        Material, PointForce, RigidBody, Spring,
        Particle, SoftBody, SoftBodyConfig, SoftSpring,
    };
    pub use crate::math::Transform;
    pub use crate::world::{
        BodyHandle, Island, IslandDetector, PhysicsStep, PhysicsWorld,
        StepConfig, StepResult,
    };
    pub use crate::fluid::{
        Kernels, poly6_value, poly6_value_2d, spiky_gradient, spiky_gradient_2d,
        viscosity_laplacian, viscosity_laplacian_2d,
    };
}

/// Prelude for 2D physics
#[cfg(all(feature = "dim2", not(feature = "dim3")))]
pub mod prelude {
    pub use glam::Vec2;
    pub use crate::dim2::*;
}
