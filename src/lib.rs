//! Vortex - A physics engine with rigid body dynamics and collision detection
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

pub mod collision;
pub mod constraints;
pub mod dynamics;
pub mod math;
pub mod world;

/// Prelude - commonly used types
pub mod prelude {
    pub use glam::{Vec3, Mat3, Quat};
    pub use crate::collision::{
        Aabb, BroadPhase, Bvh, CollisionShape, ContactManifold, ContactPoint,
        Face, SpatialHash, SweepAndPrune, gjk_intersection,
    };
    pub use crate::constraints::{
        BallJoint, ConstraintSolver, ContactConstraint, DistanceJoint,
        HingeJoint, Joint, JointLimit, JointMotor, SolverConfig,
    };
    pub use crate::dynamics::{
        BodyType, Buoyancy, Drag, ForceGenerator, ForceRegistry, Gravity,
        IntegratorType, IntegrationState, IntegrationInput, integrate_state,
        Material, PointForce, RigidBody, Spring,
    };
    pub use crate::math::Transform;
    pub use crate::world::{
        BodyHandle, Island, IslandDetector, PhysicsStep, PhysicsWorld, 
        StepConfig, StepResult,
    };
}
