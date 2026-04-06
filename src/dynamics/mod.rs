// Rigid body dynamics module

pub mod body;
pub mod forces;
pub mod integrator;
pub mod material;
pub mod softbody;

pub use body::{RigidBody, BodyType};
pub use forces::{ForceGenerator, ForceOutput, ForceRegistry, Gravity, Spring, Drag, PointForce, Buoyancy};
pub use integrator::{IntegratorType, IntegrationState, IntegrationInput, integrate_velocity, integrate_position, integrate_state};
pub use material::Material;
pub use softbody::{Particle, Spring as SoftSpring, SoftBody, SoftBodyConfig};
