//! Rigid body component.

use glam::{Mat3, Quat, Vec3};

use crate::collision::CollisionShape;
use crate::math::Transform;

/// Type of rigid body.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BodyType {
    /// Static bodies don't move and have infinite mass.
    Static,
    /// Kinematic bodies are moved by the user, not physics.
    Kinematic,
    /// Dynamic bodies are fully simulated.
    Dynamic,
}

/// A rigid body in the physics simulation.
#[derive(Debug, Clone)]
pub struct RigidBody {
    // Transform state
    pub position: Vec3,
    pub rotation: Quat,

    // Velocity state
    pub linear_velocity: Vec3,
    pub angular_velocity: Vec3,

    // Mass properties
    pub mass: f32,
    pub inv_mass: f32,
    pub local_inertia: Mat3,
    pub inv_inertia: Mat3,

    // Material
    pub friction: f32,
    pub restitution: f32,

    // Shape
    pub shape: CollisionShape,

    // Type and flags
    pub body_type: BodyType,
    pub is_sleeping: bool,

    // Accumulated forces (cleared each step)
    pub force: Vec3,
    pub torque: Vec3,
}

impl RigidBody {
    /// Create a new rigid body.
    pub fn new(shape: CollisionShape, mass: f32, body_type: BodyType) -> Self {
        let (inv_mass, local_inertia, inv_inertia) = if body_type == BodyType::Static || mass <= 0.0 {
            (0.0, Mat3::ZERO, Mat3::ZERO)
        } else {
            let inertia = shape.inertia_tensor(mass);
            (1.0 / mass, inertia, inertia.inverse())
        };

        Self {
            position: Vec3::ZERO,
            rotation: Quat::IDENTITY,
            linear_velocity: Vec3::ZERO,
            angular_velocity: Vec3::ZERO,
            mass,
            inv_mass,
            local_inertia,
            inv_inertia,
            friction: 0.5,
            restitution: 0.3,
            shape,
            body_type,
            is_sleeping: false,
            force: Vec3::ZERO,
            torque: Vec3::ZERO,
        }
    }

    /// Get the transform of this body.
    pub fn transform(&self) -> Transform {
        Transform::new(self.position, self.rotation)
    }

    /// Set the position of this body.
    pub fn set_position(&mut self, position: Vec3) {
        self.position = position;
    }

    /// Set the rotation of this body.
    pub fn set_rotation(&mut self, rotation: Quat) {
        self.rotation = rotation;
    }

    /// Apply a force at the center of mass.
    pub fn apply_force(&mut self, force: Vec3) {
        if self.body_type != BodyType::Dynamic {
            return;
        }
        self.force += force;
    }

    /// Apply a force at a world-space point.
    pub fn apply_force_at_point(&mut self, force: Vec3, point: Vec3) {
        if self.body_type != BodyType::Dynamic {
            return;
        }
        self.force += force;
        let r = point - self.position;
        self.torque += r.cross(force);
    }

    /// Apply a torque.
    pub fn apply_torque(&mut self, torque: Vec3) {
        if self.body_type != BodyType::Dynamic {
            return;
        }
        self.torque += torque;
    }

    /// Apply an impulse at the center of mass.
    pub fn apply_impulse(&mut self, impulse: Vec3) {
        if self.body_type != BodyType::Dynamic {
            return;
        }
        self.linear_velocity += impulse * self.inv_mass;
    }

    /// Apply an impulse at a world-space point.
    pub fn apply_impulse_at_point(&mut self, impulse: Vec3, point: Vec3) {
        if self.body_type != BodyType::Dynamic {
            return;
        }
        self.linear_velocity += impulse * self.inv_mass;
        let r = point - self.position;
        self.angular_velocity += self.world_inv_inertia() * r.cross(impulse);
    }

    /// Get the linear velocity at a world-space point.
    pub fn linear_velocity_at(&self, point: Vec3) -> Vec3 {
        let r = point - self.position;
        self.linear_velocity + self.angular_velocity.cross(r)
    }

    /// Get the world-space inverse inertia tensor.
    pub fn world_inv_inertia(&self) -> Mat3 {
        let rot = Mat3::from_quat(self.rotation);
        rot * self.inv_inertia * rot.transpose()
    }

    /// Set whether this body is sleeping.
    pub fn set_sleeping(&mut self, sleeping: bool) {
        self.is_sleeping = sleeping;
        if sleeping {
            self.linear_velocity = Vec3::ZERO;
            self.angular_velocity = Vec3::ZERO;
        }
    }

    /// Clear accumulated forces.
    pub fn clear_forces(&mut self) {
        self.force = Vec3::ZERO;
        self.torque = Vec3::ZERO;
    }

    /// Check if this body is static.
    pub fn is_static(&self) -> bool {
        self.body_type == BodyType::Static
    }

    /// Check if this body is kinematic.
    pub fn is_kinematic(&self) -> bool {
        self.body_type == BodyType::Kinematic
    }

    /// Check if this body is dynamic.
    pub fn is_dynamic(&self) -> bool {
        self.body_type == BodyType::Dynamic
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dynamic_box() -> RigidBody {
        RigidBody::new(CollisionShape::cube(Vec3::ONE), 1.0, BodyType::Dynamic)
    }

    #[test]
    fn test_new_dynamic() {
        let body = dynamic_box();
        assert_eq!(body.body_type, BodyType::Dynamic);
        assert!(body.inv_mass > 0.0);
    }

    #[test]
    fn test_new_static() {
        let body = RigidBody::new(CollisionShape::cube(Vec3::ONE), 1.0, BodyType::Static);
        assert_eq!(body.inv_mass, 0.0);
    }

    #[test]
    fn test_apply_force() {
        let mut body = dynamic_box();
        body.apply_force(Vec3::new(10.0, 0.0, 0.0));
        assert_eq!(body.force, Vec3::new(10.0, 0.0, 0.0));
    }

    #[test]
    fn test_apply_force_static_ignored() {
        let mut body = RigidBody::new(CollisionShape::cube(Vec3::ONE), 1.0, BodyType::Static);
        body.apply_force(Vec3::new(10.0, 0.0, 0.0));
        assert_eq!(body.force, Vec3::ZERO);
    }

    #[test]
    fn test_apply_impulse() {
        let mut body = dynamic_box();
        body.apply_impulse(Vec3::new(1.0, 0.0, 0.0));
        assert!(body.linear_velocity.x > 0.0);
    }

    #[test]
    fn test_apply_impulse_at_point() {
        let mut body = dynamic_box();
        body.apply_impulse_at_point(Vec3::new(0.0, 1.0, 0.0), Vec3::new(1.0, 0.0, 0.0));
        // Should induce both linear and angular velocity
        assert!(body.linear_velocity.length() > 0.0);
        assert!(body.angular_velocity.length() > 0.0);
    }

    #[test]
    fn test_linear_velocity_at() {
        let mut body = dynamic_box();
        body.linear_velocity = Vec3::new(1.0, 0.0, 0.0);
        body.angular_velocity = Vec3::new(0.0, 1.0, 0.0);
        let v = body.linear_velocity_at(Vec3::new(0.0, 0.0, 1.0));
        // Angular velocity around Y + offset in Z creates X velocity
        assert!(v.x > 1.0);
    }

    #[test]
    fn test_sleeping() {
        let mut body = dynamic_box();
        body.linear_velocity = Vec3::ONE;
        body.set_sleeping(true);
        assert!(body.is_sleeping);
        assert_eq!(body.linear_velocity, Vec3::ZERO);
    }

    #[test]
    fn test_clear_forces() {
        let mut body = dynamic_box();
        body.force = Vec3::ONE;
        body.torque = Vec3::ONE;
        body.clear_forces();
        assert_eq!(body.force, Vec3::ZERO);
        assert_eq!(body.torque, Vec3::ZERO);
    }

    #[test]
    fn test_transform() {
        let mut body = dynamic_box();
        body.position = Vec3::new(1.0, 2.0, 3.0);
        let t = body.transform();
        assert_eq!(t.position, Vec3::new(1.0, 2.0, 3.0));
    }

    #[test]
    fn test_is_type_checks() {
        let static_body = RigidBody::new(CollisionShape::sphere(1.0), 1.0, BodyType::Static);
        assert!(static_body.is_static());
        assert!(!static_body.is_dynamic());
    }
}
