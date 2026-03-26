//! Physics world container and simulation stepping.

use glam::Vec3;
use slotmap::{new_key_type, SlotMap};

use crate::dynamics::RigidBody;

new_key_type! {
    /// Handle to a rigid body in the physics world.
    pub struct BodyHandle;
}

/// The physics world containing all bodies and constraints.
pub struct PhysicsWorld {
    bodies: SlotMap<BodyHandle, RigidBody>,
    gravity: Vec3,
}

impl Default for PhysicsWorld {
    fn default() -> Self {
        Self::new()
    }
}

impl PhysicsWorld {
    /// Create a new empty physics world.
    pub fn new() -> Self {
        Self {
            bodies: SlotMap::with_key(),
            gravity: Vec3::new(0.0, -9.81, 0.0),
        }
    }

    /// Set the gravity vector.
    pub fn set_gravity(&mut self, gravity: Vec3) {
        self.gravity = gravity;
    }

    /// Get the gravity vector.
    pub fn gravity(&self) -> Vec3 {
        self.gravity
    }

    /// Add a rigid body to the world.
    pub fn add_body(&mut self, body: RigidBody) -> BodyHandle {
        self.bodies.insert(body)
    }

    /// Remove a rigid body from the world.
    pub fn remove_body(&mut self, handle: BodyHandle) -> Option<RigidBody> {
        self.bodies.remove(handle)
    }

    /// Get a reference to a body.
    pub fn get_body(&self, handle: BodyHandle) -> Option<&RigidBody> {
        self.bodies.get(handle)
    }

    /// Get a mutable reference to a body.
    pub fn get_body_mut(&mut self, handle: BodyHandle) -> Option<&mut RigidBody> {
        self.bodies.get_mut(handle)
    }

    /// Get the number of bodies in the world.
    pub fn body_count(&self) -> usize {
        self.bodies.len()
    }

    /// Iterate over all bodies.
    pub fn bodies(&self) -> impl Iterator<Item = (BodyHandle, &RigidBody)> {
        self.bodies.iter()
    }

    /// Iterate over all bodies mutably.
    pub fn bodies_mut(&mut self) -> impl Iterator<Item = (BodyHandle, &mut RigidBody)> {
        self.bodies.iter_mut()
    }

    /// Step the simulation forward by dt seconds.
    pub fn step(&mut self, dt: f32) {
        // Simple semi-implicit Euler integration for now
        // Full implementation will have:
        // 1. Apply forces (gravity, etc.)
        // 2. Broad phase collision detection
        // 3. Narrow phase collision detection
        // 4. Solve constraints (contacts + joints)
        // 5. Integrate velocities
        // 6. Integrate positions

        for (_, body) in self.bodies.iter_mut() {
            if !body.is_dynamic() || body.is_sleeping {
                continue;
            }

            // Apply gravity
            body.apply_force(self.gravity * body.mass);

            // Integrate velocity
            let acceleration = body.force * body.inv_mass;
            body.linear_velocity += acceleration * dt;

            let angular_acceleration = body.world_inv_inertia() * body.torque;
            body.angular_velocity += angular_acceleration * dt;

            // Apply damping
            body.linear_velocity *= 0.99;
            body.angular_velocity *= 0.99;

            // Integrate position
            body.position += body.linear_velocity * dt;

            // Integrate rotation
            let omega = body.angular_velocity;
            let dq = glam::Quat::from_xyzw(
                omega.x * 0.5 * dt,
                omega.y * 0.5 * dt,
                omega.z * 0.5 * dt,
                0.0,
            ) * body.rotation;
            body.rotation = (body.rotation + dq).normalize();

            // Clear forces
            body.clear_forces();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::collision::CollisionShape;
    use crate::dynamics::BodyType;

    #[test]
    fn test_new() {
        let world = PhysicsWorld::new();
        assert_eq!(world.body_count(), 0);
    }

    #[test]
    fn test_add_body() {
        let mut world = PhysicsWorld::new();
        let body = RigidBody::new(CollisionShape::sphere(1.0), 1.0, BodyType::Dynamic);
        let handle = world.add_body(body);
        assert_eq!(world.body_count(), 1);
        assert!(world.get_body(handle).is_some());
    }

    #[test]
    fn test_remove_body() {
        let mut world = PhysicsWorld::new();
        let body = RigidBody::new(CollisionShape::sphere(1.0), 1.0, BodyType::Dynamic);
        let handle = world.add_body(body);
        world.remove_body(handle);
        assert_eq!(world.body_count(), 0);
    }

    #[test]
    fn test_gravity() {
        let mut world = PhysicsWorld::new();
        world.set_gravity(Vec3::new(0.0, -10.0, 0.0));
        assert_eq!(world.gravity(), Vec3::new(0.0, -10.0, 0.0));
    }

    #[test]
    fn test_step_applies_gravity() {
        let mut world = PhysicsWorld::new();
        world.set_gravity(Vec3::new(0.0, -10.0, 0.0));
        
        let mut body = RigidBody::new(CollisionShape::sphere(1.0), 1.0, BodyType::Dynamic);
        body.position = Vec3::new(0.0, 10.0, 0.0);
        let handle = world.add_body(body);

        world.step(1.0);

        let body = world.get_body(handle).unwrap();
        // Should have fallen due to gravity
        assert!(body.position.y < 10.0);
        assert!(body.linear_velocity.y < 0.0);
    }

    #[test]
    fn test_static_body_not_moved() {
        let mut world = PhysicsWorld::new();
        world.set_gravity(Vec3::new(0.0, -10.0, 0.0));
        
        let mut body = RigidBody::new(CollisionShape::sphere(1.0), 1.0, BodyType::Static);
        body.position = Vec3::new(0.0, 10.0, 0.0);
        let handle = world.add_body(body);

        world.step(1.0);

        let body = world.get_body(handle).unwrap();
        assert_eq!(body.position.y, 10.0);
    }

    #[test]
    fn test_get_body_mut() {
        let mut world = PhysicsWorld::new();
        let body = RigidBody::new(CollisionShape::sphere(1.0), 1.0, BodyType::Dynamic);
        let handle = world.add_body(body);
        
        if let Some(body) = world.get_body_mut(handle) {
            body.position = Vec3::new(5.0, 0.0, 0.0);
        }
        
        let body = world.get_body(handle).unwrap();
        assert_eq!(body.position.x, 5.0);
    }

    #[test]
    fn test_bodies_iter() {
        let mut world = PhysicsWorld::new();
        world.add_body(RigidBody::new(CollisionShape::sphere(1.0), 1.0, BodyType::Dynamic));
        world.add_body(RigidBody::new(CollisionShape::sphere(2.0), 2.0, BodyType::Dynamic));
        
        let count = world.bodies().count();
        assert_eq!(count, 2);
    }
}
