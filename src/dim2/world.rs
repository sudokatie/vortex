//! 2D physics world.

use glam::Vec2;
use slotmap::{new_key_type, SlotMap};

use super::{
    BodyType2D, RigidBody2D, CollisionShape2D, Aabb2D,
    constraints::{
        SolverBody2D, SolverConfig2D, ConstraintSolver2D,
        ContactConstraint2D, DistanceJoint2D, RevoluteJoint2D,
    },
    collision::{gjk_2d, Contact2D},
};

new_key_type! {
    /// Handle to a 2D rigid body.
    pub struct BodyHandle2D;
}

/// 2D physics world.
pub struct PhysicsWorld2D {
    bodies: SlotMap<BodyHandle2D, RigidBody2D>,
    gravity: Vec2,
    distance_joints: Vec<DistanceJoint2D>,
    revolute_joints: Vec<RevoluteJoint2D>,
    solver: ConstraintSolver2D,
    contacts: Vec<ContactConstraint2D>,
}

impl Default for PhysicsWorld2D {
    fn default() -> Self {
        Self::new()
    }
}

impl PhysicsWorld2D {
    /// Create a new empty 2D physics world.
    pub fn new() -> Self {
        Self {
            bodies: SlotMap::with_key(),
            gravity: Vec2::new(0.0, -9.81),
            distance_joints: Vec::new(),
            revolute_joints: Vec::new(),
            solver: ConstraintSolver2D::new(SolverConfig2D::default()),
            contacts: Vec::new(),
        }
    }
    
    /// Set gravity.
    pub fn set_gravity(&mut self, gravity: Vec2) {
        self.gravity = gravity;
    }
    
    /// Get gravity.
    pub fn gravity(&self) -> Vec2 {
        self.gravity
    }
    
    /// Add a rigid body.
    pub fn add_body(&mut self, body: RigidBody2D) -> BodyHandle2D {
        self.bodies.insert(body)
    }
    
    /// Remove a rigid body.
    pub fn remove_body(&mut self, handle: BodyHandle2D) -> Option<RigidBody2D> {
        self.bodies.remove(handle)
    }
    
    /// Get a body by handle.
    pub fn get_body(&self, handle: BodyHandle2D) -> Option<&RigidBody2D> {
        self.bodies.get(handle)
    }
    
    /// Get a mutable body by handle.
    pub fn get_body_mut(&mut self, handle: BodyHandle2D) -> Option<&mut RigidBody2D> {
        self.bodies.get_mut(handle)
    }
    
    /// Iterate over all bodies.
    pub fn bodies(&self) -> impl Iterator<Item = (BodyHandle2D, &RigidBody2D)> {
        self.bodies.iter()
    }
    
    /// Add a distance joint.
    pub fn add_distance_joint(&mut self, joint: DistanceJoint2D) {
        self.distance_joints.push(joint);
    }
    
    /// Add a revolute joint.
    pub fn add_revolute_joint(&mut self, joint: RevoluteJoint2D) {
        self.revolute_joints.push(joint);
    }
    
    /// Step the simulation.
    pub fn step(&mut self, dt: f32) {
        // 1. Apply forces (gravity)
        self.apply_forces(dt);
        
        // 2. Broad phase collision detection
        let pairs = self.broad_phase();
        
        // 3. Narrow phase and contact generation
        self.contacts.clear();
        let handle_to_idx = self.build_handle_map();
        
        for (h_a, h_b) in pairs {
            if let Some(contact) = self.narrow_phase(h_a, h_b, &handle_to_idx) {
                self.contacts.push(contact);
            }
        }
        
        // 4. Build solver bodies
        let mut solver_bodies: Vec<SolverBody2D> = self.bodies
            .values()
            .map(SolverBody2D::from_body)
            .collect();
        
        // 5. Solve constraints
        self.solver.solve(
            &mut self.contacts,
            &mut self.distance_joints,
            &mut self.revolute_joints,
            &mut solver_bodies,
            dt,
        );
        
        // 6. Write back velocities
        for (i, (_, body)) in self.bodies.iter_mut().enumerate() {
            body.velocity = solver_bodies[i].velocity;
            body.angular_velocity = solver_bodies[i].angular_velocity;
        }
        
        // 7. Integrate positions
        self.integrate(dt);
    }
    
    fn apply_forces(&mut self, _dt: f32) {
        for (_, body) in self.bodies.iter_mut() {
            if body.body_type != BodyType2D::Dynamic {
                continue;
            }
            body.apply_force(self.gravity * body.mass);
        }
    }
    
    fn build_handle_map(&self) -> std::collections::HashMap<BodyHandle2D, usize> {
        self.bodies.iter()
            .enumerate()
            .map(|(i, (h, _))| (h, i))
            .collect()
    }
    
    fn broad_phase(&self) -> Vec<(BodyHandle2D, BodyHandle2D)> {
        let mut pairs = Vec::new();
        let handles: Vec<_> = self.bodies.keys().collect();
        
        // Simple O(n²) for now - can be optimized with SAP or grid
        for i in 0..handles.len() {
            for j in (i + 1)..handles.len() {
                let h_a = handles[i];
                let h_b = handles[j];
                
                let body_a = &self.bodies[h_a];
                let body_b = &self.bodies[h_b];
                
                // Skip if both static
                if body_a.body_type == BodyType2D::Static && body_b.body_type == BodyType2D::Static {
                    continue;
                }
                
                // AABB test
                let aabb_a = body_a.shape.aabb(body_a.position, body_a.rotation);
                let aabb_b = body_b.shape.aabb(body_b.position, body_b.rotation);
                
                if aabb_a.intersects(&aabb_b) {
                    pairs.push((h_a, h_b));
                }
            }
        }
        
        pairs
    }
    
    fn narrow_phase(
        &self,
        h_a: BodyHandle2D,
        h_b: BodyHandle2D,
        handle_to_idx: &std::collections::HashMap<BodyHandle2D, usize>,
    ) -> Option<ContactConstraint2D> {
        let body_a = self.bodies.get(h_a)?;
        let body_b = self.bodies.get(h_b)?;
        
        let contact = gjk_2d(
            &body_a.shape, body_a.position, body_a.rotation,
            &body_b.shape, body_b.position, body_b.rotation,
        )?;
        
        let idx_a = *handle_to_idx.get(&h_a)?;
        let idx_b = *handle_to_idx.get(&h_b)?;
        
        // Contact point relative to body centers
        let r_a = contact.point() - body_a.position;
        let r_b = contact.point() - body_b.position;
        
        Some(ContactConstraint2D::new(
            idx_a,
            idx_b,
            contact.normal,
            r_a,
            r_b,
            contact.depth,
            body_a.inv_mass,
            body_b.inv_mass,
            body_a.inv_inertia,
            body_b.inv_inertia,
            (body_a.friction * body_b.friction).sqrt(),
            (body_a.restitution * body_b.restitution).sqrt().max(body_a.restitution.min(body_b.restitution)),
        ))
    }
    
    fn integrate(&mut self, dt: f32) {
        for (_, body) in self.bodies.iter_mut() {
            if body.body_type != BodyType2D::Dynamic {
                body.clear_forces();
                continue;
            }
            
            // Integrate velocity
            let acceleration = body.force * body.inv_mass;
            body.velocity += acceleration * dt;
            body.angular_velocity += body.torque * body.inv_inertia * dt;
            
            // Damping
            body.velocity *= 0.99_f32.powf(dt * 60.0);
            body.angular_velocity *= 0.99_f32.powf(dt * 60.0);
            
            // Integrate position
            body.position += body.velocity * dt;
            body.rotation += body.angular_velocity * dt;
            
            body.clear_forces();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_world_new() {
        let world = PhysicsWorld2D::new();
        assert_eq!(world.gravity(), Vec2::new(0.0, -9.81));
    }
    
    #[test]
    fn test_add_body() {
        let mut world = PhysicsWorld2D::new();
        let body = RigidBody2D::new(CollisionShape2D::circle(1.0), 1.0, BodyType2D::Dynamic);
        let handle = world.add_body(body);
        
        assert!(world.get_body(handle).is_some());
    }
    
    #[test]
    fn test_step_applies_gravity() {
        let mut world = PhysicsWorld2D::new();
        world.set_gravity(Vec2::new(0.0, -10.0));
        
        let body = RigidBody2D::new(CollisionShape2D::circle(0.5), 1.0, BodyType2D::Dynamic);
        let handle = world.add_body(body);
        
        world.step(1.0 / 60.0);
        
        let body = world.get_body(handle).unwrap();
        assert!(body.velocity.y < 0.0);
    }
    
    #[test]
    fn test_static_body_not_moved() {
        let mut world = PhysicsWorld2D::new();
        world.set_gravity(Vec2::new(0.0, -10.0));
        
        let body = RigidBody2D::new(CollisionShape2D::circle(0.5), 1.0, BodyType2D::Static);
        let handle = world.add_body(body);
        
        world.step(1.0 / 60.0);
        
        let body = world.get_body(handle).unwrap();
        assert_eq!(body.position, Vec2::ZERO);
    }
    
    #[test]
    fn test_collision_detection() {
        let mut world = PhysicsWorld2D::new();
        world.set_gravity(Vec2::ZERO);
        
        // Two overlapping circles
        let mut body_a = RigidBody2D::new(CollisionShape2D::circle(1.0), 1.0, BodyType2D::Dynamic);
        body_a.position = Vec2::ZERO;
        
        let mut body_b = RigidBody2D::new(CollisionShape2D::circle(1.0), 1.0, BodyType2D::Dynamic);
        body_b.position = Vec2::new(1.5, 0.0); // Overlapping by 0.5
        
        let _h_a = world.add_body(body_a);
        let _h_b = world.add_body(body_b);
        
        // Step should detect collision and separate
        for _ in 0..100 {
            world.step(1.0 / 60.0);
        }
        
        // Bodies should be separated
        let positions: Vec<_> = world.bodies().map(|(_, b)| b.position).collect();
        let dist = (positions[0] - positions[1]).length();
        assert!(dist > 1.9); // Should be ~2.0 (sum of radii)
    }
}
