//! Physics world container and simulation stepping.

use glam::Vec3;
use slotmap::{new_key_type, SlotMap, Key};
use std::collections::HashMap;

use crate::collision::{
    Aabb, BroadPhase, ContactManifold, SweepAndPrune,
    gjk_intersection, epa, generate_contacts,
};
use crate::constraints::{
    ContactConstraint, ConstraintSolver, Joint, SolverBody, SolverConfig,
};
use crate::dynamics::RigidBody;
use crate::dynamics::material::{combine_friction, combine_restitution};
use crate::world::island::{IslandDetector, Island, ContactPair, JointPair};
use crate::world::step::{StepConfig, StepResult};

new_key_type! {
    /// Handle to a rigid body in the physics world.
    pub struct BodyHandle;
}

/// The physics world containing all bodies and constraints.
pub struct PhysicsWorld {
    bodies: SlotMap<BodyHandle, RigidBody>,
    joints: Vec<Box<dyn Joint>>,
    gravity: Vec3,
    broadphase: SweepAndPrune,
    solver: ConstraintSolver,
    island_detector: IslandDetector,
    /// Config for simulation stepping
    config: StepConfig,
    /// Sleeping state: time each body has been at rest
    sleep_timers: HashMap<BodyHandle, f32>,
    /// Cached manifolds from last step
    contact_manifolds: Vec<ContactManifold>,
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
            joints: Vec::new(),
            gravity: Vec3::new(0.0, -9.81, 0.0),
            broadphase: SweepAndPrune::new(),
            solver: ConstraintSolver::new(SolverConfig::default()),
            island_detector: IslandDetector::new(),
            config: StepConfig::default(),
            sleep_timers: HashMap::new(),
            contact_manifolds: Vec::new(),
        }
    }
    
    /// Create with custom config.
    pub fn with_config(config: StepConfig) -> Self {
        let mut world = Self::new();
        world.config = config.clone();
        world.gravity = config.gravity;
        world
    }

    /// Set the gravity vector.
    pub fn set_gravity(&mut self, gravity: Vec3) {
        self.gravity = gravity;
        self.config.gravity = gravity;
    }

    /// Get the gravity vector.
    pub fn gravity(&self) -> Vec3 {
        self.gravity
    }

    /// Add a rigid body to the world.
    pub fn add_body(&mut self, body: RigidBody) -> BodyHandle {
        let aabb = body.shape.local_aabb();
        let world_aabb = Aabb::new(
            aabb.min + body.position,
            aabb.max + body.position,
        );
        let handle = self.bodies.insert(body);
        self.broadphase.insert(handle, world_aabb);
        self.sleep_timers.insert(handle, 0.0);
        handle
    }

    /// Remove a rigid body from the world.
    pub fn remove_body(&mut self, handle: BodyHandle) -> Option<RigidBody> {
        self.broadphase.remove(handle);
        self.sleep_timers.remove(&handle);
        // Remove joints connected to this body
        self.joints.retain(|j| {
            let (a, b) = j.bodies();
            a != handle && b != handle
        });
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
    
    /// Add a joint to the world.
    pub fn add_joint(&mut self, joint: Box<dyn Joint>) {
        self.joints.push(joint);
    }
    
    /// Remove all joints.
    pub fn clear_joints(&mut self) {
        self.joints.clear();
    }
    
    /// Get number of joints.
    pub fn joint_count(&self) -> usize {
        self.joints.len()
    }
    
    /// Get contact manifolds from last step.
    pub fn contacts(&self) -> &[ContactManifold] {
        &self.contact_manifolds
    }
    
    /// Get solver config.
    pub fn solver_config(&self) -> &SolverConfig {
        self.solver.config()
    }
    
    /// Set solver iterations.
    pub fn set_solver_iterations(&mut self, velocity: usize, position: usize) {
        self.solver.set_velocity_iterations(velocity);
        self.solver.set_position_iterations(position);
    }

    /// Step the simulation forward by dt seconds.
    pub fn step(&mut self, dt: f32) {
        let _result = self.step_full(dt);
    }
    
    /// Step with full result info.
    pub fn step_full(&mut self, dt: f32) -> StepResult {
        let mut result = StepResult::new();
        result.substeps = 1;
        
        // 1. Apply forces (gravity, etc.)
        self.apply_forces(dt);
        
        // 2. Update broadphase AABBs
        self.update_broadphase();
        
        // 3. Broad phase collision detection
        let pairs = self.broadphase.query_pairs();
        
        // 4. Narrow phase collision detection (GJK/EPA)
        self.contact_manifolds.clear();
        for (handle_a, handle_b) in &pairs {
            if let Some(manifold) = self.narrow_phase(*handle_a, *handle_b) {
                self.contact_manifolds.push(manifold);
            }
        }
        result.contact_count = self.contact_manifolds.len();
        
        // 5. Build islands for optimized solving and sleeping
        let islands = self.build_islands();
        result.island_count = islands.len();
        
        // 6. Solve constraints per island (allows parallel solving, better sleeping)
        self.solve_constraints_with_islands(&islands, dt);
        
        // 7. Integrate velocities and positions
        self.integrate(dt);
        
        // 8. Update sleeping (per-island)
        if self.config.allow_sleeping {
            self.update_sleeping_with_islands(&islands, dt, &mut result);
        }
        
        result
    }
    
    /// Apply forces (gravity) to all dynamic bodies.
    fn apply_forces(&mut self, _dt: f32) {
        for (_, body) in self.bodies.iter_mut() {
            if !body.is_dynamic() || body.is_sleeping {
                continue;
            }
            // Apply gravity
            body.apply_force(self.gravity * body.mass);
        }
    }
    
    /// Update broadphase AABBs for all bodies.
    fn update_broadphase(&mut self) {
        for (handle, body) in self.bodies.iter() {
            let local_aabb = body.shape.local_aabb();
            let transform = body.transform();
            let world_aabb = local_aabb.transform(&transform);
            self.broadphase.update(handle, world_aabb);
        }
    }
    
    /// Narrow phase collision detection between two bodies.
    fn narrow_phase(&self, handle_a: BodyHandle, handle_b: BodyHandle) -> Option<ContactManifold> {
        let body_a = self.bodies.get(handle_a)?;
        let body_b = self.bodies.get(handle_b)?;
        
        // Skip if both static/kinematic
        if !body_a.is_dynamic() && !body_b.is_dynamic() {
            return None;
        }
        
        // Skip if both sleeping
        if body_a.is_sleeping && body_b.is_sleeping {
            return None;
        }
        
        let transform_a = body_a.transform();
        let transform_b = body_b.transform();
        
        // GJK intersection test
        let simplex = gjk_intersection(&body_a.shape, &transform_a, &body_b.shape, &transform_b)?;
        
        // EPA for penetration info
        let pen_info = epa(simplex, &body_a.shape, &transform_a, &body_b.shape, &transform_b)?;
        
        // Generate contact manifold
        let manifold = generate_contacts(
            handle_a, handle_b,
            pen_info.normal, pen_info.depth,
            pen_info.point_a, pen_info.point_b,
            body_a.position, body_b.position,
        );
        
        Some(manifold)
    }
    
    /// Solve all constraints (legacy non-island version).
    #[allow(dead_code)]
    fn solve_constraints(&mut self, _dt: f32) {
        if self.contact_manifolds.is_empty() && self.joints.is_empty() {
            return;
        }
        
        // Build handle-to-index map and solver bodies array
        let mut handle_to_index: HashMap<BodyHandle, usize> = HashMap::new();
        let mut solver_bodies: Vec<SolverBody> = Vec::new();
        
        for (handle, body) in self.bodies.iter() {
            handle_to_index.insert(handle, solver_bodies.len());
            solver_bodies.push(SolverBody::new(
                body.position,
                body.rotation,
                body.linear_velocity,
                body.angular_velocity,
                body.inv_mass,
                Vec3::new(
                    body.inv_inertia.x_axis.x,
                    body.inv_inertia.y_axis.y,
                    body.inv_inertia.z_axis.z,
                ),
            ));
        }
        
        // Build contact constraints
        let mut contact_constraints: Vec<ContactConstraint> = Vec::new();
        for manifold in &self.contact_manifolds {
            let idx_a = handle_to_index[&manifold.body_a];
            let idx_b = handle_to_index[&manifold.body_b];
            
            let body_a = &self.bodies[manifold.body_a];
            let body_b = &self.bodies[manifold.body_b];
            
            let friction = combine_friction(body_a.friction, body_b.friction);
            let restitution = combine_restitution(body_a.restitution, body_b.restitution);
            
            let constraint = ContactConstraint::from_manifold_with_indices(
                manifold,
                idx_a, idx_b,
                body_a.position, body_b.position,
                body_a.inv_mass, body_b.inv_mass,
                Vec3::new(body_a.inv_inertia.x_axis.x, body_a.inv_inertia.y_axis.y, body_a.inv_inertia.z_axis.z),
                Vec3::new(body_b.inv_inertia.x_axis.x, body_b.inv_inertia.y_axis.y, body_b.inv_inertia.z_axis.z),
                friction, restitution,
            );
            contact_constraints.push(constraint);
        }
        
        // Solve contacts
        self.solver.solve_contacts(&mut contact_constraints, &mut solver_bodies);
        
        // Solve joints (if any)
        if !self.joints.is_empty() {
            let map = handle_to_index.clone();
            self.solver.solve_joints(&mut self.joints, &mut solver_bodies, |h| map[&h]);
        }
        
        // Write back solver results to bodies
        for (handle, body) in self.bodies.iter_mut() {
            if let Some(&idx) = handle_to_index.get(&handle) {
                let solver_body = &solver_bodies[idx];
                body.linear_velocity = solver_body.velocity;
                body.angular_velocity = solver_body.angular_velocity;
                body.position = solver_body.position;
                body.rotation = solver_body.rotation;
            }
        }
    }
    
    /// Integrate velocities and positions.
    fn integrate(&mut self, dt: f32) {
        for (_, body) in self.bodies.iter_mut() {
            if !body.is_dynamic() || body.is_sleeping {
                body.clear_forces();
                continue;
            }
            
            // Integrate velocity
            let acceleration = body.force * body.inv_mass;
            body.linear_velocity += acceleration * dt;
            
            let angular_acceleration = body.world_inv_inertia() * body.torque;
            body.angular_velocity += angular_acceleration * dt;
            
            // Apply damping
            body.linear_velocity *= (1.0 - 0.01_f32).powf(dt);
            body.angular_velocity *= (1.0 - 0.01_f32).powf(dt);
            
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
    
    /// Build islands from contacts and joints.
    fn build_islands(&mut self) -> Vec<Island> {
        // Build contact pairs
        let _contact_pairs: Vec<ContactPair> = self.contact_manifolds.iter()
            .map(|m| ContactPair {
                body_a: m.body_a.data().as_ffi() as u32,
                body_b: m.body_b.data().as_ffi() as u32,
            })
            .collect();
        
        // Build joint pairs
        let _joint_pairs: Vec<JointPair> = self.joints.iter()
            .map(|j| {
                let (a, b) = j.bodies();
                JointPair {
                    body_a: a.data().as_ffi() as u32,
                    body_b: b.data().as_ffi() as u32,
                }
            })
            .collect();
        
        // Build handle-to-index mapping
        let handle_to_idx: HashMap<BodyHandle, u32> = self.bodies.iter()
            .enumerate()
            .map(|(i, (h, _))| (h, i as u32))
            .collect();
        
        // Convert manifold body handles to indices
        let contact_pairs_indexed: Vec<ContactPair> = self.contact_manifolds.iter()
            .filter_map(|m| {
                Some(ContactPair {
                    body_a: *handle_to_idx.get(&m.body_a)?,
                    body_b: *handle_to_idx.get(&m.body_b)?,
                })
            })
            .collect();
        
        let joint_pairs_indexed: Vec<JointPair> = self.joints.iter()
            .filter_map(|j| {
                let (a, b) = j.bodies();
                Some(JointPair {
                    body_a: *handle_to_idx.get(&a)?,
                    body_b: *handle_to_idx.get(&b)?,
                })
            })
            .collect();
        
        // Get is_static closure
        let bodies_vec: Vec<_> = self.bodies.iter().map(|(_, b)| b.is_static()).collect();
        
        self.island_detector.find_islands(
            self.bodies.len(),
            |idx| bodies_vec.get(idx as usize).copied().unwrap_or(true),
            &contact_pairs_indexed,
            &joint_pairs_indexed,
        )
    }
    
    /// Solve constraints with island optimization.
    fn solve_constraints_with_islands(&mut self, islands: &[Island], dt: f32) {
        // For now, still solve all together but mark islands for future parallel solving
        // Full island-based solving would process each island independently
        
        if self.contact_manifolds.is_empty() && self.joints.is_empty() {
            return;
        }
        
        // Build handle-to-index map and solver bodies array
        let mut handle_to_index: HashMap<BodyHandle, usize> = HashMap::new();
        let mut solver_bodies: Vec<SolverBody> = Vec::new();
        
        for (handle, body) in self.bodies.iter() {
            handle_to_index.insert(handle, solver_bodies.len());
            solver_bodies.push(SolverBody::new(
                body.position,
                body.rotation,
                body.linear_velocity,
                body.angular_velocity,
                body.inv_mass,
                Vec3::new(
                    body.inv_inertia.x_axis.x,
                    body.inv_inertia.y_axis.y,
                    body.inv_inertia.z_axis.z,
                ),
            ));
        }
        
        // Build contact constraints
        let mut contact_constraints: Vec<ContactConstraint> = Vec::new();
        for manifold in &self.contact_manifolds {
            let idx_a = handle_to_index[&manifold.body_a];
            let idx_b = handle_to_index[&manifold.body_b];
            
            let body_a = &self.bodies[manifold.body_a];
            let body_b = &self.bodies[manifold.body_b];
            
            let friction = combine_friction(body_a.friction, body_b.friction);
            let restitution = combine_restitution(body_a.restitution, body_b.restitution);
            
            let constraint = ContactConstraint::from_manifold_with_indices(
                manifold,
                idx_a, idx_b,
                body_a.position, body_b.position,
                body_a.inv_mass, body_b.inv_mass,
                Vec3::new(body_a.inv_inertia.x_axis.x, body_a.inv_inertia.y_axis.y, body_a.inv_inertia.z_axis.z),
                Vec3::new(body_b.inv_inertia.x_axis.x, body_b.inv_inertia.y_axis.y, body_b.inv_inertia.z_axis.z),
                friction, restitution,
            );
            contact_constraints.push(constraint);
        }
        
        // Solve contacts
        self.solver.solve_contacts(&mut contact_constraints, &mut solver_bodies);
        
        // Solve joints (if any)
        if !self.joints.is_empty() {
            let map = handle_to_index.clone();
            self.solver.solve_joints(&mut self.joints, &mut solver_bodies, |h| map[&h]);
        }
        
        // Write back solver results to bodies
        for (handle, body) in self.bodies.iter_mut() {
            if let Some(&idx) = handle_to_index.get(&handle) {
                let solver_body = &solver_bodies[idx];
                body.linear_velocity = solver_body.velocity;
                body.angular_velocity = solver_body.angular_velocity;
                body.position = solver_body.position;
                body.rotation = solver_body.rotation;
            }
        }
        
        // Mark island count
        let _ = islands; // Used for future parallel solving
        let _ = dt;
    }
    
    /// Update sleeping with island optimization.
    /// All bodies in an island sleep together if the island can sleep.
    fn update_sleeping_with_islands(&mut self, islands: &[Island], dt: f32, result: &mut StepResult) {
        let threshold_sq = self.config.sleep_threshold * self.config.sleep_threshold;
        
        // Build index-to-handle map
        let idx_to_handle: Vec<BodyHandle> = self.bodies.keys().collect();
        
        for island in islands {
            // Check if entire island can sleep
            let mut island_can_sleep = true;
            
            for &body_idx in &island.bodies {
                if let Some(&handle) = idx_to_handle.get(body_idx as usize) {
                    if let Some(body) = self.bodies.get(handle) {
                        let energy = body.linear_velocity.length_squared() 
                            + body.angular_velocity.length_squared();
                        if energy >= threshold_sq {
                            island_can_sleep = false;
                            break;
                        }
                    }
                }
            }
            
            // Update sleep timers for all bodies in island
            for &body_idx in &island.bodies {
                if let Some(&handle) = idx_to_handle.get(body_idx as usize) {
                    if island_can_sleep {
                        let timer = self.sleep_timers.entry(handle).or_insert(0.0);
                        *timer += dt;
                        
                        if *timer >= self.config.time_to_sleep {
                            if let Some(body) = self.bodies.get_mut(handle) {
                                if !body.is_sleeping {
                                    body.set_sleeping(true);
                                    result.newly_sleeping.push(handle.data().as_ffi() as u32);
                                }
                            }
                        }
                    } else {
                        // Wake entire island
                        self.sleep_timers.insert(handle, 0.0);
                        if let Some(body) = self.bodies.get_mut(handle) {
                            if body.is_sleeping {
                                body.set_sleeping(false);
                                result.newly_awake.push(handle.data().as_ffi() as u32);
                            }
                        }
                    }
                }
            }
        }
    }
    
    /// Update sleeping state (legacy non-island version).
    #[allow(dead_code)]
    fn update_sleeping(&mut self, dt: f32, result: &mut StepResult) {
        let threshold_sq = self.config.sleep_threshold * self.config.sleep_threshold;
        
        for (handle, body) in self.bodies.iter_mut() {
            if !body.is_dynamic() {
                continue;
            }
            
            let energy = body.linear_velocity.length_squared() + body.angular_velocity.length_squared();
            
            if energy < threshold_sq {
                // Accumulate sleep time
                let timer = self.sleep_timers.entry(handle).or_insert(0.0);
                *timer += dt;
                
                if *timer >= self.config.time_to_sleep && !body.is_sleeping {
                    body.set_sleeping(true);
                    result.newly_sleeping.push(handle.data().as_ffi() as u32);
                }
            } else {
                // Reset sleep timer and wake up
                self.sleep_timers.insert(handle, 0.0);
                if body.is_sleeping {
                    body.set_sleeping(false);
                    result.newly_awake.push(handle.data().as_ffi() as u32);
                }
            }
        }
    }
    
    /// Wake a body and its neighbors.
    pub fn wake_body(&mut self, handle: BodyHandle) {
        if let Some(body) = self.bodies.get_mut(handle) {
            body.set_sleeping(false);
            self.sleep_timers.insert(handle, 0.0);
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
    
    #[test]
    fn test_collision_detection() {
        let mut world = PhysicsWorld::new();
        world.set_gravity(Vec3::ZERO); // No gravity for this test
        
        // Two overlapping spheres
        let mut body1 = RigidBody::new(CollisionShape::sphere(1.0), 1.0, BodyType::Dynamic);
        body1.position = Vec3::ZERO;
        world.add_body(body1);
        
        let mut body2 = RigidBody::new(CollisionShape::sphere(1.0), 1.0, BodyType::Dynamic);
        body2.position = Vec3::new(1.5, 0.0, 0.0); // Overlapping
        world.add_body(body2);
        
        world.step(1.0 / 60.0);
        
        // Should have detected collision
        assert!(!world.contacts().is_empty());
    }
    
    #[test]
    fn test_sleeping() {
        let mut world = PhysicsWorld::new();
        world.set_gravity(Vec3::ZERO);
        world.config.allow_sleeping = true;
        world.config.time_to_sleep = 0.1;
        world.config.sleep_threshold = 0.1;
        
        let body = RigidBody::new(CollisionShape::sphere(1.0), 1.0, BodyType::Dynamic);
        let handle = world.add_body(body);
        
        // Step multiple times
        for _ in 0..20 {
            world.step(0.016);
        }
        
        // Body should be sleeping after being stationary
        assert!(world.get_body(handle).unwrap().is_sleeping);
    }
}
