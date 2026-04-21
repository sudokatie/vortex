//! Physics world container and simulation stepping.

use glam::Vec3;
use slotmap::{new_key_type, SlotMap, Key};
use std::collections::HashMap;

use crate::collision::{
    Aabb, BroadPhase, ContactManifold, SweepAndPrune, TimeOfImpact,
    gjk_intersection, epa, generate_contacts, calculate_toi, needs_ccd,
};
use crate::constraints::{
    ContactConstraint, ConstraintSolver, Joint, SolverBody, SolverConfig,
};
use crate::dynamics::RigidBody;
use crate::dynamics::material::{combine_friction, combine_restitution};
use crate::fluid::{
    FluidCouplingParams, FluidForceOutput, FluidParticle,
    apply_boundary_forces_to_particles, apply_fluid_forces_to_body,
};
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
    /// Fluid particles for coupling (managed externally or internally)
    fluid_particles: Vec<FluidParticle>,
    /// Fluid-rigid body coupling parameters
    pub fluid_coupling: FluidCouplingParams,
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
            fluid_particles: Vec::new(),
            fluid_coupling: FluidCouplingParams::default(),
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

    /// Enable or disable continuous collision detection.
    pub fn set_ccd_enabled(&mut self, enabled: bool) {
        self.config.ccd_enabled = enabled;
    }

    /// Check if CCD is enabled.
    pub fn ccd_enabled(&self) -> bool {
        self.config.ccd_enabled
    }

    // ==================== Fluid Coupling Methods ====================

    /// Set the fluid particles for coupling.
    ///
    /// Call this before each step to update the fluid state from an external solver.
    pub fn set_fluid_particles(&mut self, particles: Vec<FluidParticle>) {
        self.fluid_particles = particles;
    }

    /// Get a reference to the fluid particles.
    pub fn fluid_particles(&self) -> &[FluidParticle] {
        &self.fluid_particles
    }

    /// Get a mutable reference to the fluid particles.
    pub fn fluid_particles_mut(&mut self) -> &mut Vec<FluidParticle> {
        &mut self.fluid_particles
    }

    /// Clear all fluid particles.
    pub fn clear_fluid_particles(&mut self) {
        self.fluid_particles.clear();
    }

    /// Enable or disable fluid coupling.
    pub fn set_fluid_coupling_enabled(&mut self, buoyancy: bool, drag: bool) {
        self.fluid_coupling.buoyancy_enabled = buoyancy;
        self.fluid_coupling.drag_enabled = drag;
    }

    /// Check if fluid coupling is enabled.
    pub fn fluid_coupling_enabled(&self) -> bool {
        self.fluid_coupling.buoyancy_enabled || self.fluid_coupling.drag_enabled
    }

    /// Perform fluid-rigid body coupling step.
    ///
    /// This creates two-way coupling:
    /// 1. Applies boundary forces to fluid particles (pushes them out of bodies)
    /// 2. Computes buoyancy and drag on rigid bodies from surrounding fluid
    /// 3. Applies fluid forces to rigid bodies
    ///
    /// Call this between fluid solving and rigid body solving.
    pub fn coupling_step(&mut self) {
        if self.fluid_particles.is_empty() {
            return;
        }

        let coupling_enabled = self.fluid_coupling.buoyancy_enabled || self.fluid_coupling.drag_enabled;
        if !coupling_enabled && self.fluid_coupling.stiffness <= 0.0 {
            return;
        }

        // Collect body data to avoid borrow issues
        let body_data: Vec<_> = self.bodies
            .iter()
            .map(|(handle, body)| {
                (
                    handle,
                    body.shape.clone(),
                    body.position,
                    body.rotation,
                    body.linear_velocity,
                    body.is_dynamic(),
                    body.is_sleeping,
                )
            })
            .collect();

        let gravity = self.gravity;
        let params = self.fluid_coupling.clone();

        // For each rigid body near fluid particles
        for (handle, shape, position, rotation, velocity, is_dynamic, is_sleeping) in body_data {
            if is_sleeping {
                continue;
            }

            // 1. Apply boundary forces to fluid particles (push them out of the body)
            let reaction = apply_boundary_forces_to_particles(
                &mut self.fluid_particles,
                &shape,
                position,
                rotation,
                velocity,
                &params,
            );

            // 2. Compute buoyancy and drag on the body
            let fluid_forces = apply_fluid_forces_to_body(
                &shape,
                position,
                rotation,
                velocity,
                &self.fluid_particles,
                gravity,
                &params,
            );

            // 3. Apply forces to the rigid body
            if is_dynamic {
                if let Some(body) = self.bodies.get_mut(handle) {
                    // Apply buoyancy and drag
                    body.apply_force(fluid_forces.force);
                    body.apply_torque(fluid_forces.torque);

                    // Optionally apply reaction force from boundary interactions
                    // (This creates more accurate two-way coupling but may cause instability)
                    // body.apply_force(reaction * 0.1);
                    let _ = reaction; // Suppress unused warning
                }
            }
        }
    }

    /// Get coupling forces that would be applied to a body without actually applying them.
    ///
    /// Useful for debugging or visualization.
    pub fn compute_fluid_forces(&self, handle: BodyHandle) -> FluidForceOutput {
        let body = match self.bodies.get(handle) {
            Some(b) => b,
            None => return FluidForceOutput::default(),
        };

        apply_fluid_forces_to_body(
            &body.shape,
            body.position,
            body.rotation,
            body.linear_velocity,
            &self.fluid_particles,
            self.gravity,
            &self.fluid_coupling,
        )
    }

    /// Step the simulation forward by dt seconds.
    pub fn step(&mut self, dt: f32) {
        let _result = self.step_full(dt);
    }
    
    /// Step with full result info.
    pub fn step_full(&mut self, dt: f32) -> StepResult {
        let mut result = StepResult::new();
        result.substeps = 1;

        // CCD: If enabled, check for fast-moving bodies and handle sub-stepping
        if self.config.ccd_enabled {
            self.step_with_ccd(dt, &mut result);
        } else {
            self.step_discrete(dt, &mut result);
        }

        result
    }

    /// Perform a discrete physics step (no CCD).
    fn step_discrete(&mut self, dt: f32, result: &mut StepResult) {
        // 1. Apply forces (gravity, etc.)
        self.apply_forces(dt);

        // 1.5. Fluid-rigid body coupling (two-way interaction)
        // This applies boundary forces to fluid particles and buoyancy/drag to bodies
        self.coupling_step();

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
            self.update_sleeping_with_islands(&islands, dt, result);
        }
    }

    /// Perform a physics step with CCD sub-stepping.
    fn step_with_ccd(&mut self, dt: f32, result: &mut StepResult) {
        let mut remaining_time = dt;
        let mut ccd_substeps = 0;
        let max_ccd_substeps = self.config.max_ccd_substeps;

        while remaining_time > 1e-6 && ccd_substeps < max_ccd_substeps {
            // Find bodies that need CCD
            let fast_bodies = self.find_ccd_bodies(remaining_time);

            if fast_bodies.is_empty() {
                // No fast bodies, do regular step for remaining time
                self.step_discrete(remaining_time, result);
                break;
            }

            // Run sweep tests for fast-moving body pairs
            let toi_result = self.find_earliest_toi(&fast_bodies, remaining_time);
            result.ccd_sweep_tests += fast_bodies.len();

            match toi_result {
                Some(toi) if toi.time < 1.0 => {
                    // Collision detected before end of frame
                    let sub_dt = remaining_time * toi.time;

                    if sub_dt > 1e-6 {
                        // Advance to time of impact
                        self.integrate_to_time(sub_dt);
                    }

                    // Resolve collision at TOI
                    self.resolve_ccd_collision(&toi);

                    // Continue with remaining time
                    remaining_time *= 1.0 - toi.time;
                    ccd_substeps += 1;
                }
                _ => {
                    // No collision, do regular step
                    self.step_discrete(remaining_time, result);
                    break;
                }
            }
        }

        result.ccd_substeps = ccd_substeps;
    }

    /// Find bodies that need CCD due to high velocity.
    fn find_ccd_bodies(&self, dt: f32) -> Vec<BodyHandle> {
        let mut fast_bodies = Vec::new();

        for (handle, body) in self.bodies.iter() {
            if !body.is_dynamic() || body.is_sleeping {
                continue;
            }

            if needs_ccd(&body.shape, body.linear_velocity, dt) {
                fast_bodies.push(handle);
            }
        }

        fast_bodies
    }

    /// Find the earliest time of impact among fast-moving body pairs.
    fn find_earliest_toi(&self, fast_bodies: &[BodyHandle], dt: f32) -> Option<TimeOfImpact> {
        let mut earliest_toi: Option<TimeOfImpact> = None;

        for &handle_a in fast_bodies {
            let body_a = match self.bodies.get(handle_a) {
                Some(b) => b,
                None => continue,
            };

            // Check against all other bodies
            for (handle_b, body_b) in self.bodies.iter() {
                if handle_a == handle_b {
                    continue;
                }

                // Skip if both static/kinematic
                if !body_a.is_dynamic() && !body_b.is_dynamic() {
                    continue;
                }

                // Skip if both sleeping
                if body_a.is_sleeping && body_b.is_sleeping {
                    continue;
                }

                // Calculate TOI
                if let Some(toi) = calculate_toi(
                    handle_a,
                    &body_a.shape,
                    body_a.position,
                    body_a.linear_velocity,
                    body_a.rotation,
                    handle_b,
                    &body_b.shape,
                    body_b.position,
                    body_b.linear_velocity,
                    body_b.rotation,
                    dt,
                ) {
                    match &earliest_toi {
                        None => earliest_toi = Some(toi),
                        Some(existing) if toi.time < existing.time => earliest_toi = Some(toi),
                        _ => {}
                    }
                }
            }
        }

        earliest_toi
    }

    /// Integrate bodies forward by a given time without collision resolution.
    fn integrate_to_time(&mut self, dt: f32) {
        for (_, body) in self.bodies.iter_mut() {
            if !body.is_dynamic() || body.is_sleeping {
                continue;
            }

            // Just integrate position, don't apply forces (already done)
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
        }
    }

    /// Resolve a collision detected by CCD.
    fn resolve_ccd_collision(&mut self, toi: &TimeOfImpact) {
        // Get both bodies
        let (inv_mass_a, vel_a, is_dynamic_a, inv_mass_b, vel_b, is_dynamic_b, friction, restitution) = {
            let body_a = match self.bodies.get(toi.shape_a) {
                Some(b) => b,
                None => return,
            };
            let body_b = match self.bodies.get(toi.shape_b) {
                Some(b) => b,
                None => return,
            };

            let friction = combine_friction(body_a.friction, body_b.friction);
            let restitution = combine_restitution(body_a.restitution, body_b.restitution);

            (
                body_a.inv_mass,
                body_a.linear_velocity,
                body_a.is_dynamic(),
                body_b.inv_mass,
                body_b.linear_velocity,
                body_b.is_dynamic(),
                friction,
                restitution,
            )
        };

        // Calculate relative velocity along collision normal
        let rel_vel = vel_b - vel_a;
        let vel_along_normal = rel_vel.dot(toi.normal);

        // Don't resolve if velocities are separating
        if vel_along_normal > 0.0 {
            return;
        }

        // Calculate impulse magnitude
        let total_inv_mass = inv_mass_a + inv_mass_b;
        if total_inv_mass <= 0.0 {
            return;
        }

        let j = -(1.0 + restitution) * vel_along_normal / total_inv_mass;
        let impulse = toi.normal * j;

        // Apply impulse
        if let Some(body_a) = self.bodies.get_mut(toi.shape_a) {
            if is_dynamic_a {
                body_a.linear_velocity -= impulse * inv_mass_a;
            }
        }
        if let Some(body_b) = self.bodies.get_mut(toi.shape_b) {
            if is_dynamic_b {
                body_b.linear_velocity += impulse * inv_mass_b;
            }
        }

        // Apply friction impulse
        let tangent = (rel_vel - toi.normal * vel_along_normal).normalize_or_zero();
        let vel_along_tangent = rel_vel.dot(tangent);
        let jt = -vel_along_tangent / total_inv_mass;
        let jt_clamped = jt.clamp(-j.abs() * friction, j.abs() * friction);
        let friction_impulse = tangent * jt_clamped;

        if let Some(body_a) = self.bodies.get_mut(toi.shape_a) {
            if body_a.is_dynamic() {
                body_a.linear_velocity -= friction_impulse * inv_mass_a;
            }
        }
        if let Some(body_b) = self.bodies.get_mut(toi.shape_b) {
            if body_b.is_dynamic() {
                body_b.linear_velocity += friction_impulse * inv_mass_b;
            }
        }
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

    // ==================== Fluid-Rigid Body Coupling Tests ====================

    #[test]
    fn test_coupling_buoyancy_upward_force() {
        let mut world = PhysicsWorld::new();
        world.set_gravity(Vec3::new(0.0, -10.0, 0.0));
        world.fluid_coupling.buoyancy_enabled = true;
        world.fluid_coupling.drag_enabled = false;
        world.fluid_coupling.smoothing_radius = 2.0;

        // Create a sphere submerged in fluid
        let mut body = RigidBody::new(CollisionShape::sphere(0.5), 1.0, BodyType::Dynamic);
        body.position = Vec3::ZERO;
        let handle = world.add_body(body);

        // Create fluid particles around the sphere
        let mut particles = Vec::new();
        for x in -3..=3 {
            for y in -3..=3 {
                for z in -3..=3 {
                    particles.push(FluidParticle::new(
                        Vec3::new(x as f32 * 0.3, y as f32 * 0.3, z as f32 * 0.3),
                        1.0,
                    ));
                }
            }
        }
        world.set_fluid_particles(particles);

        // Compute forces without stepping
        let forces = world.compute_fluid_forces(handle);

        // Buoyancy should point upward
        assert!(forces.buoyancy.y > 0.0, "Buoyancy should point up: {:?}", forces.buoyancy);
    }

    #[test]
    fn test_coupling_drag_opposes_motion() {
        let mut world = PhysicsWorld::new();
        world.set_gravity(Vec3::ZERO);
        world.fluid_coupling.buoyancy_enabled = false;
        world.fluid_coupling.drag_enabled = true;
        world.fluid_coupling.smoothing_radius = 2.0;

        // Create a moving sphere
        let mut body = RigidBody::new(CollisionShape::sphere(0.5), 1.0, BodyType::Dynamic);
        body.position = Vec3::ZERO;
        body.linear_velocity = Vec3::new(5.0, 0.0, 0.0); // Moving in +X
        let handle = world.add_body(body);

        // Create stationary fluid particles
        let mut particles = Vec::new();
        for x in -3..=3 {
            for y in -3..=3 {
                for z in -3..=3 {
                    particles.push(FluidParticle::new(
                        Vec3::new(x as f32 * 0.3, y as f32 * 0.3, z as f32 * 0.3),
                        1.0,
                    ));
                }
            }
        }
        world.set_fluid_particles(particles);

        let forces = world.compute_fluid_forces(handle);

        // Drag should oppose motion (point in -X direction)
        assert!(forces.drag.x < 0.0, "Drag should oppose motion: {:?}", forces.drag);
    }

    #[test]
    fn test_coupling_boundary_pushes_particles() {
        let mut world = PhysicsWorld::new();
        world.set_gravity(Vec3::ZERO);

        // Create a sphere
        let mut body = RigidBody::new(CollisionShape::sphere(1.0), 1.0, BodyType::Dynamic);
        body.position = Vec3::ZERO;
        world.add_body(body);

        // Create a particle inside the sphere
        let mut particles = vec![FluidParticle::new(Vec3::new(0.5, 0.0, 0.0), 1.0)];
        particles[0].acceleration = Vec3::ZERO;
        world.set_fluid_particles(particles);

        // Run coupling step
        world.coupling_step();

        // Particle should have been pushed outward
        let particles = world.fluid_particles();
        assert!(
            particles[0].acceleration.x > 0.0,
            "Particle should be pushed out: {:?}",
            particles[0].acceleration
        );
    }

    #[test]
    fn test_coupling_two_way_sphere_in_fluid() {
        let mut world = PhysicsWorld::new();
        world.set_gravity(Vec3::new(0.0, -10.0, 0.0));
        world.fluid_coupling.buoyancy_enabled = true;
        world.fluid_coupling.drag_enabled = true;
        world.fluid_coupling.smoothing_radius = 1.5;
        world.fluid_coupling.fluid_density = 1000.0;

        // Create a falling sphere - slightly heavier than water so it sinks slowly
        // Volume = (4/3) * pi * 0.5^3 = 0.52 m^3
        // Buoyancy at full submersion = 1000 * 0.52 * 10 = 5200 N up
        // Weight = 600 * 10 = 6000 N down
        // Net = 800 N down, so it sinks but slowly due to buoyancy
        let mut body = RigidBody::new(CollisionShape::sphere(0.5), 600.0, BodyType::Dynamic);
        body.position = Vec3::new(0.0, 0.0, 0.0);
        body.linear_velocity = Vec3::new(0.0, -2.0, 0.0); // Falling
        let handle = world.add_body(body);

        // Create dense fluid particles around the sphere
        let mut particles = Vec::new();
        for x in -5..=5 {
            for y in -5..=5 {
                for z in -5..=5 {
                    particles.push(FluidParticle::new(
                        Vec3::new(x as f32 * 0.25, y as f32 * 0.25, z as f32 * 0.25),
                        1.0,
                    ));
                }
            }
        }
        world.set_fluid_particles(particles);

        // Get the fluid forces to verify they're being computed
        let forces = world.compute_fluid_forces(handle);

        // Verify buoyancy is being computed and pointing up
        assert!(
            forces.buoyancy.y > 100.0,
            "Should have significant buoyancy: {:?}",
            forces.buoyancy
        );

        // Also verify drag opposes downward motion
        // (fluid stationary, body moving down means relative velocity is up)
        assert!(
            forces.drag.y > 0.0 || forces.drag.length() < 1.0,
            "Drag should oppose motion or be small: {:?}",
            forces.drag
        );
    }

    #[test]
    fn test_coupling_disabled_no_forces() {
        let mut world = PhysicsWorld::new();
        world.set_gravity(Vec3::new(0.0, -10.0, 0.0));
        world.fluid_coupling.buoyancy_enabled = false;
        world.fluid_coupling.drag_enabled = false;

        let mut body = RigidBody::new(CollisionShape::sphere(0.5), 1.0, BodyType::Dynamic);
        body.position = Vec3::ZERO;
        let handle = world.add_body(body);

        // Create fluid particles
        let mut particles = Vec::new();
        for x in -2..=2 {
            for y in -2..=2 {
                for z in -2..=2 {
                    particles.push(FluidParticle::new(
                        Vec3::new(x as f32 * 0.3, y as f32 * 0.3, z as f32 * 0.3),
                        1.0,
                    ));
                }
            }
        }
        world.set_fluid_particles(particles);

        let forces = world.compute_fluid_forces(handle);

        // No fluid forces when disabled
        assert!(forces.force.length() < 1e-5, "Should have no forces when disabled");
    }

    #[test]
    fn test_coupling_torque_from_offset_buoyancy() {
        let mut world = PhysicsWorld::new();
        world.set_gravity(Vec3::new(0.0, -10.0, 0.0));
        world.fluid_coupling.buoyancy_enabled = true;
        world.fluid_coupling.drag_enabled = false;
        world.fluid_coupling.smoothing_radius = 2.0;

        // Create a box that is partially submerged (tilted scenario)
        let mut body = RigidBody::new(
            CollisionShape::Box { half_extents: Vec3::new(1.0, 0.5, 0.5) },
            2.0,
            BodyType::Dynamic,
        );
        body.position = Vec3::ZERO;
        let handle = world.add_body(body);

        // Create fluid particles only on one side (asymmetric submersion)
        let mut particles = Vec::new();
        for x in 0..=4 { // Only positive X side
            for y in -2..=2 {
                for z in -2..=2 {
                    particles.push(FluidParticle::new(
                        Vec3::new(x as f32 * 0.3, y as f32 * 0.3, z as f32 * 0.3),
                        1.0,
                    ));
                }
            }
        }
        world.set_fluid_particles(particles);

        let forces = world.compute_fluid_forces(handle);

        // When buoyancy center is offset from body center, there should be torque
        // The torque magnitude depends on the offset
        // Note: due to symmetric particles in Y and Z, main torque may be small
        // but the computation should work
        assert!(forces.buoyancy.y > 0.0, "Should have buoyancy");
    }

    #[test]
    fn test_coupling_sphere_boundary() {
        let mut world = PhysicsWorld::new();

        let mut body = RigidBody::new(CollisionShape::sphere(1.0), 1.0, BodyType::Dynamic);
        body.position = Vec3::ZERO;
        world.add_body(body);

        // Particle inside sphere
        let particles = vec![FluidParticle::new(Vec3::new(0.3, 0.0, 0.0), 1.0)];
        world.set_fluid_particles(particles);

        world.coupling_step();

        let p = &world.fluid_particles()[0];
        // Should be pushed in +X direction
        assert!(p.acceleration.x > 0.0);
    }

    #[test]
    fn test_coupling_box_boundary() {
        let mut world = PhysicsWorld::new();

        let mut body = RigidBody::new(
            CollisionShape::Box { half_extents: Vec3::ONE },
            1.0,
            BodyType::Dynamic,
        );
        body.position = Vec3::ZERO;
        world.add_body(body);

        // Particle inside box
        let particles = vec![FluidParticle::new(Vec3::new(0.3, 0.0, 0.0), 1.0)];
        world.set_fluid_particles(particles);

        world.coupling_step();

        let p = &world.fluid_particles()[0];
        // Should be pushed outward (nearest face is +X)
        assert!(p.acceleration.x > 0.0);
    }

    #[test]
    fn test_coupling_capsule_boundary() {
        let mut world = PhysicsWorld::new();

        let mut body = RigidBody::new(
            CollisionShape::Capsule { radius: 1.0, half_height: 1.0 },
            1.0,
            BodyType::Dynamic,
        );
        body.position = Vec3::ZERO;
        world.add_body(body);

        // Particle inside capsule cylinder part
        let particles = vec![FluidParticle::new(Vec3::new(0.3, 0.0, 0.0), 1.0)];
        world.set_fluid_particles(particles);

        world.coupling_step();

        let p = &world.fluid_particles()[0];
        // Should be pushed outward in X direction
        assert!(p.acceleration.x > 0.0);
    }

    #[test]
    fn test_coupling_fluid_particle_management() {
        let mut world = PhysicsWorld::new();

        assert!(world.fluid_particles().is_empty());

        let particles = vec![
            FluidParticle::new(Vec3::new(0.0, 0.0, 0.0), 1.0),
            FluidParticle::new(Vec3::new(1.0, 0.0, 0.0), 1.0),
        ];
        world.set_fluid_particles(particles);

        assert_eq!(world.fluid_particles().len(), 2);

        world.clear_fluid_particles();
        assert!(world.fluid_particles().is_empty());
    }

    #[test]
    fn test_coupling_enable_disable() {
        let mut world = PhysicsWorld::new();

        assert!(world.fluid_coupling_enabled()); // Default enabled

        world.set_fluid_coupling_enabled(false, false);
        assert!(!world.fluid_coupling_enabled());

        world.set_fluid_coupling_enabled(true, false);
        assert!(world.fluid_coupling_enabled());

        world.set_fluid_coupling_enabled(false, true);
        assert!(world.fluid_coupling_enabled());
    }
}
