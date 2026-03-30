//! Structure-of-Arrays (SoA) data layout for cache-friendly physics.
//!
//! This module provides SoA storage for body data, which improves cache
//! performance when iterating over specific properties of many bodies.

use glam::{Mat3, Quat, Vec3};
use crate::world::BodyHandle;
use std::collections::HashMap;

/// SoA storage for rigid body state.
/// 
/// This layout groups similar data together for better cache utilization
/// when processing many bodies.
#[derive(Debug, Default)]
pub struct BodySoA {
    /// Number of bodies
    pub len: usize,
    
    // Position data
    pub position_x: Vec<f32>,
    pub position_y: Vec<f32>,
    pub position_z: Vec<f32>,
    
    // Rotation data (quaternion)
    pub rotation_x: Vec<f32>,
    pub rotation_y: Vec<f32>,
    pub rotation_z: Vec<f32>,
    pub rotation_w: Vec<f32>,
    
    // Linear velocity
    pub velocity_x: Vec<f32>,
    pub velocity_y: Vec<f32>,
    pub velocity_z: Vec<f32>,
    
    // Angular velocity
    pub angular_x: Vec<f32>,
    pub angular_y: Vec<f32>,
    pub angular_z: Vec<f32>,
    
    // Mass properties
    pub inv_mass: Vec<f32>,
    pub inv_inertia_xx: Vec<f32>,
    pub inv_inertia_yy: Vec<f32>,
    pub inv_inertia_zz: Vec<f32>,
    
    // Forces
    pub force_x: Vec<f32>,
    pub force_y: Vec<f32>,
    pub force_z: Vec<f32>,
    pub torque_x: Vec<f32>,
    pub torque_y: Vec<f32>,
    pub torque_z: Vec<f32>,
    
    // Flags
    pub is_dynamic: Vec<bool>,
    pub is_sleeping: Vec<bool>,
    
    // Handle mapping
    pub handles: Vec<BodyHandle>,
    pub handle_to_index: HashMap<BodyHandle, usize>,
}

impl BodySoA {
    /// Create empty SoA storage.
    pub fn new() -> Self {
        Self::default()
    }
    
    /// Create with capacity.
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            len: 0,
            position_x: Vec::with_capacity(capacity),
            position_y: Vec::with_capacity(capacity),
            position_z: Vec::with_capacity(capacity),
            rotation_x: Vec::with_capacity(capacity),
            rotation_y: Vec::with_capacity(capacity),
            rotation_z: Vec::with_capacity(capacity),
            rotation_w: Vec::with_capacity(capacity),
            velocity_x: Vec::with_capacity(capacity),
            velocity_y: Vec::with_capacity(capacity),
            velocity_z: Vec::with_capacity(capacity),
            angular_x: Vec::with_capacity(capacity),
            angular_y: Vec::with_capacity(capacity),
            angular_z: Vec::with_capacity(capacity),
            inv_mass: Vec::with_capacity(capacity),
            inv_inertia_xx: Vec::with_capacity(capacity),
            inv_inertia_yy: Vec::with_capacity(capacity),
            inv_inertia_zz: Vec::with_capacity(capacity),
            force_x: Vec::with_capacity(capacity),
            force_y: Vec::with_capacity(capacity),
            force_z: Vec::with_capacity(capacity),
            torque_x: Vec::with_capacity(capacity),
            torque_y: Vec::with_capacity(capacity),
            torque_z: Vec::with_capacity(capacity),
            is_dynamic: Vec::with_capacity(capacity),
            is_sleeping: Vec::with_capacity(capacity),
            handles: Vec::with_capacity(capacity),
            handle_to_index: HashMap::with_capacity(capacity),
        }
    }
    
    /// Clear all data.
    pub fn clear(&mut self) {
        self.len = 0;
        self.position_x.clear();
        self.position_y.clear();
        self.position_z.clear();
        self.rotation_x.clear();
        self.rotation_y.clear();
        self.rotation_z.clear();
        self.rotation_w.clear();
        self.velocity_x.clear();
        self.velocity_y.clear();
        self.velocity_z.clear();
        self.angular_x.clear();
        self.angular_y.clear();
        self.angular_z.clear();
        self.inv_mass.clear();
        self.inv_inertia_xx.clear();
        self.inv_inertia_yy.clear();
        self.inv_inertia_zz.clear();
        self.force_x.clear();
        self.force_y.clear();
        self.force_z.clear();
        self.torque_x.clear();
        self.torque_y.clear();
        self.torque_z.clear();
        self.is_dynamic.clear();
        self.is_sleeping.clear();
        self.handles.clear();
        self.handle_to_index.clear();
    }
    
    /// Add a body.
    pub fn push(
        &mut self,
        handle: BodyHandle,
        position: Vec3,
        rotation: Quat,
        velocity: Vec3,
        angular_velocity: Vec3,
        inv_mass: f32,
        inv_inertia: Mat3,
        force: Vec3,
        torque: Vec3,
        is_dynamic: bool,
        is_sleeping: bool,
    ) {
        let idx = self.len;
        self.len += 1;
        
        self.position_x.push(position.x);
        self.position_y.push(position.y);
        self.position_z.push(position.z);
        self.rotation_x.push(rotation.x);
        self.rotation_y.push(rotation.y);
        self.rotation_z.push(rotation.z);
        self.rotation_w.push(rotation.w);
        self.velocity_x.push(velocity.x);
        self.velocity_y.push(velocity.y);
        self.velocity_z.push(velocity.z);
        self.angular_x.push(angular_velocity.x);
        self.angular_y.push(angular_velocity.y);
        self.angular_z.push(angular_velocity.z);
        self.inv_mass.push(inv_mass);
        self.inv_inertia_xx.push(inv_inertia.x_axis.x);
        self.inv_inertia_yy.push(inv_inertia.y_axis.y);
        self.inv_inertia_zz.push(inv_inertia.z_axis.z);
        self.force_x.push(force.x);
        self.force_y.push(force.y);
        self.force_z.push(force.z);
        self.torque_x.push(torque.x);
        self.torque_y.push(torque.y);
        self.torque_z.push(torque.z);
        self.is_dynamic.push(is_dynamic);
        self.is_sleeping.push(is_sleeping);
        self.handles.push(handle);
        self.handle_to_index.insert(handle, idx);
    }
    
    /// Get position at index.
    #[inline]
    pub fn get_position(&self, idx: usize) -> Vec3 {
        Vec3::new(
            self.position_x[idx],
            self.position_y[idx],
            self.position_z[idx],
        )
    }
    
    /// Set position at index.
    #[inline]
    pub fn set_position(&mut self, idx: usize, pos: Vec3) {
        self.position_x[idx] = pos.x;
        self.position_y[idx] = pos.y;
        self.position_z[idx] = pos.z;
    }
    
    /// Get velocity at index.
    #[inline]
    pub fn get_velocity(&self, idx: usize) -> Vec3 {
        Vec3::new(
            self.velocity_x[idx],
            self.velocity_y[idx],
            self.velocity_z[idx],
        )
    }
    
    /// Set velocity at index.
    #[inline]
    pub fn set_velocity(&mut self, idx: usize, vel: Vec3) {
        self.velocity_x[idx] = vel.x;
        self.velocity_y[idx] = vel.y;
        self.velocity_z[idx] = vel.z;
    }
    
    /// Get rotation at index.
    #[inline]
    pub fn get_rotation(&self, idx: usize) -> Quat {
        Quat::from_xyzw(
            self.rotation_x[idx],
            self.rotation_y[idx],
            self.rotation_z[idx],
            self.rotation_w[idx],
        )
    }
    
    /// Set rotation at index.
    #[inline]
    pub fn set_rotation(&mut self, idx: usize, rot: Quat) {
        self.rotation_x[idx] = rot.x;
        self.rotation_y[idx] = rot.y;
        self.rotation_z[idx] = rot.z;
        self.rotation_w[idx] = rot.w;
    }
    
    /// Get angular velocity at index.
    #[inline]
    pub fn get_angular_velocity(&self, idx: usize) -> Vec3 {
        Vec3::new(
            self.angular_x[idx],
            self.angular_y[idx],
            self.angular_z[idx],
        )
    }
    
    /// Set angular velocity at index.
    #[inline]
    pub fn set_angular_velocity(&mut self, idx: usize, ang: Vec3) {
        self.angular_x[idx] = ang.x;
        self.angular_y[idx] = ang.y;
        self.angular_z[idx] = ang.z;
    }
    
    /// Integrate velocities with gravity (SoA-optimized).
    pub fn integrate_velocities(&mut self, gravity: Vec3, dt: f32) {
        for i in 0..self.len {
            if !self.is_dynamic[i] || self.is_sleeping[i] {
                continue;
            }
            
            // Apply gravity + forces
            let acc_x = self.force_x[i] * self.inv_mass[i] + gravity.x;
            let acc_y = self.force_y[i] * self.inv_mass[i] + gravity.y;
            let acc_z = self.force_z[i] * self.inv_mass[i] + gravity.z;
            
            self.velocity_x[i] += acc_x * dt;
            self.velocity_y[i] += acc_y * dt;
            self.velocity_z[i] += acc_z * dt;
            
            // Apply torques
            self.angular_x[i] += self.torque_x[i] * self.inv_inertia_xx[i] * dt;
            self.angular_y[i] += self.torque_y[i] * self.inv_inertia_yy[i] * dt;
            self.angular_z[i] += self.torque_z[i] * self.inv_inertia_zz[i] * dt;
        }
    }
    
    /// Integrate positions (SoA-optimized).
    pub fn integrate_positions(&mut self, dt: f32) {
        for i in 0..self.len {
            if !self.is_dynamic[i] || self.is_sleeping[i] {
                continue;
            }
            
            // Position
            self.position_x[i] += self.velocity_x[i] * dt;
            self.position_y[i] += self.velocity_y[i] * dt;
            self.position_z[i] += self.velocity_z[i] * dt;
            
            // Rotation (quaternion integration)
            let half_dt = 0.5 * dt;
            let ox = self.angular_x[i] * half_dt;
            let oy = self.angular_y[i] * half_dt;
            let oz = self.angular_z[i] * half_dt;
            
            let qx = self.rotation_x[i];
            let qy = self.rotation_y[i];
            let qz = self.rotation_z[i];
            let qw = self.rotation_w[i];
            
            // delta = omega_quat * q
            let dx = ox * qw + oy * qz - oz * qy;
            let dy = oy * qw + oz * qx - ox * qz;
            let dz = oz * qw + ox * qy - oy * qx;
            let dw = -ox * qx - oy * qy - oz * qz;
            
            // q += delta
            let new_x = qx + dx;
            let new_y = qy + dy;
            let new_z = qz + dz;
            let new_w = qw + dw;
            
            // Normalize
            let len = (new_x * new_x + new_y * new_y + new_z * new_z + new_w * new_w).sqrt();
            if len > 1e-10 {
                self.rotation_x[i] = new_x / len;
                self.rotation_y[i] = new_y / len;
                self.rotation_z[i] = new_z / len;
                self.rotation_w[i] = new_w / len;
            }
        }
    }
    
    /// Apply damping (SoA-optimized).
    pub fn apply_damping(&mut self, linear_damping: f32, angular_damping: f32, dt: f32) {
        let linear_factor = (1.0 - linear_damping).powf(dt);
        let angular_factor = (1.0 - angular_damping).powf(dt);
        
        for i in 0..self.len {
            if !self.is_dynamic[i] || self.is_sleeping[i] {
                continue;
            }
            
            self.velocity_x[i] *= linear_factor;
            self.velocity_y[i] *= linear_factor;
            self.velocity_z[i] *= linear_factor;
            
            self.angular_x[i] *= angular_factor;
            self.angular_y[i] *= angular_factor;
            self.angular_z[i] *= angular_factor;
        }
    }
    
    /// Clear forces (SoA-optimized).
    pub fn clear_forces(&mut self) {
        self.force_x.fill(0.0);
        self.force_y.fill(0.0);
        self.force_z.fill(0.0);
        self.torque_x.fill(0.0);
        self.torque_y.fill(0.0);
        self.torque_z.fill(0.0);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use slotmap::SlotMap;
    
    fn make_handle() -> BodyHandle {
        let mut map: SlotMap<BodyHandle, ()> = SlotMap::with_key();
        map.insert(())
    }
    
    #[test]
    fn test_soa_new() {
        let soa = BodySoA::new();
        assert_eq!(soa.len, 0);
    }
    
    #[test]
    fn test_soa_push() {
        let mut soa = BodySoA::new();
        let handle = make_handle();
        
        soa.push(
            handle,
            Vec3::new(1.0, 2.0, 3.0),
            Quat::IDENTITY,
            Vec3::ZERO,
            Vec3::ZERO,
            1.0,
            Mat3::IDENTITY,
            Vec3::ZERO,
            Vec3::ZERO,
            true,
            false,
        );
        
        assert_eq!(soa.len, 1);
        assert_eq!(soa.get_position(0), Vec3::new(1.0, 2.0, 3.0));
    }
    
    #[test]
    fn test_soa_integrate_velocities() {
        let mut soa = BodySoA::new();
        let handle = make_handle();
        
        soa.push(
            handle,
            Vec3::ZERO,
            Quat::IDENTITY,
            Vec3::ZERO,
            Vec3::ZERO,
            1.0,
            Mat3::IDENTITY,
            Vec3::ZERO,
            Vec3::ZERO,
            true,
            false,
        );
        
        soa.integrate_velocities(Vec3::new(0.0, -10.0, 0.0), 1.0);
        
        assert!((soa.get_velocity(0).y - (-10.0)).abs() < 0.001);
    }
    
    #[test]
    fn test_soa_integrate_positions() {
        let mut soa = BodySoA::new();
        let handle = make_handle();
        
        soa.push(
            handle,
            Vec3::ZERO,
            Quat::IDENTITY,
            Vec3::new(10.0, 0.0, 0.0),
            Vec3::ZERO,
            1.0,
            Mat3::IDENTITY,
            Vec3::ZERO,
            Vec3::ZERO,
            true,
            false,
        );
        
        soa.integrate_positions(1.0);
        
        assert!((soa.get_position(0).x - 10.0).abs() < 0.001);
    }
    
    #[test]
    fn test_soa_apply_damping() {
        let mut soa = BodySoA::new();
        let handle = make_handle();
        
        soa.push(
            handle,
            Vec3::ZERO,
            Quat::IDENTITY,
            Vec3::new(100.0, 0.0, 0.0),
            Vec3::ZERO,
            1.0,
            Mat3::IDENTITY,
            Vec3::ZERO,
            Vec3::ZERO,
            true,
            false,
        );
        
        soa.apply_damping(0.1, 0.1, 1.0);
        
        assert!(soa.get_velocity(0).x < 100.0);
    }
    
    #[test]
    fn test_soa_sleeping_skipped() {
        let mut soa = BodySoA::new();
        let handle = make_handle();
        
        soa.push(
            handle,
            Vec3::ZERO,
            Quat::IDENTITY,
            Vec3::ZERO,
            Vec3::ZERO,
            1.0,
            Mat3::IDENTITY,
            Vec3::ZERO,
            Vec3::ZERO,
            true,
            true, // sleeping
        );
        
        soa.integrate_velocities(Vec3::new(0.0, -10.0, 0.0), 1.0);
        
        // Sleeping body should not be affected
        assert_eq!(soa.get_velocity(0), Vec3::ZERO);
    }
}
