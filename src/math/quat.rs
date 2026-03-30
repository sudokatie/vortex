//! Quaternion type (re-exported from glam).
//!
//! This module provides a thin wrapper around glam's Quat for consistency
//! with the rest of the math module.

pub use glam::Quat;

use glam::Vec3;

/// Additional Quat utilities for physics.
pub trait QuatExt {
    /// Integrate angular velocity over time step.
    /// Returns new orientation after rotating by omega * dt.
    fn integrate(self, omega: Vec3, dt: f32) -> Quat;
    
    /// Get the axis of rotation (returns zero vector for identity).
    fn axis_or_zero(self) -> Vec3;
    
    /// Get the angle of rotation in radians.
    fn angle_radians(self) -> f32;
    
    /// Rotate a vector by this quaternion.
    fn rotate_vec3(self, v: Vec3) -> Vec3;
    
    /// Get the twist component around an axis.
    fn twist_around(self, axis: Vec3) -> Quat;
    
    /// Get the swing component perpendicular to an axis.
    fn swing_around(self, axis: Vec3) -> Quat;
}

impl QuatExt for Quat {
    #[inline]
    fn integrate(self, omega: Vec3, dt: f32) -> Quat {
        let half_dt = dt * 0.5;
        let omega_quat = Quat::from_xyzw(
            omega.x * half_dt,
            omega.y * half_dt,
            omega.z * half_dt,
            0.0,
        );
        let delta = omega_quat * self;
        Quat::from_xyzw(
            self.x + delta.x,
            self.y + delta.y,
            self.z + delta.z,
            self.w + delta.w,
        ).normalize()
    }
    
    #[inline]
    fn axis_or_zero(self) -> Vec3 {
        let (axis, _angle) = self.to_axis_angle();
        if axis.is_nan() {
            Vec3::ZERO
        } else {
            axis
        }
    }
    
    #[inline]
    fn angle_radians(self) -> f32 {
        let (_axis, angle) = self.to_axis_angle();
        angle
    }
    
    #[inline]
    fn rotate_vec3(self, v: Vec3) -> Vec3 {
        self * v
    }
    
    fn twist_around(self, axis: Vec3) -> Quat {
        let axis = axis.normalize();
        let projection = Vec3::new(self.x, self.y, self.z).dot(axis) * axis;
        let twist = Quat::from_xyzw(projection.x, projection.y, projection.z, self.w);
        let len = (twist.x * twist.x + twist.y * twist.y + twist.z * twist.z + twist.w * twist.w).sqrt();
        if len < 1e-10 {
            Quat::IDENTITY
        } else {
            Quat::from_xyzw(twist.x / len, twist.y / len, twist.z / len, twist.w / len)
        }
    }
    
    fn swing_around(self, axis: Vec3) -> Quat {
        let twist = self.twist_around(axis);
        self * twist.inverse()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f32::consts::PI;
    
    #[test]
    fn test_integrate() {
        let q = Quat::IDENTITY;
        let omega = Vec3::new(0.0, PI, 0.0); // 180 deg/s around Y
        let result = q.integrate(omega, 1.0);
        
        // After 1 second at PI rad/s, should have rotated significantly
        // The integration is approximate (Euler), so just check it changed
        assert!(result.w < 1.0); // Changed from identity
        assert!((result.w.powi(2) + result.x.powi(2) + result.y.powi(2) + result.z.powi(2) - 1.0).abs() < 0.01); // Still normalized
    }
    
    #[test]
    fn test_axis_angle() {
        let q = Quat::from_rotation_y(PI / 2.0);
        let axis = q.axis_or_zero();
        let angle = q.angle_radians();
        
        assert!(axis.y.abs() > 0.9);
        assert!((angle - PI / 2.0).abs() < 0.01);
    }
    
    #[test]
    fn test_rotate_vec3() {
        let q = Quat::from_rotation_z(PI / 2.0);
        let v = Vec3::X;
        let rotated = q.rotate_vec3(v);
        
        assert!((rotated - Vec3::Y).length() < 0.01);
    }
    
    #[test]
    fn test_twist_swing() {
        let q = Quat::from_rotation_y(PI / 4.0) * Quat::from_rotation_x(PI / 6.0);
        let axis = Vec3::Y;
        
        let twist = q.twist_around(axis);
        let swing = q.swing_around(axis);
        
        // Twist * Swing should reconstruct original (approximately)
        let reconstructed = swing * twist;
        let diff = q * reconstructed.inverse();
        assert!(diff.w.abs() > 0.99);
    }
}
