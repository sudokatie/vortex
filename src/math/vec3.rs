//! 3D vector type (re-exported from glam).
//!
//! This module provides a thin wrapper around glam's Vec3 for consistency
//! with the rest of the math module.

pub use glam::Vec3;

/// Additional Vec3 utilities for physics.
pub trait Vec3Ext {
    /// Project this vector onto another.
    fn project_onto(self, other: Vec3) -> Vec3;
    
    /// Reject this vector from another (perpendicular component).
    fn reject_from(self, other: Vec3) -> Vec3;
    
    /// Reflect this vector about a normal.
    fn reflect(self, normal: Vec3) -> Vec3;
    
    /// Get any perpendicular vector.
    fn any_orthogonal(self) -> Vec3;
}

impl Vec3Ext for Vec3 {
    #[inline]
    fn project_onto(self, other: Vec3) -> Vec3 {
        let dot = self.dot(other);
        let len_sq = other.length_squared();
        if len_sq < 1e-10 {
            Vec3::ZERO
        } else {
            other * (dot / len_sq)
        }
    }
    
    #[inline]
    fn reject_from(self, other: Vec3) -> Vec3 {
        self - self.project_onto(other)
    }
    
    #[inline]
    fn reflect(self, normal: Vec3) -> Vec3 {
        self - 2.0 * self.dot(normal) * normal
    }
    
    #[inline]
    fn any_orthogonal(self) -> Vec3 {
        if self.x.abs() < 0.9 {
            self.cross(Vec3::X).normalize()
        } else {
            self.cross(Vec3::Y).normalize()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_project_onto() {
        let v = Vec3::new(3.0, 4.0, 0.0);
        let onto = Vec3::X;
        let proj = v.project_onto(onto);
        assert!((proj - Vec3::new(3.0, 0.0, 0.0)).length() < 0.001);
    }
    
    #[test]
    fn test_reject_from() {
        let v = Vec3::new(3.0, 4.0, 0.0);
        let from = Vec3::X;
        let rej = v.reject_from(from);
        assert!((rej - Vec3::new(0.0, 4.0, 0.0)).length() < 0.001);
    }
    
    #[test]
    fn test_reflect() {
        let v = Vec3::new(1.0, -1.0, 0.0).normalize();
        let normal = Vec3::Y;
        let reflected = v.reflect(normal);
        assert!(reflected.y > 0.0);
    }
    
    #[test]
    fn test_any_orthogonal() {
        let v = Vec3::new(1.0, 2.0, 3.0).normalize();
        let orth = v.any_orthogonal();
        assert!(v.dot(orth).abs() < 0.001);
    }
}
