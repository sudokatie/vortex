//! 3x3 matrix type (re-exported from glam).
//!
//! This module provides a thin wrapper around glam's Mat3 for consistency
//! with the rest of the math module.

pub use glam::Mat3;

use glam::Vec3;

/// Additional Mat3 utilities for physics.
pub trait Mat3Ext {
    /// Create a skew-symmetric matrix from a vector.
    /// Used for cross product: skew(a) * b = a × b
    fn skew_symmetric(v: Vec3) -> Mat3;
    
    /// Create outer product matrix: a ⊗ b
    fn outer_product(a: Vec3, b: Vec3) -> Mat3;
    
    /// Extract diagonal as Vec3.
    fn diagonal(&self) -> Vec3;
    
    /// Create from diagonal Vec3.
    fn from_diagonal_vec3(diag: Vec3) -> Mat3;
}

impl Mat3Ext for Mat3 {
    #[inline]
    fn skew_symmetric(v: Vec3) -> Mat3 {
        Mat3::from_cols(
            Vec3::new(0.0, v.z, -v.y),
            Vec3::new(-v.z, 0.0, v.x),
            Vec3::new(v.y, -v.x, 0.0),
        )
    }
    
    #[inline]
    fn outer_product(a: Vec3, b: Vec3) -> Mat3 {
        Mat3::from_cols(
            a * b.x,
            a * b.y,
            a * b.z,
        )
    }
    
    #[inline]
    fn diagonal(&self) -> Vec3 {
        Vec3::new(
            self.x_axis.x,
            self.y_axis.y,
            self.z_axis.z,
        )
    }
    
    #[inline]
    fn from_diagonal_vec3(diag: Vec3) -> Mat3 {
        Mat3::from_diagonal(diag)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_skew_symmetric() {
        let v = Vec3::new(1.0, 2.0, 3.0);
        let skew = Mat3::skew_symmetric(v);
        let b = Vec3::new(4.0, 5.0, 6.0);
        
        let cross = v.cross(b);
        let skew_result = skew * b;
        
        assert!((cross - skew_result).length() < 0.001);
    }
    
    #[test]
    fn test_outer_product() {
        let a = Vec3::X;
        let b = Vec3::Y;
        let outer = Mat3::outer_product(a, b);
        
        // a ⊗ b applied to any vector c gives a * (b · c)
        let c = Vec3::new(0.0, 2.0, 0.0);
        let result = outer * c;
        let expected = a * b.dot(c);
        
        assert!((result - expected).length() < 0.001);
    }
    
    #[test]
    fn test_diagonal() {
        let m = Mat3::from_diagonal(Vec3::new(1.0, 2.0, 3.0));
        let diag = m.diagonal();
        assert!((diag - Vec3::new(1.0, 2.0, 3.0)).length() < 0.001);
    }
}
