//! 2D vector type (re-exported from glam).
//!
//! This module provides a thin wrapper around glam's Vec2 for consistency
//! with the rest of the math module.

pub use glam::Vec2;

/// Additional Vec2 utilities for physics.
pub trait Vec2Ext {
    /// Perpendicular vector (rotated 90 degrees counter-clockwise).
    fn perp(self) -> Vec2;
    
    /// Cross product (returns scalar z-component).
    fn cross_2d(self, other: Vec2) -> f32;
}

impl Vec2Ext for Vec2 {
    #[inline]
    fn perp(self) -> Vec2 {
        Vec2::new(-self.y, self.x)
    }
    
    #[inline]
    fn cross_2d(self, other: Vec2) -> f32 {
        self.x * other.y - self.y * other.x
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_perp() {
        let v = Vec2::new(1.0, 0.0);
        let p = v.perp();
        assert!((p - Vec2::new(0.0, 1.0)).length() < 0.001);
    }
    
    #[test]
    fn test_cross_2d() {
        let a = Vec2::new(1.0, 0.0);
        let b = Vec2::new(0.0, 1.0);
        assert!((a.cross_2d(b) - 1.0).abs() < 0.001);
    }
}
