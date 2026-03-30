//! 2D physics module.
//! 
//! Provides 2D-specific types and algorithms for physics simulation.
//! Enable with the `dim2` feature flag.

mod body;
mod collision;
pub mod constraints;
mod shapes;
mod world;

pub use body::*;
pub use collision::*;
pub use constraints::{
    SolverBody2D, SolverConfig2D, ConstraintSolver2D,
    ContactConstraint2D, DistanceJoint2D, RevoluteJoint2D,
};
pub use shapes::*;

/// Alias for Shape2D for API consistency.
pub type CollisionShape2D = Shape2D;
pub use world::{BodyHandle2D, PhysicsWorld2D};

use glam::Vec2;

/// 2D transform (position + rotation angle)
#[derive(Debug, Clone, Copy, Default)]
pub struct Transform2D {
    pub position: Vec2,
    pub rotation: f32, // radians
}

impl Transform2D {
    pub const IDENTITY: Self = Self {
        position: Vec2::ZERO,
        rotation: 0.0,
    };
    
    pub fn new(position: Vec2, rotation: f32) -> Self {
        Self { position, rotation }
    }
    
    pub fn from_position(position: Vec2) -> Self {
        Self { position, rotation: 0.0 }
    }
    
    pub fn from_rotation(rotation: f32) -> Self {
        Self { position: Vec2::ZERO, rotation }
    }
    
    /// Transform a local point to world space
    pub fn transform_point(&self, point: Vec2) -> Vec2 {
        let cos = self.rotation.cos();
        let sin = self.rotation.sin();
        Vec2::new(
            point.x * cos - point.y * sin + self.position.x,
            point.x * sin + point.y * cos + self.position.y,
        )
    }
    
    /// Transform a local direction to world space
    pub fn transform_direction(&self, dir: Vec2) -> Vec2 {
        let cos = self.rotation.cos();
        let sin = self.rotation.sin();
        Vec2::new(
            dir.x * cos - dir.y * sin,
            dir.x * sin + dir.y * cos,
        )
    }
    
    /// Transform world point to local space
    pub fn inverse_transform_point(&self, point: Vec2) -> Vec2 {
        let p = point - self.position;
        let cos = self.rotation.cos();
        let sin = self.rotation.sin();
        Vec2::new(
            p.x * cos + p.y * sin,
            -p.x * sin + p.y * cos,
        )
    }
    
    /// Compose two transforms: self * other
    pub fn compose(&self, other: &Transform2D) -> Transform2D {
        Transform2D {
            position: self.transform_point(other.position),
            rotation: self.rotation + other.rotation,
        }
    }
    
    /// Get the inverse transform
    pub fn inverse(&self) -> Transform2D {
        let cos = self.rotation.cos();
        let sin = self.rotation.sin();
        Transform2D {
            position: Vec2::new(
                -self.position.x * cos - self.position.y * sin,
                self.position.x * sin - self.position.y * cos,
            ),
            rotation: -self.rotation,
        }
    }
}

/// 2D axis-aligned bounding box
#[derive(Debug, Clone, Copy, Default)]
pub struct Aabb2D {
    pub min: Vec2,
    pub max: Vec2,
}

impl Aabb2D {
    pub fn new(min: Vec2, max: Vec2) -> Self {
        Self { min, max }
    }
    
    pub fn from_center_extents(center: Vec2, extents: Vec2) -> Self {
        Self {
            min: center - extents,
            max: center + extents,
        }
    }
    
    pub fn intersects(&self, other: &Aabb2D) -> bool {
        self.max.x >= other.min.x && self.min.x <= other.max.x &&
        self.max.y >= other.min.y && self.min.y <= other.max.y
    }
    
    pub fn contains_point(&self, point: Vec2) -> bool {
        point.x >= self.min.x && point.x <= self.max.x &&
        point.y >= self.min.y && point.y <= self.max.y
    }
    
    pub fn center(&self) -> Vec2 {
        (self.min + self.max) * 0.5
    }
    
    pub fn extents(&self) -> Vec2 {
        (self.max - self.min) * 0.5
    }
    
    pub fn merged(&self, other: &Aabb2D) -> Aabb2D {
        Aabb2D {
            min: self.min.min(other.min),
            max: self.max.max(other.max),
        }
    }
    
    pub fn expand(&mut self, point: Vec2) {
        self.min = self.min.min(point);
        self.max = self.max.max(point);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f32::consts::PI;

    #[test]
    fn test_transform_identity() {
        let t = Transform2D::IDENTITY;
        let p = Vec2::new(1.0, 2.0);
        assert_eq!(t.transform_point(p), p);
    }
    
    #[test]
    fn test_transform_translate() {
        let t = Transform2D::from_position(Vec2::new(3.0, 4.0));
        let p = Vec2::new(1.0, 2.0);
        assert_eq!(t.transform_point(p), Vec2::new(4.0, 6.0));
    }
    
    #[test]
    fn test_transform_rotate() {
        let t = Transform2D::from_rotation(PI / 2.0); // 90 degrees
        let p = Vec2::new(1.0, 0.0);
        let result = t.transform_point(p);
        assert!((result.x - 0.0).abs() < 0.001);
        assert!((result.y - 1.0).abs() < 0.001);
    }
    
    #[test]
    fn test_transform_inverse() {
        let t = Transform2D::new(Vec2::new(3.0, 4.0), PI / 4.0);
        let inv = t.inverse();
        let p = Vec2::new(1.0, 2.0);
        let transformed = t.transform_point(p);
        let back = inv.transform_point(transformed);
        assert!((back - p).length() < 0.001);
    }
    
    #[test]
    fn test_aabb_intersects() {
        let a = Aabb2D::new(Vec2::ZERO, Vec2::ONE);
        let b = Aabb2D::new(Vec2::splat(0.5), Vec2::splat(1.5));
        let c = Aabb2D::new(Vec2::splat(2.0), Vec2::splat(3.0));
        
        assert!(a.intersects(&b));
        assert!(!a.intersects(&c));
    }
    
    #[test]
    fn test_aabb_merged() {
        let a = Aabb2D::new(Vec2::ZERO, Vec2::ONE);
        let b = Aabb2D::new(Vec2::splat(2.0), Vec2::splat(3.0));
        let merged = a.merged(&b);
        
        assert_eq!(merged.min, Vec2::ZERO);
        assert_eq!(merged.max, Vec2::splat(3.0));
    }
}
