//! Axis-aligned bounding boxes for broad-phase collision detection.

use glam::Vec3;

use crate::math::Transform;

/// Axis-aligned bounding box.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Aabb {
    /// Minimum corner.
    pub min: Vec3,
    /// Maximum corner.
    pub max: Vec3,
}

impl Aabb {
    /// Create a new AABB from min and max corners.
    pub fn new(min: Vec3, max: Vec3) -> Self {
        Self { min, max }
    }

    /// Create an AABB that contains all given points.
    pub fn from_points(points: &[Vec3]) -> Self {
        if points.is_empty() {
            return Self::new(Vec3::ZERO, Vec3::ZERO);
        }

        let mut min = points[0];
        let mut max = points[0];

        for p in points.iter().skip(1) {
            min = min.min(*p);
            max = max.max(*p);
        }

        Self { min, max }
    }

    /// Create an empty (inverted) AABB for use with expand.
    pub fn empty() -> Self {
        Self {
            min: Vec3::splat(f32::MAX),
            max: Vec3::splat(f32::MIN),
        }
    }
    
    /// Create an AABB from center point and half-extents.
    pub fn from_center_extents(center: Vec3, extents: Vec3) -> Self {
        Self {
            min: center - extents,
            max: center + extents,
        }
    }

    /// Check if this AABB intersects another.
    pub fn intersects(&self, other: &Aabb) -> bool {
        self.min.x <= other.max.x
            && self.max.x >= other.min.x
            && self.min.y <= other.max.y
            && self.max.y >= other.min.y
            && self.min.z <= other.max.z
            && self.max.z >= other.min.z
    }

    /// Check if this AABB contains a point.
    pub fn contains(&self, point: Vec3) -> bool {
        point.x >= self.min.x
            && point.x <= self.max.x
            && point.y >= self.min.y
            && point.y <= self.max.y
            && point.z >= self.min.z
            && point.z <= self.max.z
    }
    
    /// Alias for contains() - check if this AABB contains a point.
    pub fn contains_point(&self, point: Vec3) -> bool {
        self.contains(point)
    }

    /// Check if this AABB fully contains another.
    pub fn contains_aabb(&self, other: &Aabb) -> bool {
        self.min.x <= other.min.x
            && self.max.x >= other.max.x
            && self.min.y <= other.min.y
            && self.max.y >= other.max.y
            && self.min.z <= other.min.z
            && self.max.z >= other.max.z
    }

    /// Create an AABB that contains both this and another.
    pub fn merged(&self, other: &Aabb) -> Aabb {
        Aabb {
            min: self.min.min(other.min),
            max: self.max.max(other.max),
        }
    }

    /// Expand the AABB to include a point.
    pub fn expand(&mut self, point: Vec3) {
        self.min = self.min.min(point);
        self.max = self.max.max(point);
    }

    /// Get the center of the AABB.
    pub fn center(&self) -> Vec3 {
        (self.min + self.max) * 0.5
    }

    /// Get the half-extents (half-size) of the AABB.
    pub fn half_extents(&self) -> Vec3 {
        (self.max - self.min) * 0.5
    }
    
    /// Alias for half_extents() - common in physics engines.
    pub fn extents(&self) -> Vec3 {
        self.half_extents()
    }

    /// Get the full size of the AABB.
    pub fn size(&self) -> Vec3 {
        self.max - self.min
    }

    /// Get the surface area of the AABB.
    pub fn surface_area(&self) -> f32 {
        let d = self.max - self.min;
        2.0 * (d.x * d.y + d.y * d.z + d.z * d.x)
    }

    /// Get the volume of the AABB.
    pub fn volume(&self) -> f32 {
        let d = self.max - self.min;
        d.x * d.y * d.z
    }

    /// Transform this AABB by a transform, returning a new AABB that contains
    /// all 8 corners of the transformed box.
    pub fn transform(&self, t: &Transform) -> Aabb {
        // Transform all 8 corners
        let corners = [
            Vec3::new(self.min.x, self.min.y, self.min.z),
            Vec3::new(self.min.x, self.min.y, self.max.z),
            Vec3::new(self.min.x, self.max.y, self.min.z),
            Vec3::new(self.min.x, self.max.y, self.max.z),
            Vec3::new(self.max.x, self.min.y, self.min.z),
            Vec3::new(self.max.x, self.min.y, self.max.z),
            Vec3::new(self.max.x, self.max.y, self.min.z),
            Vec3::new(self.max.x, self.max.y, self.max.z),
        ];

        let transformed: Vec<Vec3> = corners.iter().map(|c| t.transform_point(*c)).collect();
        Aabb::from_points(&transformed)
    }

    /// Expand the AABB by a margin in all directions.
    pub fn expanded(&self, margin: f32) -> Aabb {
        Aabb {
            min: self.min - Vec3::splat(margin),
            max: self.max + Vec3::splat(margin),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use glam::Quat;
    use std::f32::consts::FRAC_PI_4;

    #[test]
    fn test_new() {
        let aabb = Aabb::new(Vec3::ZERO, Vec3::ONE);
        assert_eq!(aabb.min, Vec3::ZERO);
        assert_eq!(aabb.max, Vec3::ONE);
    }

    #[test]
    fn test_from_points() {
        let points = vec![
            Vec3::new(-1.0, -2.0, -3.0),
            Vec3::new(4.0, 5.0, 6.0),
            Vec3::new(0.0, 0.0, 0.0),
        ];
        let aabb = Aabb::from_points(&points);
        assert_eq!(aabb.min, Vec3::new(-1.0, -2.0, -3.0));
        assert_eq!(aabb.max, Vec3::new(4.0, 5.0, 6.0));
    }

    #[test]
    fn test_intersects() {
        let a = Aabb::new(Vec3::ZERO, Vec3::ONE);
        let b = Aabb::new(Vec3::splat(0.5), Vec3::splat(1.5));
        assert!(a.intersects(&b));
    }

    #[test]
    fn test_no_intersect() {
        let a = Aabb::new(Vec3::ZERO, Vec3::ONE);
        let b = Aabb::new(Vec3::splat(2.0), Vec3::splat(3.0));
        assert!(!a.intersects(&b));
    }

    #[test]
    fn test_contains_point() {
        let aabb = Aabb::new(Vec3::ZERO, Vec3::ONE);
        assert!(aabb.contains(Vec3::splat(0.5)));
        assert!(!aabb.contains(Vec3::splat(2.0)));
    }

    #[test]
    fn test_merged() {
        let a = Aabb::new(Vec3::ZERO, Vec3::ONE);
        let b = Aabb::new(Vec3::splat(2.0), Vec3::splat(3.0));
        let merged = a.merged(&b);
        assert_eq!(merged.min, Vec3::ZERO);
        assert_eq!(merged.max, Vec3::splat(3.0));
    }

    #[test]
    fn test_center() {
        let aabb = Aabb::new(Vec3::ZERO, Vec3::new(2.0, 4.0, 6.0));
        let center = aabb.center();
        assert!((center - Vec3::new(1.0, 2.0, 3.0)).length() < 0.001);
    }

    #[test]
    fn test_surface_area() {
        let aabb = Aabb::new(Vec3::ZERO, Vec3::new(1.0, 2.0, 3.0));
        let area = aabb.surface_area();
        // 2 * (1*2 + 2*3 + 3*1) = 2 * (2 + 6 + 3) = 22
        assert!((area - 22.0).abs() < 0.001);
    }

    #[test]
    fn test_transform_identity() {
        let aabb = Aabb::new(Vec3::ZERO, Vec3::ONE);
        let transformed = aabb.transform(&Transform::IDENTITY);
        assert!((transformed.min - aabb.min).length() < 0.001);
        assert!((transformed.max - aabb.max).length() < 0.001);
    }

    #[test]
    fn test_transform_rotation() {
        let aabb = Aabb::new(Vec3::splat(-1.0), Vec3::splat(1.0));
        let t = Transform::from_rotation(Quat::from_rotation_z(FRAC_PI_4));
        let transformed = aabb.transform(&t);
        // Rotated cube should have larger AABB
        assert!(transformed.half_extents().x > 1.0);
    }
}
