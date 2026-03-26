//! Transform representing position and rotation in 3D space.

use glam::{Mat4, Quat, Vec3};
use std::ops::Mul;

/// A transform representing position and rotation.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Transform {
    /// Position in world space.
    pub position: Vec3,
    /// Rotation as a quaternion.
    pub rotation: Quat,
}

impl Default for Transform {
    fn default() -> Self {
        Self::IDENTITY
    }
}

impl Transform {
    /// Identity transform (no translation or rotation).
    pub const IDENTITY: Self = Self {
        position: Vec3::ZERO,
        rotation: Quat::IDENTITY,
    };

    /// Create a new transform from position and rotation.
    pub fn new(position: Vec3, rotation: Quat) -> Self {
        Self { position, rotation }
    }

    /// Create a transform from just a position.
    pub fn from_position(position: Vec3) -> Self {
        Self {
            position,
            rotation: Quat::IDENTITY,
        }
    }

    /// Create a transform from just a rotation.
    pub fn from_rotation(rotation: Quat) -> Self {
        Self {
            position: Vec3::ZERO,
            rotation,
        }
    }

    /// Create a transform from xyz coordinates.
    pub fn from_xyz(x: f32, y: f32, z: f32) -> Self {
        Self::from_position(Vec3::new(x, y, z))
    }

    /// Transform a point from local space to world space.
    pub fn transform_point(&self, point: Vec3) -> Vec3 {
        self.rotation * point + self.position
    }

    /// Transform a vector from local space to world space (ignores translation).
    pub fn transform_vector(&self, vector: Vec3) -> Vec3 {
        self.rotation * vector
    }

    /// Transform a point from world space to local space.
    pub fn inverse_transform_point(&self, point: Vec3) -> Vec3 {
        self.rotation.inverse() * (point - self.position)
    }

    /// Transform a vector from world space to local space.
    pub fn inverse_transform_vector(&self, vector: Vec3) -> Vec3 {
        self.rotation.inverse() * vector
    }

    /// Get the inverse of this transform.
    pub fn inverse(&self) -> Self {
        let inv_rotation = self.rotation.inverse();
        Self {
            position: inv_rotation * (-self.position),
            rotation: inv_rotation,
        }
    }

    /// Convert to a 4x4 transformation matrix.
    pub fn to_matrix(&self) -> Mat4 {
        Mat4::from_rotation_translation(self.rotation, self.position)
    }

    /// Create from a 4x4 transformation matrix.
    pub fn from_matrix(matrix: Mat4) -> Self {
        let (_, rotation, position) = matrix.to_scale_rotation_translation();
        Self { position, rotation }
    }

    /// Linearly interpolate between two transforms.
    pub fn lerp(&self, other: &Self, t: f32) -> Self {
        Self {
            position: self.position.lerp(other.position, t),
            rotation: self.rotation.slerp(other.rotation, t),
        }
    }

    /// Get the forward direction (negative Z in local space).
    pub fn forward(&self) -> Vec3 {
        self.rotation * Vec3::NEG_Z
    }

    /// Get the right direction (positive X in local space).
    pub fn right(&self) -> Vec3 {
        self.rotation * Vec3::X
    }

    /// Get the up direction (positive Y in local space).
    pub fn up(&self) -> Vec3 {
        self.rotation * Vec3::Y
    }
}

impl Mul<Transform> for Transform {
    type Output = Transform;

    /// Combine two transforms (apply self, then other).
    fn mul(self, other: Transform) -> Transform {
        Transform {
            position: self.transform_point(other.position),
            rotation: self.rotation * other.rotation,
        }
    }
}

impl Mul<Vec3> for Transform {
    type Output = Vec3;

    /// Transform a point.
    fn mul(self, point: Vec3) -> Vec3 {
        self.transform_point(point)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f32::consts::FRAC_PI_2;

    #[test]
    fn test_identity() {
        let t = Transform::IDENTITY;
        assert_eq!(t.position, Vec3::ZERO);
        assert_eq!(t.rotation, Quat::IDENTITY);
    }

    #[test]
    fn test_from_position() {
        let t = Transform::from_position(Vec3::new(1.0, 2.0, 3.0));
        assert_eq!(t.position, Vec3::new(1.0, 2.0, 3.0));
        assert_eq!(t.rotation, Quat::IDENTITY);
    }

    #[test]
    fn test_transform_point_translation() {
        let t = Transform::from_position(Vec3::new(1.0, 0.0, 0.0));
        let p = t.transform_point(Vec3::new(2.0, 0.0, 0.0));
        assert!((p - Vec3::new(3.0, 0.0, 0.0)).length() < 0.001);
    }

    #[test]
    fn test_transform_point_rotation() {
        let t = Transform::from_rotation(Quat::from_rotation_y(FRAC_PI_2));
        let p = t.transform_point(Vec3::new(1.0, 0.0, 0.0));
        // 90 degree rotation around Y: X becomes -Z
        assert!((p - Vec3::new(0.0, 0.0, -1.0)).length() < 0.001);
    }

    #[test]
    fn test_transform_vector_ignores_translation() {
        let t = Transform::new(
            Vec3::new(10.0, 10.0, 10.0),
            Quat::IDENTITY,
        );
        let v = t.transform_vector(Vec3::new(1.0, 0.0, 0.0));
        assert!((v - Vec3::new(1.0, 0.0, 0.0)).length() < 0.001);
    }

    #[test]
    fn test_inverse() {
        let t = Transform::new(
            Vec3::new(1.0, 2.0, 3.0),
            Quat::from_rotation_z(FRAC_PI_2),
        );
        let inv = t.inverse();
        let result = t * inv;
        assert!((result.position - Vec3::ZERO).length() < 0.001);
        assert!((result.rotation.x - Quat::IDENTITY.x).abs() < 0.001);
    }

    #[test]
    fn test_inverse_transform_point() {
        let t = Transform::from_position(Vec3::new(5.0, 0.0, 0.0));
        let world_point = Vec3::new(7.0, 0.0, 0.0);
        let local = t.inverse_transform_point(world_point);
        assert!((local - Vec3::new(2.0, 0.0, 0.0)).length() < 0.001);
    }

    #[test]
    fn test_mul_transforms() {
        let t1 = Transform::from_position(Vec3::new(1.0, 0.0, 0.0));
        let t2 = Transform::from_position(Vec3::new(0.0, 1.0, 0.0));
        let combined = t1 * t2;
        assert!((combined.position - Vec3::new(1.0, 1.0, 0.0)).length() < 0.001);
    }

    #[test]
    fn test_lerp() {
        let t1 = Transform::from_position(Vec3::ZERO);
        let t2 = Transform::from_position(Vec3::new(10.0, 0.0, 0.0));
        let mid = t1.lerp(&t2, 0.5);
        assert!((mid.position - Vec3::new(5.0, 0.0, 0.0)).length() < 0.001);
    }

    #[test]
    fn test_directions() {
        let t = Transform::IDENTITY;
        assert!((t.forward() - Vec3::NEG_Z).length() < 0.001);
        assert!((t.right() - Vec3::X).length() < 0.001);
        assert!((t.up() - Vec3::Y).length() < 0.001);
    }
}
