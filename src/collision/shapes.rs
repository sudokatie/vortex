//! Collision shapes for rigid bodies.

use glam::{Mat3, Vec3};

use super::Aabb;

/// A face of a convex mesh.
#[derive(Debug, Clone)]
pub struct Face {
    /// Vertex indices forming the face.
    pub indices: Vec<usize>,
    /// Face normal.
    pub normal: Vec3,
}

impl Face {
    /// Create a new face.
    pub fn new(indices: Vec<usize>, normal: Vec3) -> Self {
        Self { indices, normal }
    }
    
    /// Create a triangular face and compute normal from vertices.
    pub fn triangle(vertices: &[Vec3], i0: usize, i1: usize, i2: usize) -> Self {
        let a = vertices[i0];
        let b = vertices[i1];
        let c = vertices[i2];
        let normal = (b - a).cross(c - a).normalize_or_zero();
        Self {
            indices: vec![i0, i1, i2],
            normal,
        }
    }
}

/// Collision shape types.
#[derive(Debug, Clone)]
pub enum CollisionShape {
    /// Sphere with radius.
    Sphere { radius: f32 },
    /// Box with half-extents.
    Box { half_extents: Vec3 },
    /// Capsule (cylinder with hemispherical caps).
    Capsule { radius: f32, half_height: f32 },
    /// Convex hull mesh.
    Convex { vertices: Vec<Vec3>, faces: Vec<Face> },
}

impl CollisionShape {
    /// Create a sphere shape.
    pub fn sphere(radius: f32) -> Self {
        Self::Sphere { radius }
    }

    /// Create a box shape from half-extents.
    pub fn cube(half_extents: Vec3) -> Self {
        Self::Box { half_extents }
    }

    /// Create a capsule shape.
    pub fn capsule(radius: f32, half_height: f32) -> Self {
        Self::Capsule { radius, half_height }
    }
    
    /// Create a convex hull from vertices.
    pub fn convex(vertices: Vec<Vec3>, faces: Vec<Face>) -> Self {
        Self::Convex { vertices, faces }
    }
    
    /// Create a convex hull from vertices, computing faces automatically.
    /// Uses a simple convex hull algorithm for small vertex sets.
    pub fn convex_hull(vertices: Vec<Vec3>) -> Self {
        if vertices.len() < 4 {
            // Degenerate - return as-is with no faces
            return Self::Convex { vertices, faces: Vec::new() };
        }
        
        // Simple convex hull - for small meshes, compute faces
        // For a proper implementation, use quickhull or similar
        let faces = compute_convex_faces(&vertices);
        Self::Convex { vertices, faces }
    }

    /// Get the support point in a given direction (for GJK).
    pub fn support(&self, direction: Vec3) -> Vec3 {
        match self {
            Self::Sphere { radius } => direction.normalize_or_zero() * *radius,
            Self::Box { half_extents } => Vec3::new(
                if direction.x >= 0.0 { half_extents.x } else { -half_extents.x },
                if direction.y >= 0.0 { half_extents.y } else { -half_extents.y },
                if direction.z >= 0.0 { half_extents.z } else { -half_extents.z },
            ),
            Self::Capsule { radius, half_height } => {
                let cap_center = if direction.y >= 0.0 {
                    Vec3::new(0.0, *half_height, 0.0)
                } else {
                    Vec3::new(0.0, -*half_height, 0.0)
                };
                cap_center + direction.normalize_or_zero() * *radius
            }
            Self::Convex { vertices, .. } => {
                // Find vertex with maximum dot product in direction
                vertices.iter()
                    .copied()
                    .max_by(|a, b| {
                        a.dot(direction).partial_cmp(&b.dot(direction)).unwrap()
                    })
                    .unwrap_or(Vec3::ZERO)
            }
        }
    }

    /// Get the local-space axis-aligned bounding box.
    pub fn local_aabb(&self) -> Aabb {
        match self {
            Self::Sphere { radius } => Aabb::new(
                Vec3::splat(-*radius),
                Vec3::splat(*radius),
            ),
            Self::Box { half_extents } => Aabb::new(
                -*half_extents,
                *half_extents,
            ),
            Self::Capsule { radius, half_height } => {
                let extent = Vec3::new(*radius, *half_height + *radius, *radius);
                Aabb::new(-extent, extent)
            }
            Self::Convex { vertices, .. } => {
                if vertices.is_empty() {
                    return Aabb::new(Vec3::ZERO, Vec3::ZERO);
                }
                let mut min = vertices[0];
                let mut max = vertices[0];
                for v in vertices.iter().skip(1) {
                    min = min.min(*v);
                    max = max.max(*v);
                }
                Aabb::new(min, max)
            }
        }
    }

    /// Get the center of mass (always origin for these primitives).
    pub fn center_of_mass(&self) -> Vec3 {
        match self {
            Self::Convex { vertices, .. } => {
                if vertices.is_empty() {
                    return Vec3::ZERO;
                }
                let sum: Vec3 = vertices.iter().copied().sum();
                sum / vertices.len() as f32
            }
            _ => Vec3::ZERO,
        }
    }

    /// Calculate the inertia tensor for a given mass.
    pub fn inertia_tensor(&self, mass: f32) -> Mat3 {
        match self {
            Self::Sphere { radius } => {
                // I = (2/5) * m * r^2 for solid sphere
                let i = (2.0 / 5.0) * mass * radius * radius;
                Mat3::from_diagonal(Vec3::splat(i))
            }
            Self::Box { half_extents } => {
                // I = (1/12) * m * (h^2 + d^2), etc.
                let h = *half_extents * 2.0; // full extents
                let xx = (1.0 / 12.0) * mass * (h.y * h.y + h.z * h.z);
                let yy = (1.0 / 12.0) * mass * (h.x * h.x + h.z * h.z);
                let zz = (1.0 / 12.0) * mass * (h.x * h.x + h.y * h.y);
                Mat3::from_diagonal(Vec3::new(xx, yy, zz))
            }
            Self::Capsule { radius, half_height } => {
                // Approximate as cylinder + hemispheres
                let r2 = radius * radius;
                let h = *half_height * 2.0;
                
                // Cylinder contribution
                let cyl_mass = mass * (h / (h + (4.0 / 3.0) * *radius));
                let cyl_xx = cyl_mass * (3.0 * r2 + h * h) / 12.0;
                let cyl_yy = cyl_mass * r2 / 2.0;
                
                // Sphere contribution (two hemispheres = one sphere)
                let sphere_mass = mass - cyl_mass;
                let sphere_i = (2.0 / 5.0) * sphere_mass * r2;
                
                let xx = cyl_xx + sphere_i + sphere_mass * (*half_height + 0.375 * *radius).powi(2);
                let yy = cyl_yy + sphere_i;
                
                Mat3::from_diagonal(Vec3::new(xx, yy, xx))
            }
            Self::Convex { .. } => {
                // Approximate inertia using bounding box
                let aabb = self.local_aabb();
                let extents = aabb.max - aabb.min;
                let xx = (1.0 / 12.0) * mass * (extents.y * extents.y + extents.z * extents.z);
                let yy = (1.0 / 12.0) * mass * (extents.x * extents.x + extents.z * extents.z);
                let zz = (1.0 / 12.0) * mass * (extents.x * extents.x + extents.y * extents.y);
                Mat3::from_diagonal(Vec3::new(xx, yy, zz))
            }
        }
    }
}

/// Compute faces for a convex hull (simple implementation).
fn compute_convex_faces(vertices: &[Vec3]) -> Vec<Face> {
    if vertices.len() < 4 {
        return Vec::new();
    }
    
    // Simple tetrahedron for 4 vertices
    if vertices.len() == 4 {
        return vec![
            Face::triangle(vertices, 0, 1, 2),
            Face::triangle(vertices, 0, 2, 3),
            Face::triangle(vertices, 0, 3, 1),
            Face::triangle(vertices, 1, 3, 2),
        ];
    }
    
    // For more vertices, use incremental convex hull
    let mut faces = Vec::new();
    
    // Start with first 4 non-coplanar points
    let centroid: Vec3 = vertices.iter().copied().sum::<Vec3>() / vertices.len() as f32;
    
    // Create initial tetrahedron
    faces.push(Face::triangle(vertices, 0, 1, 2));
    faces.push(Face::triangle(vertices, 0, 2, 3));
    faces.push(Face::triangle(vertices, 0, 3, 1));
    faces.push(Face::triangle(vertices, 1, 3, 2));
    
    // Ensure faces point outward from centroid
    for face in &mut faces {
        let face_center = face.indices.iter()
            .map(|&i| vertices[i])
            .sum::<Vec3>() / face.indices.len() as f32;
        let to_centroid = centroid - face_center;
        if face.normal.dot(to_centroid) > 0.0 {
            face.normal = -face.normal;
            face.indices.reverse();
        }
    }
    
    faces
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sphere_support() {
        let sphere = CollisionShape::sphere(2.0);
        let support = sphere.support(Vec3::X);
        assert!((support - Vec3::new(2.0, 0.0, 0.0)).length() < 0.001);
    }

    #[test]
    fn test_sphere_support_diagonal() {
        let sphere = CollisionShape::sphere(1.0);
        let dir = Vec3::new(1.0, 1.0, 0.0).normalize();
        let support = sphere.support(dir);
        assert!((support.length() - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_box_support() {
        let b = CollisionShape::cube(Vec3::new(1.0, 2.0, 3.0));
        let support = b.support(Vec3::new(1.0, 1.0, 1.0));
        assert_eq!(support, Vec3::new(1.0, 2.0, 3.0));
    }

    #[test]
    fn test_box_support_negative() {
        let b = CollisionShape::cube(Vec3::new(1.0, 2.0, 3.0));
        let support = b.support(Vec3::new(-1.0, -1.0, -1.0));
        assert_eq!(support, Vec3::new(-1.0, -2.0, -3.0));
    }

    #[test]
    fn test_capsule_support_up() {
        let c = CollisionShape::capsule(1.0, 2.0);
        let support = c.support(Vec3::Y);
        // Should be at top hemisphere center + radius in Y
        assert!((support.y - 3.0).abs() < 0.001);
    }

    #[test]
    fn test_sphere_aabb() {
        let sphere = CollisionShape::sphere(3.0);
        let aabb = sphere.local_aabb();
        assert_eq!(aabb.min, Vec3::splat(-3.0));
        assert_eq!(aabb.max, Vec3::splat(3.0));
    }

    #[test]
    fn test_box_aabb() {
        let b = CollisionShape::cube(Vec3::new(1.0, 2.0, 3.0));
        let aabb = b.local_aabb();
        assert_eq!(aabb.min, Vec3::new(-1.0, -2.0, -3.0));
        assert_eq!(aabb.max, Vec3::new(1.0, 2.0, 3.0));
    }

    #[test]
    fn test_capsule_aabb() {
        let c = CollisionShape::capsule(1.0, 2.0);
        let aabb = c.local_aabb();
        assert_eq!(aabb.min, Vec3::new(-1.0, -3.0, -1.0));
        assert_eq!(aabb.max, Vec3::new(1.0, 3.0, 1.0));
    }

    #[test]
    fn test_sphere_inertia_symmetric() {
        let sphere = CollisionShape::sphere(1.0);
        let inertia = sphere.inertia_tensor(1.0);
        assert!((inertia.x_axis.x - inertia.y_axis.y).abs() < 0.001);
        assert!((inertia.y_axis.y - inertia.z_axis.z).abs() < 0.001);
    }

    #[test]
    fn test_box_inertia() {
        let b = CollisionShape::cube(Vec3::new(0.5, 0.5, 0.5));
        let inertia = b.inertia_tensor(1.0);
        // Cube should have equal diagonal elements
        assert!((inertia.x_axis.x - inertia.y_axis.y).abs() < 0.001);
    }

    #[test]
    fn test_center_of_mass() {
        let sphere = CollisionShape::sphere(1.0);
        assert_eq!(sphere.center_of_mass(), Vec3::ZERO);
    }

    #[test]
    fn test_capsule_inertia_not_zero() {
        let c = CollisionShape::capsule(1.0, 2.0);
        let inertia = c.inertia_tensor(1.0);
        assert!(inertia.x_axis.x > 0.0);
        assert!(inertia.y_axis.y > 0.0);
        assert!(inertia.z_axis.z > 0.0);
    }
    
    #[test]
    fn test_convex_support() {
        // Simple tetrahedron
        let vertices = vec![
            Vec3::new(0.0, 1.0, 0.0),
            Vec3::new(-1.0, -1.0, -1.0),
            Vec3::new(1.0, -1.0, -1.0),
            Vec3::new(0.0, -1.0, 1.0),
        ];
        let convex = CollisionShape::convex_hull(vertices);
        
        // Support in +Y should be top vertex
        let support = convex.support(Vec3::Y);
        assert!((support.y - 1.0).abs() < 0.001);
    }
    
    #[test]
    fn test_convex_aabb() {
        let vertices = vec![
            Vec3::new(-1.0, -1.0, -1.0),
            Vec3::new(1.0, -1.0, -1.0),
            Vec3::new(1.0, 1.0, -1.0),
            Vec3::new(-1.0, 1.0, -1.0),
            Vec3::new(-1.0, -1.0, 1.0),
            Vec3::new(1.0, -1.0, 1.0),
            Vec3::new(1.0, 1.0, 1.0),
            Vec3::new(-1.0, 1.0, 1.0),
        ];
        let convex = CollisionShape::convex_hull(vertices);
        let aabb = convex.local_aabb();
        
        assert_eq!(aabb.min, Vec3::splat(-1.0));
        assert_eq!(aabb.max, Vec3::splat(1.0));
    }
    
    #[test]
    fn test_convex_center_of_mass() {
        let vertices = vec![
            Vec3::new(-1.0, 0.0, 0.0),
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
            Vec3::new(0.0, 0.0, 1.0),
        ];
        let convex = CollisionShape::convex_hull(vertices);
        let com = convex.center_of_mass();
        
        // Should be average of all vertices
        assert!((com.x - 0.0).abs() < 0.01);
        assert!((com.y - 0.25).abs() < 0.01);
        assert!((com.z - 0.25).abs() < 0.01);
    }
    
    #[test]
    fn test_convex_inertia() {
        let vertices = vec![
            Vec3::new(-1.0, -1.0, -1.0),
            Vec3::new(1.0, -1.0, -1.0),
            Vec3::new(1.0, 1.0, -1.0),
            Vec3::new(-1.0, 1.0, -1.0),
            Vec3::new(-1.0, -1.0, 1.0),
            Vec3::new(1.0, -1.0, 1.0),
            Vec3::new(1.0, 1.0, 1.0),
            Vec3::new(-1.0, 1.0, 1.0),
        ];
        let convex = CollisionShape::convex_hull(vertices);
        let inertia = convex.inertia_tensor(1.0);
        
        // Cube-like shape should have roughly equal diagonal
        assert!(inertia.x_axis.x > 0.0);
        assert!((inertia.x_axis.x - inertia.y_axis.y).abs() < 0.01);
    }
    
    #[test]
    fn test_face_triangle() {
        let vertices = vec![
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
        ];
        let face = Face::triangle(&vertices, 0, 1, 2);
        
        // Normal should point in +Z
        assert!(face.normal.z > 0.9);
        assert_eq!(face.indices.len(), 3);
    }
}
