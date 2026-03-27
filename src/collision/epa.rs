//! EPA (Expanding Polytope Algorithm) for penetration depth calculation.

use glam::Vec3;

use super::gjk::Simplex;
use super::CollisionShape;
use crate::math::Transform;

/// Result of EPA - penetration depth and contact information.
#[derive(Debug, Clone, Copy)]
pub struct PenetrationInfo {
    /// Penetration depth (how far shapes overlap).
    pub depth: f32,
    /// Contact normal (from shape B to shape A).
    pub normal: Vec3,
    /// Contact point on shape A (world space).
    pub point_a: Vec3,
    /// Contact point on shape B (world space).
    pub point_b: Vec3,
}

/// Face of the polytope.
#[derive(Debug, Clone, Copy)]
struct Face {
    indices: [usize; 3],
    normal: Vec3,
    distance: f32,
}

impl Face {
    fn new(vertices: &[Vec3], i0: usize, i1: usize, i2: usize) -> Self {
        let a = vertices[i0];
        let b = vertices[i1];
        let c = vertices[i2];

        let ab = b - a;
        let ac = c - a;
        let mut normal = ab.cross(ac);
        
        if normal.length_squared() < 1e-10 {
            normal = Vec3::Y; // Degenerate face
        } else {
            normal = normal.normalize();
        }

        // Ensure normal points away from origin
        if normal.dot(a) < 0.0 {
            Self {
                indices: [i0, i2, i1], // Flip winding
                normal: -normal,
                distance: -normal.dot(a),
            }
        } else {
            Self {
                indices: [i0, i1, i2],
                normal,
                distance: normal.dot(a),
            }
        }
    }
}

/// Polytope for EPA expansion.
struct Polytope {
    vertices: Vec<Vec3>,
    faces: Vec<Face>,
}

impl Polytope {
    /// Create polytope from GJK simplex (must be a tetrahedron).
    fn from_simplex(simplex: &Simplex) -> Option<Self> {
        if simplex.len() != 4 {
            return None;
        }

        let vertices = vec![
            simplex.get(0),
            simplex.get(1),
            simplex.get(2),
            simplex.get(3),
        ];

        let faces = vec![
            Face::new(&vertices, 0, 1, 2),
            Face::new(&vertices, 0, 2, 3),
            Face::new(&vertices, 0, 3, 1),
            Face::new(&vertices, 1, 3, 2),
        ];

        Some(Self { vertices, faces })
    }

    /// Find the face closest to the origin.
    fn closest_face(&self) -> (usize, f32, Vec3) {
        let mut min_dist = f32::MAX;
        let mut min_idx = 0;
        let mut min_normal = Vec3::Y;

        for (i, face) in self.faces.iter().enumerate() {
            if face.distance < min_dist {
                min_dist = face.distance;
                min_idx = i;
                min_normal = face.normal;
            }
        }

        (min_idx, min_dist, min_normal)
    }

    /// Add a new vertex and update faces.
    fn expand(&mut self, new_vertex: Vec3, visible_faces: &[usize]) {
        let new_idx = self.vertices.len();
        self.vertices.push(new_vertex);

        // Collect edges from visible faces
        let mut edges: Vec<(usize, usize)> = Vec::new();

        for &face_idx in visible_faces {
            let face = &self.faces[face_idx];
            for i in 0..3 {
                let edge = (face.indices[i], face.indices[(i + 1) % 3]);
                // Check if edge is shared (appears in reverse)
                let reverse = (edge.1, edge.0);
                if let Some(pos) = edges.iter().position(|e| *e == reverse) {
                    edges.remove(pos);
                } else {
                    edges.push(edge);
                }
            }
        }

        // Remove visible faces (in reverse order to preserve indices)
        let mut to_remove: Vec<_> = visible_faces.to_vec();
        to_remove.sort_by(|a, b| b.cmp(a));
        for idx in to_remove {
            self.faces.swap_remove(idx);
        }

        // Create new faces from horizon edges
        for (i0, i1) in edges {
            let new_face = Face::new(&self.vertices, i0, i1, new_idx);
            self.faces.push(new_face);
        }
    }
}

/// Compute support point for Minkowski difference.
fn support(
    shape_a: &CollisionShape,
    transform_a: &Transform,
    shape_b: &CollisionShape,
    transform_b: &Transform,
    direction: Vec3,
) -> Vec3 {
    let local_dir_a = transform_a.inverse_transform_vector(direction);
    let support_a = transform_a.transform_point(shape_a.support(local_dir_a));

    let local_dir_b = transform_b.inverse_transform_vector(-direction);
    let support_b = transform_b.transform_point(shape_b.support(local_dir_b));

    support_a - support_b
}

/// Run EPA to find penetration depth and contact info.
/// 
/// Requires a valid tetrahedron simplex from GJK.
pub fn epa(
    simplex: Simplex,
    shape_a: &CollisionShape,
    transform_a: &Transform,
    shape_b: &CollisionShape,
    transform_b: &Transform,
) -> Option<PenetrationInfo> {
    let mut polytope = Polytope::from_simplex(&simplex)?;

    const MAX_ITERATIONS: usize = 32;
    const TOLERANCE: f32 = 1e-4;

    for _ in 0..MAX_ITERATIONS {
        let (_face_idx, dist, normal) = polytope.closest_face();

        // Get new support point in direction of closest face normal
        let new_point = support(shape_a, transform_a, shape_b, transform_b, normal);
        let new_dist = new_point.dot(normal);

        // Check convergence
        if (new_dist - dist).abs() < TOLERANCE {
            // Calculate contact points
            let point_a = support_single(shape_a, transform_a, normal);
            let point_b = support_single(shape_b, transform_b, -normal);

            return Some(PenetrationInfo {
                depth: dist,
                normal,
                point_a,
                point_b,
            });
        }

        // Find faces visible from new point
        let mut visible = Vec::new();
        for (i, face) in polytope.faces.iter().enumerate() {
            if face.normal.dot(new_point - polytope.vertices[face.indices[0]]) > 0.0 {
                visible.push(i);
            }
        }

        if visible.is_empty() {
            // No visible faces - we're done
            let point_a = support_single(shape_a, transform_a, normal);
            let point_b = support_single(shape_b, transform_b, -normal);

            return Some(PenetrationInfo {
                depth: dist,
                normal,
                point_a,
                point_b,
            });
        }

        polytope.expand(new_point, &visible);
    }

    // Return best result after max iterations
    let (_, dist, normal) = polytope.closest_face();
    let point_a = support_single(shape_a, transform_a, normal);
    let point_b = support_single(shape_b, transform_b, -normal);

    Some(PenetrationInfo {
        depth: dist,
        normal,
        point_a,
        point_b,
    })
}

/// Get support point for a single shape.
fn support_single(shape: &CollisionShape, transform: &Transform, direction: Vec3) -> Vec3 {
    let local_dir = transform.inverse_transform_vector(direction);
    transform.transform_point(shape.support(local_dir))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::collision::gjk_intersection;

    #[test]
    fn test_spheres_penetration() {
        let sphere = CollisionShape::sphere(1.0);
        let t1 = Transform::from_position(Vec3::ZERO);
        let t2 = Transform::from_position(Vec3::new(1.0, 0.0, 0.0));

        if let Some(simplex) = gjk_intersection(&sphere, &t1, &sphere, &t2) {
            if let Some(info) = epa(simplex, &sphere, &t1, &sphere, &t2) {
                // Penetration should be about 1.0 (2 - 1 = 1)
                assert!(info.depth > 0.5 && info.depth < 1.5);
                // Normal should point along X axis
                assert!(info.normal.x.abs() > 0.9);
            }
        }
    }

    #[test]
    fn test_boxes_penetration() {
        let box_shape = CollisionShape::cube(Vec3::ONE);
        let t1 = Transform::from_position(Vec3::ZERO);
        let t2 = Transform::from_position(Vec3::new(1.0, 0.0, 0.0));

        if let Some(simplex) = gjk_intersection(&box_shape, &t1, &box_shape, &t2) {
            if let Some(info) = epa(simplex, &box_shape, &t1, &box_shape, &t2) {
                // Boxes overlap by 1.0 on X
                assert!(info.depth > 0.5);
                // Normal should be along X
                assert!(info.normal.x.abs() > 0.9);
            }
        }
    }

    #[test]
    fn test_sphere_box_penetration() {
        let sphere = CollisionShape::sphere(1.0);
        let box_shape = CollisionShape::cube(Vec3::ONE);
        let t1 = Transform::from_position(Vec3::ZERO);
        let t2 = Transform::from_position(Vec3::new(1.5, 0.0, 0.0));

        if let Some(simplex) = gjk_intersection(&sphere, &t1, &box_shape, &t2) {
            if let Some(info) = epa(simplex, &sphere, &t1, &box_shape, &t2) {
                assert!(info.depth > 0.0);
            }
        }
    }

    #[test]
    fn test_contact_points() {
        let sphere = CollisionShape::sphere(1.0);
        let t1 = Transform::from_position(Vec3::ZERO);
        let t2 = Transform::from_position(Vec3::new(1.5, 0.0, 0.0));

        if let Some(simplex) = gjk_intersection(&sphere, &t1, &sphere, &t2) {
            if let Some(info) = epa(simplex, &sphere, &t1, &sphere, &t2) {
                // Contact points should be on sphere surfaces
                assert!((info.point_a.length() - 1.0).abs() < 0.1);
                assert!(((info.point_b - t2.position).length() - 1.0).abs() < 0.1);
            }
        }
    }

    #[test]
    fn test_penetration_info_fields() {
        let info = PenetrationInfo {
            depth: 0.5,
            normal: Vec3::X,
            point_a: Vec3::new(1.0, 0.0, 0.0),
            point_b: Vec3::new(0.5, 0.0, 0.0),
        };
        assert_eq!(info.depth, 0.5);
        assert_eq!(info.normal, Vec3::X);
    }

    #[test]
    fn test_deep_penetration() {
        let sphere = CollisionShape::sphere(1.0);
        let t1 = Transform::from_position(Vec3::ZERO);
        let t2 = Transform::from_position(Vec3::new(0.1, 0.0, 0.0));

        if let Some(simplex) = gjk_intersection(&sphere, &t1, &sphere, &t2) {
            if let Some(info) = epa(simplex, &sphere, &t1, &sphere, &t2) {
                // Almost fully overlapping - depth should be close to 1.9
                assert!(info.depth > 1.5);
            }
        }
    }
}
