//! GJK (Gilbert-Johnson-Keerthi) collision detection algorithm.

use glam::Vec3;

use super::CollisionShape;
use crate::math::Transform;

/// A simplex used in GJK iteration (up to 4 points for 3D).
#[derive(Debug, Clone)]
pub struct Simplex {
    points: [Vec3; 4],
    size: usize,
}

impl Default for Simplex {
    fn default() -> Self {
        Self::new()
    }
}

impl Simplex {
    /// Create an empty simplex.
    pub fn new() -> Self {
        Self {
            points: [Vec3::ZERO; 4],
            size: 0,
        }
    }

    /// Add a point to the simplex.
    pub fn push(&mut self, point: Vec3) {
        if self.size < 4 {
            // Shift existing points
            for i in (1..=self.size).rev() {
                self.points[i] = self.points[i - 1];
            }
            self.points[0] = point;
            self.size += 1;
        }
    }

    /// Get the number of points.
    pub fn len(&self) -> usize {
        self.size
    }

    /// Check if empty.
    pub fn is_empty(&self) -> bool {
        self.size == 0
    }

    /// Get a point by index.
    pub fn get(&self, index: usize) -> Vec3 {
        self.points[index]
    }

    /// Set the simplex to a line.
    pub fn set_line(&mut self, a: Vec3, b: Vec3) {
        self.points[0] = a;
        self.points[1] = b;
        self.size = 2;
    }

    /// Set the simplex to a triangle.
    pub fn set_triangle(&mut self, a: Vec3, b: Vec3, c: Vec3) {
        self.points[0] = a;
        self.points[1] = b;
        self.points[2] = c;
        self.size = 3;
    }

    /// Set the simplex to a tetrahedron.
    pub fn set_tetrahedron(&mut self, a: Vec3, b: Vec3, c: Vec3, d: Vec3) {
        self.points[0] = a;
        self.points[1] = b;
        self.points[2] = c;
        self.points[3] = d;
        self.size = 4;
    }
}

/// Compute the support point for the Minkowski difference A - B.
fn support(
    shape_a: &CollisionShape,
    transform_a: &Transform,
    shape_b: &CollisionShape,
    transform_b: &Transform,
    direction: Vec3,
) -> Vec3 {
    // Support in direction for A
    let local_dir_a = transform_a.inverse_transform_vector(direction);
    let support_a = transform_a.transform_point(shape_a.support(local_dir_a));

    // Support in opposite direction for B
    let local_dir_b = transform_b.inverse_transform_vector(-direction);
    let support_b = transform_b.transform_point(shape_b.support(local_dir_b));

    // Minkowski difference
    support_a - support_b
}

/// Same direction test (dot product > 0).
fn same_direction(a: Vec3, b: Vec3) -> bool {
    a.dot(b) > 1e-10
}

/// Process a line simplex and update direction toward origin.
fn do_simplex_line(simplex: &mut Simplex, direction: &mut Vec3) -> bool {
    let a = simplex.get(0);
    let b = simplex.get(1);

    let ab = b - a;
    let ao = -a;

    if same_direction(ab, ao) {
        // Origin is between A and B (closest to edge AB)
        *direction = ab.cross(ao).cross(ab);
    } else {
        // Origin is past A
        simplex.size = 1;
        *direction = ao;
    }

    false
}

/// Process a triangle simplex and update direction toward origin.
fn do_simplex_triangle(simplex: &mut Simplex, direction: &mut Vec3) -> bool {
    let a = simplex.get(0);
    let b = simplex.get(1);
    let c = simplex.get(2);

    let ab = b - a;
    let ac = c - a;
    let ao = -a;

    let abc = ab.cross(ac);

    if same_direction(abc.cross(ac), ao) {
        if same_direction(ac, ao) {
            simplex.set_line(a, c);
            *direction = ac.cross(ao).cross(ac);
        } else {
            simplex.set_line(a, b);
            return do_simplex_line(simplex, direction);
        }
    } else if same_direction(ab.cross(abc), ao) {
        simplex.set_line(a, b);
        return do_simplex_line(simplex, direction);
    } else if same_direction(abc, ao) {
        *direction = abc;
    } else {
        simplex.set_triangle(a, c, b);
        *direction = -abc;
    }

    false
}

/// Process a tetrahedron simplex and update direction toward origin.
fn do_simplex_tetrahedron(simplex: &mut Simplex, direction: &mut Vec3) -> bool {
    let a = simplex.get(0);
    let b = simplex.get(1);
    let c = simplex.get(2);
    let d = simplex.get(3);

    let ab = b - a;
    let ac = c - a;
    let ad = d - a;
    let ao = -a;

    let abc = ab.cross(ac);
    let acd = ac.cross(ad);
    let adb = ad.cross(ab);

    if same_direction(abc, ao) {
        simplex.set_triangle(a, b, c);
        return do_simplex_triangle(simplex, direction);
    }

    if same_direction(acd, ao) {
        simplex.set_triangle(a, c, d);
        return do_simplex_triangle(simplex, direction);
    }

    if same_direction(adb, ao) {
        simplex.set_triangle(a, d, b);
        return do_simplex_triangle(simplex, direction);
    }

    // Origin is inside tetrahedron
    true
}

/// Update the simplex and search direction. Returns true if origin is contained.
fn do_simplex(simplex: &mut Simplex, direction: &mut Vec3) -> bool {
    match simplex.len() {
        2 => do_simplex_line(simplex, direction),
        3 => do_simplex_triangle(simplex, direction),
        4 => do_simplex_tetrahedron(simplex, direction),
        _ => false,
    }
}

/// Test if two shapes intersect using GJK.
/// 
/// Returns the final simplex if they intersect (for EPA), or None if they don't.
pub fn gjk_intersection(
    shape_a: &CollisionShape,
    transform_a: &Transform,
    shape_b: &CollisionShape,
    transform_b: &Transform,
) -> Option<Simplex> {
    // Initial direction: from A to B
    let mut direction = transform_b.position - transform_a.position;
    if direction.length_squared() < 1e-10 {
        direction = Vec3::X;
    }
    direction = direction.normalize();

    let mut simplex = Simplex::new();

    // Get first support point
    let first = support(shape_a, transform_a, shape_b, transform_b, direction);
    simplex.push(first);

    // New direction toward origin
    direction = -first;
    if direction.length_squared() < 1e-10 {
        // First point is at origin - shapes are intersecting
        return Some(simplex);
    }
    direction = direction.normalize();

    const MAX_ITERATIONS: usize = 64;

    for _ in 0..MAX_ITERATIONS {
        let new_point = support(shape_a, transform_a, shape_b, transform_b, direction);

        // If new point didn't pass the origin, no intersection
        if new_point.dot(direction) < 1e-10 {
            return None;
        }

        simplex.push(new_point);

        if do_simplex(&mut simplex, &mut direction) {
            return Some(simplex);
        }

        // Normalize and check for zero direction
        let len_sq = direction.length_squared();
        if len_sq < 1e-10 {
            // Direction collapsed - likely at origin
            return Some(simplex);
        }
        direction /= len_sq.sqrt();
    }

    // If we haven't found a result after max iterations, assume intersection
    // (conservative approach for physics)
    Some(simplex)
}

/// Calculate the distance between two shapes.
/// Returns 0 if they are intersecting.
pub fn gjk_distance(
    shape_a: &CollisionShape,
    transform_a: &Transform,
    shape_b: &CollisionShape,
    transform_b: &Transform,
) -> f32 {
    // First check if intersecting
    if gjk_intersection(shape_a, transform_a, shape_b, transform_b).is_some() {
        return 0.0;
    }

    let mut direction = transform_b.position - transform_a.position;
    if direction.length_squared() < 1e-10 {
        direction = Vec3::X;
    }
    direction = direction.normalize();

    let mut simplex = Simplex::new();
    let first = support(shape_a, transform_a, shape_b, transform_b, direction);
    simplex.push(first);

    let mut closest = first;
    let mut closest_dist = first.length();

    const MAX_ITERATIONS: usize = 32;

    for _ in 0..MAX_ITERATIONS {
        direction = -closest;
        if direction.length_squared() < 1e-10 {
            return 0.0;
        }
        direction = direction.normalize();

        let new_point = support(shape_a, transform_a, shape_b, transform_b, direction);
        let projection = new_point.dot(direction);

        // Check if we made progress
        if closest_dist - projection < 1e-6 {
            break;
        }

        simplex.push(new_point);

        // Find closest point on simplex to origin
        closest = closest_point_to_origin(&simplex);
        closest_dist = closest.length();

        if closest_dist < 1e-10 {
            return 0.0;
        }
    }

    closest_dist
}

/// Find the closest point on a simplex to the origin.
fn closest_point_to_origin(simplex: &Simplex) -> Vec3 {
    match simplex.len() {
        1 => simplex.get(0),
        2 => closest_point_on_line(simplex.get(0), simplex.get(1)),
        3 => closest_point_on_triangle(simplex.get(0), simplex.get(1), simplex.get(2)),
        _ => simplex.get(0), // Fallback
    }
}

fn closest_point_on_line(a: Vec3, b: Vec3) -> Vec3 {
    let ab = b - a;
    let t = (-a).dot(ab) / ab.dot(ab);
    let t = t.clamp(0.0, 1.0);
    a + ab * t
}

fn closest_point_on_triangle(a: Vec3, b: Vec3, c: Vec3) -> Vec3 {
    let ab = b - a;
    let ac = c - a;
    let ao = -a;

    let d1 = ab.dot(ao);
    let d2 = ac.dot(ao);
    if d1 <= 0.0 && d2 <= 0.0 {
        return a;
    }

    let bo = -b;
    let d3 = ab.dot(bo);
    let d4 = ac.dot(bo);
    if d3 >= 0.0 && d4 <= d3 {
        return b;
    }

    let vc = d1 * d4 - d3 * d2;
    if vc <= 0.0 && d1 >= 0.0 && d3 <= 0.0 {
        let v = d1 / (d1 - d3);
        return a + ab * v;
    }

    let co = -c;
    let d5 = ab.dot(co);
    let d6 = ac.dot(co);
    if d6 >= 0.0 && d5 <= d6 {
        return c;
    }

    let vb = d5 * d2 - d1 * d6;
    if vb <= 0.0 && d2 >= 0.0 && d6 <= 0.0 {
        let w = d2 / (d2 - d6);
        return a + ac * w;
    }

    let va = d3 * d6 - d5 * d4;
    if va <= 0.0 && (d4 - d3) >= 0.0 && (d5 - d6) >= 0.0 {
        let w = (d4 - d3) / ((d4 - d3) + (d5 - d6));
        return b + (c - b) * w;
    }

    let denom = 1.0 / (va + vb + vc);
    let v = vb * denom;
    let w = vc * denom;
    a + ab * v + ac * w
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simplex_push() {
        let mut s = Simplex::new();
        s.push(Vec3::X);
        s.push(Vec3::Y);
        assert_eq!(s.len(), 2);
        assert_eq!(s.get(0), Vec3::Y); // Most recent first
    }

    #[test]
    fn test_spheres_intersecting() {
        let sphere = CollisionShape::sphere(1.0);
        let t1 = Transform::from_position(Vec3::ZERO);
        let t2 = Transform::from_position(Vec3::new(1.0, 0.0, 0.0));

        let result = gjk_intersection(&sphere, &t1, &sphere, &t2);
        assert!(result.is_some());
    }

    #[test]
    fn test_spheres_not_intersecting() {
        let sphere = CollisionShape::sphere(1.0);
        let t1 = Transform::from_position(Vec3::ZERO);
        let t2 = Transform::from_position(Vec3::new(5.0, 0.0, 0.0));

        let result = gjk_intersection(&sphere, &t1, &sphere, &t2);
        assert!(result.is_none());
    }

    #[test]
    fn test_boxes_intersecting() {
        let box_shape = CollisionShape::cube(Vec3::ONE);
        let t1 = Transform::from_position(Vec3::ZERO);
        let t2 = Transform::from_position(Vec3::new(1.0, 0.0, 0.0));

        let result = gjk_intersection(&box_shape, &t1, &box_shape, &t2);
        assert!(result.is_some());
    }

    #[test]
    fn test_boxes_not_intersecting() {
        let box_shape = CollisionShape::cube(Vec3::ONE);
        let t1 = Transform::from_position(Vec3::ZERO);
        let t2 = Transform::from_position(Vec3::new(5.0, 0.0, 0.0));

        let result = gjk_intersection(&box_shape, &t1, &box_shape, &t2);
        assert!(result.is_none());
    }

    #[test]
    fn test_sphere_box_intersecting() {
        let sphere = CollisionShape::sphere(1.0);
        let box_shape = CollisionShape::cube(Vec3::ONE);
        let t1 = Transform::from_position(Vec3::ZERO);
        let t2 = Transform::from_position(Vec3::new(1.5, 0.0, 0.0));

        let result = gjk_intersection(&sphere, &t1, &box_shape, &t2);
        assert!(result.is_some());
    }

    #[test]
    fn test_distance_spheres() {
        let sphere = CollisionShape::sphere(1.0);
        let t1 = Transform::from_position(Vec3::ZERO);
        let t2 = Transform::from_position(Vec3::new(5.0, 0.0, 0.0));

        let dist = gjk_distance(&sphere, &t1, &sphere, &t2);
        // Distance should be 5 - 2 = 3 (minus two radii)
        assert!((dist - 3.0).abs() < 0.1);
    }

    #[test]
    fn test_distance_touching() {
        let sphere = CollisionShape::sphere(1.0);
        let t1 = Transform::from_position(Vec3::ZERO);
        let t2 = Transform::from_position(Vec3::new(2.0, 0.0, 0.0));

        let dist = gjk_distance(&sphere, &t1, &sphere, &t2);
        assert!(dist < 0.1);
    }

    #[test]
    fn test_distance_intersecting() {
        let sphere = CollisionShape::sphere(1.0);
        let t1 = Transform::from_position(Vec3::ZERO);
        let t2 = Transform::from_position(Vec3::new(1.0, 0.0, 0.0));

        let dist = gjk_distance(&sphere, &t1, &sphere, &t2);
        assert!(dist < 0.1);
    }

    #[test]
    fn test_capsule_sphere() {
        let capsule = CollisionShape::capsule(0.5, 1.0);
        let sphere = CollisionShape::sphere(0.5);
        let t1 = Transform::from_position(Vec3::ZERO);
        let t2 = Transform::from_position(Vec3::new(0.0, 2.0, 0.0));

        // Should intersect (capsule height + radius = 1.5, sphere center at 2, radius 0.5)
        let result = gjk_intersection(&capsule, &t1, &sphere, &t2);
        assert!(result.is_some());
    }

    #[test]
    fn test_rotated_boxes() {
        use glam::Quat;
        use std::f32::consts::FRAC_PI_4;

        let box_shape = CollisionShape::cube(Vec3::ONE);
        let t1 = Transform::from_position(Vec3::ZERO);
        let t2 = Transform::new(
            Vec3::new(1.8, 0.0, 0.0),
            Quat::from_rotation_z(FRAC_PI_4),
        );

        // Rotated box should still intersect
        let result = gjk_intersection(&box_shape, &t1, &box_shape, &t2);
        assert!(result.is_some());
    }

    #[test]
    fn test_same_position_spheres() {
        let sphere = CollisionShape::sphere(1.0);
        let t1 = Transform::from_position(Vec3::ZERO);
        let t2 = Transform::from_position(Vec3::ZERO);

        let result = gjk_intersection(&sphere, &t1, &sphere, &t2);
        assert!(result.is_some());
    }

    #[test]
    fn test_closest_point_on_line() {
        let closest = closest_point_on_line(Vec3::new(-1.0, 0.0, 0.0), Vec3::new(1.0, 0.0, 0.0));
        assert!((closest - Vec3::ZERO).length() < 0.001);
    }

    #[test]
    fn test_closest_point_on_line_endpoint() {
        let closest = closest_point_on_line(Vec3::new(1.0, 0.0, 0.0), Vec3::new(2.0, 0.0, 0.0));
        assert!((closest - Vec3::new(1.0, 0.0, 0.0)).length() < 0.001);
    }

    #[test]
    fn test_convex_sphere_intersecting() {
        // Tetrahedron convex hull
        let vertices = vec![
            Vec3::new(0.0, 1.0, 0.0),
            Vec3::new(-1.0, -0.5, -0.5),
            Vec3::new(1.0, -0.5, -0.5),
            Vec3::new(0.0, -0.5, 1.0),
        ];
        let convex = CollisionShape::convex_hull(vertices);
        let sphere = CollisionShape::sphere(0.5);

        let t1 = Transform::from_position(Vec3::ZERO);
        let t2 = Transform::from_position(Vec3::new(0.5, 0.0, 0.0));

        let result = gjk_intersection(&convex, &t1, &sphere, &t2);
        assert!(result.is_some());
    }

    #[test]
    fn test_convex_sphere_not_intersecting() {
        let vertices = vec![
            Vec3::new(0.0, 1.0, 0.0),
            Vec3::new(-1.0, -0.5, -0.5),
            Vec3::new(1.0, -0.5, -0.5),
            Vec3::new(0.0, -0.5, 1.0),
        ];
        let convex = CollisionShape::convex_hull(vertices);
        let sphere = CollisionShape::sphere(0.5);

        let t1 = Transform::from_position(Vec3::ZERO);
        let t2 = Transform::from_position(Vec3::new(5.0, 0.0, 0.0));

        let result = gjk_intersection(&convex, &t1, &sphere, &t2);
        assert!(result.is_none());
    }

    #[test]
    fn test_convex_box_intersecting() {
        let vertices = vec![
            Vec3::new(0.0, 1.0, 0.0),
            Vec3::new(-1.0, -0.5, -0.5),
            Vec3::new(1.0, -0.5, -0.5),
            Vec3::new(0.0, -0.5, 1.0),
        ];
        let convex = CollisionShape::convex_hull(vertices);
        let box_shape = CollisionShape::cube(Vec3::ONE);

        let t1 = Transform::from_position(Vec3::ZERO);
        let t2 = Transform::from_position(Vec3::new(1.0, 0.0, 0.0));

        let result = gjk_intersection(&convex, &t1, &box_shape, &t2);
        assert!(result.is_some());
    }

    #[test]
    fn test_convex_box_not_intersecting() {
        let vertices = vec![
            Vec3::new(0.0, 1.0, 0.0),
            Vec3::new(-1.0, -0.5, -0.5),
            Vec3::new(1.0, -0.5, -0.5),
            Vec3::new(0.0, -0.5, 1.0),
        ];
        let convex = CollisionShape::convex_hull(vertices);
        let box_shape = CollisionShape::cube(Vec3::ONE);

        let t1 = Transform::from_position(Vec3::ZERO);
        let t2 = Transform::from_position(Vec3::new(5.0, 0.0, 0.0));

        let result = gjk_intersection(&convex, &t1, &box_shape, &t2);
        assert!(result.is_none());
    }

    #[test]
    fn test_convex_convex_intersecting() {
        let vertices = vec![
            Vec3::new(0.0, 1.0, 0.0),
            Vec3::new(-1.0, -0.5, -0.5),
            Vec3::new(1.0, -0.5, -0.5),
            Vec3::new(0.0, -0.5, 1.0),
        ];
        let convex_a = CollisionShape::convex_hull(vertices.clone());
        let convex_b = CollisionShape::convex_hull(vertices);

        let t1 = Transform::from_position(Vec3::ZERO);
        let t2 = Transform::from_position(Vec3::new(0.5, 0.0, 0.0));

        let result = gjk_intersection(&convex_a, &t1, &convex_b, &t2);
        assert!(result.is_some());
    }

    #[test]
    fn test_convex_distance() {
        let vertices = vec![
            Vec3::new(0.0, 1.0, 0.0),
            Vec3::new(-1.0, -0.5, -0.5),
            Vec3::new(1.0, -0.5, -0.5),
            Vec3::new(0.0, -0.5, 1.0),
        ];
        let convex = CollisionShape::convex_hull(vertices);
        let sphere = CollisionShape::sphere(0.5);

        let t1 = Transform::from_position(Vec3::ZERO);
        let t2 = Transform::from_position(Vec3::new(5.0, 0.0, 0.0));

        let dist = gjk_distance(&convex, &t1, &sphere, &t2);
        // Distance should be approximately 5 - 1 (convex extent) - 0.5 (sphere radius)
        assert!(dist > 2.0);
    }

    // Mesh collider tests

    #[test]
    fn test_mesh_sphere_intersecting() {
        use crate::collision::Triangle;

        // Simple triangle mesh
        let triangles = vec![
            Triangle::new(
                Vec3::new(-1.0, -1.0, 0.0),
                Vec3::new(1.0, -1.0, 0.0),
                Vec3::new(0.0, 1.0, 0.0),
            ),
        ];
        let mesh = CollisionShape::mesh(triangles);
        let sphere = CollisionShape::sphere(0.5);

        let t1 = Transform::from_position(Vec3::ZERO);
        let t2 = Transform::from_position(Vec3::new(0.0, 0.0, 0.3)); // Close to mesh

        let result = gjk_intersection(&mesh, &t1, &sphere, &t2);
        assert!(result.is_some());
    }

    #[test]
    fn test_mesh_sphere_not_intersecting() {
        use crate::collision::Triangle;

        let triangles = vec![
            Triangle::new(
                Vec3::new(-1.0, -1.0, 0.0),
                Vec3::new(1.0, -1.0, 0.0),
                Vec3::new(0.0, 1.0, 0.0),
            ),
        ];
        let mesh = CollisionShape::mesh(triangles);
        let sphere = CollisionShape::sphere(0.5);

        let t1 = Transform::from_position(Vec3::ZERO);
        let t2 = Transform::from_position(Vec3::new(5.0, 0.0, 0.0));

        let result = gjk_intersection(&mesh, &t1, &sphere, &t2);
        assert!(result.is_none());
    }

    #[test]
    fn test_mesh_box_intersecting() {
        use crate::collision::Triangle;

        let triangles = vec![
            Triangle::new(
                Vec3::new(-1.0, -1.0, 0.0),
                Vec3::new(1.0, -1.0, 0.0),
                Vec3::new(0.0, 1.0, 0.0),
            ),
        ];
        let mesh = CollisionShape::mesh(triangles);
        let box_shape = CollisionShape::cube(Vec3::ONE);

        let t1 = Transform::from_position(Vec3::ZERO);
        let t2 = Transform::from_position(Vec3::new(0.5, 0.0, 0.0));

        let result = gjk_intersection(&mesh, &t1, &box_shape, &t2);
        assert!(result.is_some());
    }

    #[test]
    fn test_mesh_capsule_intersecting() {
        use crate::collision::Triangle;

        let triangles = vec![
            Triangle::new(
                Vec3::new(-1.0, -1.0, 0.0),
                Vec3::new(1.0, -1.0, 0.0),
                Vec3::new(0.0, 1.0, 0.0),
            ),
        ];
        let mesh = CollisionShape::mesh(triangles);
        let capsule = CollisionShape::capsule(0.5, 1.0);

        let t1 = Transform::from_position(Vec3::ZERO);
        let t2 = Transform::from_position(Vec3::new(0.0, 0.0, 0.3));

        let result = gjk_intersection(&mesh, &t1, &capsule, &t2);
        assert!(result.is_some());
    }

    #[test]
    fn test_mesh_distance() {
        use crate::collision::Triangle;

        let triangles = vec![
            Triangle::new(
                Vec3::new(-1.0, -1.0, 0.0),
                Vec3::new(1.0, -1.0, 0.0),
                Vec3::new(0.0, 1.0, 0.0),
            ),
        ];
        let mesh = CollisionShape::mesh(triangles);
        let sphere = CollisionShape::sphere(0.5);

        let t1 = Transform::from_position(Vec3::ZERO);
        let t2 = Transform::from_position(Vec3::new(5.0, 0.0, 0.0));

        let dist = gjk_distance(&mesh, &t1, &sphere, &t2);
        // Mesh extends to x=1, sphere has radius 0.5, distance should be ~3.5
        assert!(dist > 3.0);
    }
}
