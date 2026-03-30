//! 2D collision detection using GJK and EPA

use glam::Vec2;
use super::{Shape2D, Transform2D};

/// Contact information from collision detection
#[derive(Debug, Clone, Copy)]
pub struct Contact2D {
    pub point_a: Vec2,
    pub point_b: Vec2,
    pub normal: Vec2,
    pub depth: f32,
}

impl Contact2D {
    /// Get average contact point.
    pub fn point(&self) -> Vec2 {
        (self.point_a + self.point_b) * 0.5
    }
}

/// GJK simplex point in Minkowski difference
#[derive(Debug, Clone, Copy)]
pub struct SimplexPoint {
    point: Vec2,
    support_a: Vec2,
    support_b: Vec2,
}

/// 2D GJK intersection test
pub fn gjk_intersects(
    shape_a: &Shape2D,
    transform_a: &Transform2D,
    shape_b: &Shape2D,
    transform_b: &Transform2D,
) -> bool {
    gjk_intersection(shape_a, transform_a, shape_b, transform_b).is_some()
}

/// 2D GJK with simplex return for EPA
pub fn gjk_intersection(
    shape_a: &Shape2D,
    transform_a: &Transform2D,
    shape_b: &Shape2D,
    transform_b: &Transform2D,
) -> Option<Vec<SimplexPoint>> {
    let dir = transform_b.position - transform_a.position;
    let dir = if dir.length_squared() < 1e-10 { Vec2::X } else { dir };
    
    let mut simplex: Vec<SimplexPoint> = Vec::with_capacity(3);
    simplex.push(support_point(shape_a, transform_a, shape_b, transform_b, dir));
    
    let mut direction = -simplex[0].point;
    
    const MAX_ITERS: usize = 32;
    for _ in 0..MAX_ITERS {
        if direction.length_squared() < 1e-10 {
            return Some(simplex);
        }
        
        let new_point = support_point(shape_a, transform_a, shape_b, transform_b, direction);
        
        if new_point.point.dot(direction) < 0.0 {
            return None;
        }
        
        simplex.push(new_point);
        
        if let Some(new_dir) = handle_simplex(&mut simplex) {
            direction = new_dir;
        } else {
            return Some(simplex);
        }
    }
    
    None
}

/// Compute distance between shapes using GJK
pub fn gjk_distance(
    shape_a: &Shape2D,
    transform_a: &Transform2D,
    shape_b: &Shape2D,
    transform_b: &Transform2D,
) -> f32 {
    let dir = transform_b.position - transform_a.position;
    let dir = if dir.length_squared() < 1e-10 { Vec2::X } else { dir };
    
    let mut simplex: Vec<SimplexPoint> = Vec::with_capacity(3);
    simplex.push(support_point(shape_a, transform_a, shape_b, transform_b, dir));
    
    let mut direction = -simplex[0].point;
    
    const MAX_ITERS: usize = 32;
    for _ in 0..MAX_ITERS {
        if direction.length_squared() < 1e-10 {
            return 0.0; // Intersection or touching
        }
        
        let new_point = support_point(shape_a, transform_a, shape_b, transform_b, direction);
        
        if new_point.point.dot(direction) < 0.0 {
            // Find closest point on simplex to origin
            return closest_to_origin(&simplex);
        }
        
        simplex.push(new_point);
        
        if let Some(new_dir) = handle_simplex(&mut simplex) {
            direction = new_dir;
        } else {
            return 0.0; // Contains origin
        }
    }
    
    closest_to_origin(&simplex)
}

/// EPA algorithm for penetration depth and normal
pub fn epa(
    mut simplex: Vec<SimplexPoint>,
    shape_a: &Shape2D,
    transform_a: &Transform2D,
    shape_b: &Shape2D,
    transform_b: &Transform2D,
) -> Option<Contact2D> {
    const EPSILON: f32 = 1e-6;
    const MAX_ITERS: usize = 32;
    
    // Ensure we have at least 3 points
    while simplex.len() < 3 {
        let dir = match simplex.len() {
            1 => Vec2::X,
            2 => {
                let edge = simplex[1].point - simplex[0].point;
                Vec2::new(-edge.y, edge.x)
            }
            _ => return None,
        };
        simplex.push(support_point(shape_a, transform_a, shape_b, transform_b, dir));
    }
    
    // Ensure CCW winding
    let cross = (simplex[1].point - simplex[0].point).perp_dot(simplex[2].point - simplex[0].point);
    if cross < 0.0 {
        simplex.swap(0, 1);
    }
    
    for _ in 0..MAX_ITERS {
        // Find closest edge to origin
        let (edge_idx, closest_dist, normal) = find_closest_edge(&simplex);
        
        // Get support in normal direction
        let new_point = support_point(shape_a, transform_a, shape_b, transform_b, normal);
        let new_dist = new_point.point.dot(normal);
        
        if new_dist - closest_dist < EPSILON {
            // Found the closest feature
            let i = edge_idx;
            let j = (i + 1) % simplex.len();
            
            // Calculate contact points
            let (point_a, point_b) = calculate_contact_points(&simplex[i], &simplex[j], normal);
            
            return Some(Contact2D {
                point_a,
                point_b,
                normal,
                depth: closest_dist,
            });
        }
        
        // Insert new point
        simplex.insert((edge_idx + 1) % (simplex.len() + 1), new_point);
    }
    
    None
}

/// Full collision detection returning contact info if colliding
pub fn collide(
    shape_a: &Shape2D,
    transform_a: &Transform2D,
    shape_b: &Shape2D,
    transform_b: &Transform2D,
) -> Option<Contact2D> {
    let simplex = gjk_intersection(shape_a, transform_a, shape_b, transform_b)?;
    epa(simplex, shape_a, transform_a, shape_b, transform_b)
}

/// Convenience function for collision detection with position and rotation.
pub fn gjk_2d(
    shape_a: &Shape2D,
    pos_a: Vec2,
    rot_a: f32,
    shape_b: &Shape2D,
    pos_b: Vec2,
    rot_b: f32,
) -> Option<Contact2D> {
    let transform_a = Transform2D::new(pos_a, rot_a);
    let transform_b = Transform2D::new(pos_b, rot_b);
    collide(shape_a, &transform_a, shape_b, &transform_b)
}

// Helper functions

fn support_point(
    shape_a: &Shape2D,
    transform_a: &Transform2D,
    shape_b: &Shape2D,
    transform_b: &Transform2D,
    dir: Vec2,
) -> SimplexPoint {
    // Transform direction to local spaces
    let dir_a = transform_a.inverse().transform_direction(dir);
    let dir_b = transform_b.inverse().transform_direction(-dir);
    
    // Get support points in local space
    let local_a = shape_a.support(dir_a);
    let local_b = shape_b.support(dir_b);
    
    // Transform to world space
    let world_a = transform_a.transform_point(local_a);
    let world_b = transform_b.transform_point(local_b);
    
    SimplexPoint {
        point: world_a - world_b,
        support_a: world_a,
        support_b: world_b,
    }
}

fn handle_simplex(simplex: &mut Vec<SimplexPoint>) -> Option<Vec2> {
    match simplex.len() {
        2 => handle_line(simplex),
        3 => handle_triangle(simplex),
        _ => Some(Vec2::X),
    }
}

fn handle_line(simplex: &mut Vec<SimplexPoint>) -> Option<Vec2> {
    let a = simplex[1].point;
    let b = simplex[0].point;
    let ab = b - a;
    let ao = -a;
    
    if ab.dot(ao) > 0.0 {
        // Origin is between a and b, get perpendicular toward origin
        let dir = Vec2::new(-ab.y, ab.x);
        if dir.dot(ao) < 0.0 {
            Some(-dir)
        } else {
            Some(dir)
        }
    } else {
        // Origin is beyond a
        simplex.remove(0);
        Some(ao)
    }
}

fn handle_triangle(simplex: &mut Vec<SimplexPoint>) -> Option<Vec2> {
    let a = simplex[2].point;
    let b = simplex[1].point;
    let c = simplex[0].point;
    
    let ab = b - a;
    let ac = c - a;
    let ao = -a;
    
    let ab_perp = triple_product(ac, ab, ab);
    let ac_perp = triple_product(ab, ac, ac);
    
    if ab_perp.dot(ao) > 0.0 {
        // Origin is outside AB edge
        simplex.remove(0); // Remove c
        Some(ab_perp)
    } else if ac_perp.dot(ao) > 0.0 {
        // Origin is outside AC edge
        simplex.remove(1); // Remove b
        Some(ac_perp)
    } else {
        // Origin is inside triangle
        None
    }
}

fn triple_product(a: Vec2, b: Vec2, c: Vec2) -> Vec2 {
    let ac = a.dot(c);
    let bc = b.dot(c);
    Vec2::new(b.x * ac - a.x * bc, b.y * ac - a.y * bc)
}

fn find_closest_edge(simplex: &[SimplexPoint]) -> (usize, f32, Vec2) {
    let mut min_dist = f32::MAX;
    let mut min_idx = 0;
    let mut min_normal = Vec2::ZERO;
    
    for i in 0..simplex.len() {
        let j = (i + 1) % simplex.len();
        let edge = simplex[j].point - simplex[i].point;
        let normal = Vec2::new(edge.y, -edge.x).normalize();
        let dist = simplex[i].point.dot(normal);
        
        if dist < min_dist {
            min_dist = dist;
            min_idx = i;
            min_normal = normal;
        }
    }
    
    (min_idx, min_dist, min_normal)
}

fn closest_to_origin(simplex: &[SimplexPoint]) -> f32 {
    match simplex.len() {
        1 => simplex[0].point.length(),
        2 => {
            let a = simplex[0].point;
            let b = simplex[1].point;
            let ab = b - a;
            let t = (-a.dot(ab) / ab.length_squared()).clamp(0.0, 1.0);
            (a + ab * t).length()
        }
        _ => 0.0, // Triangle contains origin or should have been handled
    }
}

fn calculate_contact_points(p0: &SimplexPoint, p1: &SimplexPoint, _normal: Vec2) -> (Vec2, Vec2) {
    // Simple interpolation based on closest point projection
    let edge = p1.point - p0.point;
    let t = if edge.length_squared() > 1e-10 {
        (-p0.point.dot(edge) / edge.length_squared()).clamp(0.0, 1.0)
    } else {
        0.5
    };
    
    let point_a = p0.support_a + (p1.support_a - p0.support_a) * t;
    let point_b = p0.support_b + (p1.support_b - p0.support_b) * t;
    
    (point_a, point_b)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_circle_circle_intersect() {
        let c = Shape2D::circle(1.0);
        let t1 = Transform2D::from_position(Vec2::ZERO);
        let t2 = Transform2D::from_position(Vec2::new(1.5, 0.0));
        
        assert!(gjk_intersects(&c, &t1, &c, &t2));
    }
    
    #[test]
    fn test_circle_circle_no_intersect() {
        let c = Shape2D::circle(1.0);
        let t1 = Transform2D::from_position(Vec2::ZERO);
        let t2 = Transform2D::from_position(Vec2::new(5.0, 0.0));
        
        assert!(!gjk_intersects(&c, &t1, &c, &t2));
    }
    
    #[test]
    fn test_box_box_intersect() {
        let b = Shape2D::square(1.0);
        let t1 = Transform2D::from_position(Vec2::ZERO);
        let t2 = Transform2D::from_position(Vec2::new(1.5, 0.0));
        
        assert!(gjk_intersects(&b, &t1, &b, &t2));
    }
    
    #[test]
    fn test_epa_circles() {
        let c = Shape2D::circle(1.0);
        let t1 = Transform2D::from_position(Vec2::ZERO);
        let t2 = Transform2D::from_position(Vec2::new(1.0, 0.0));
        
        if let Some(contact) = collide(&c, &t1, &c, &t2) {
            assert!(contact.depth > 0.5 && contact.depth < 1.5);
            assert!(contact.normal.x.abs() > 0.9);
        }
    }
    
    #[test]
    fn test_gjk_distance() {
        let c = Shape2D::circle(1.0);
        let t1 = Transform2D::from_position(Vec2::ZERO);
        let t2 = Transform2D::from_position(Vec2::new(5.0, 0.0));
        
        let dist = gjk_distance(&c, &t1, &c, &t2);
        assert!((dist - 3.0).abs() < 0.1);
    }
}
