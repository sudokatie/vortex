//! Contact generation and manifold management

use glam::{Vec2, Vec3};
use crate::world::BodyHandle;

/// Single contact point between two bodies
#[derive(Debug, Clone, Copy)]
pub struct ContactPoint {
    /// Contact point in local space on body A
    pub local_a: Vec3,
    /// Contact point in local space on body B
    pub local_b: Vec3,
    /// Penetration depth (positive when overlapping)
    pub penetration: f32,
    /// Cached normal impulse for warm starting
    pub normal_impulse: f32,
    /// Cached tangent impulse for warm starting (2D friction)
    pub tangent_impulse: Vec2,
}

impl ContactPoint {
    pub fn new(local_a: Vec3, local_b: Vec3, normal: Vec3, penetration: f32) -> Self {
        let _ = normal; // Used for orientation but stored in manifold
        Self {
            local_a,
            local_b,
            penetration,
            normal_impulse: 0.0,
            tangent_impulse: Vec2::ZERO,
        }
    }
    
    /// Create with world-space points and body positions
    pub fn from_world(
        point_a: Vec3, 
        point_b: Vec3, 
        body_a_pos: Vec3, 
        body_b_pos: Vec3,
        penetration: f32,
    ) -> Self {
        Self {
            local_a: point_a - body_a_pos,
            local_b: point_b - body_b_pos,
            penetration,
            normal_impulse: 0.0,
            tangent_impulse: Vec2::ZERO,
        }
    }

    /// Get world-space point on body A
    pub fn world_point_a(&self, body_a_pos: Vec3) -> Vec3 {
        body_a_pos + self.local_a
    }

    /// Get world-space point on body B
    pub fn world_point_b(&self, body_b_pos: Vec3) -> Vec3 {
        body_b_pos + self.local_b
    }
}

/// Contact manifold - collection of contact points between two bodies
#[derive(Debug, Clone)]
pub struct ContactManifold {
    /// Body A handle
    pub body_a: BodyHandle,
    /// Body B handle
    pub body_b: BodyHandle,
    /// Contact points (max 4 for 3D)
    pub contacts: arrayvec::ArrayVec<ContactPoint, 4>,
    /// Contact normal (from A to B)
    pub normal: Vec3,
}

impl ContactManifold {
    pub fn new(body_a: BodyHandle, body_b: BodyHandle) -> Self {
        Self {
            body_a,
            body_b,
            contacts: arrayvec::ArrayVec::new(),
            normal: Vec3::ZERO,
        }
    }

    /// Add a contact point, maintaining max 4 points
    pub fn add_point(&mut self, point: ContactPoint) {
        if self.contacts.len() < 4 {
            self.contacts.push(point);
        } else {
            // Replace the point that maximizes contact area
            self.replace_worst_point(point);
        }
    }
    
    /// Set the contact normal
    pub fn set_normal(&mut self, normal: Vec3) {
        self.normal = normal;
    }

    /// Find and replace the point that contributes least to manifold area
    fn replace_worst_point(&mut self, new_point: ContactPoint) {
        let mut worst_idx = 0;
        let mut worst_area = f32::MAX;

        for i in 0..self.contacts.len() {
            // Calculate area without this point
            let area = self.area_without_point(i, new_point.local_a);
            if area < worst_area {
                worst_area = area;
                worst_idx = i;
            }
        }

        self.contacts[worst_idx] = new_point;
    }

    /// Calculate manifold area if we replaced point at idx with new_point
    fn area_without_point(&self, skip_idx: usize, new_point: Vec3) -> f32 {
        let mut pts: arrayvec::ArrayVec<Vec3, 4> = arrayvec::ArrayVec::new();
        for (i, p) in self.contacts.iter().enumerate() {
            if i != skip_idx {
                pts.push(p.local_a);
            }
        }
        pts.push(new_point);

        if pts.len() < 3 {
            return 0.0;
        }

        // Calculate area using cross products
        let mut area = 0.0;
        for i in 1..pts.len() - 1 {
            let v1 = pts[i] - pts[0];
            let v2 = pts[i + 1] - pts[0];
            area += v1.cross(v2).length() * 0.5;
        }
        area
    }

    /// Get deepest penetration point
    pub fn deepest_point(&self) -> Option<&ContactPoint> {
        self.contacts.iter().max_by(|a, b| {
            a.penetration.partial_cmp(&b.penetration).unwrap_or(std::cmp::Ordering::Equal)
        })
    }

    /// Refresh contacts - remove separating points
    pub fn refresh(&mut self, threshold: f32) {
        self.contacts.retain(|p| p.penetration > -threshold);
    }

    /// Clear cached impulses
    pub fn clear_impulses(&mut self) {
        for p in &mut self.contacts {
            p.normal_impulse = 0.0;
            p.tangent_impulse = Vec2::ZERO;
        }
    }
    
    /// Number of contact points
    pub fn len(&self) -> usize {
        self.contacts.len()
    }
    
    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.contacts.is_empty()
    }
}

/// Generate contact manifold from GJK/EPA results
pub fn generate_contacts(
    body_a: BodyHandle,
    body_b: BodyHandle,
    penetration_normal: Vec3,
    penetration_depth: f32,
    point_on_a: Vec3,
    point_on_b: Vec3,
    body_a_pos: Vec3,
    body_b_pos: Vec3,
) -> ContactManifold {
    let mut manifold = ContactManifold::new(body_a, body_b);
    manifold.set_normal(penetration_normal);
    
    let contact = ContactPoint::from_world(
        point_on_a,
        point_on_b,
        body_a_pos,
        body_b_pos,
        penetration_depth,
    );
    manifold.add_point(contact);
    
    manifold
}

/// Clip polygon against plane, returning clipped vertices
pub fn clip_polygon(vertices: &[Vec3], plane_normal: Vec3, plane_dist: f32) -> Vec<Vec3> {
    if vertices.is_empty() {
        return Vec::new();
    }

    let mut result = Vec::new();
    let n = vertices.len();

    for i in 0..n {
        let current = vertices[i];
        let next = vertices[(i + 1) % n];

        let d_current = current.dot(plane_normal) - plane_dist;
        let d_next = next.dot(plane_normal) - plane_dist;

        if d_current >= 0.0 {
            result.push(current);
        }

        // Edge crosses plane
        if (d_current > 0.0 && d_next < 0.0) || (d_current < 0.0 && d_next > 0.0) {
            let t = d_current / (d_current - d_next);
            let intersection = current + t * (next - current);
            result.push(intersection);
        }
    }

    result
}

/// Reduce manifold to at most 4 points while maintaining coverage
pub fn reduce_manifold(points: &[ContactPoint]) -> arrayvec::ArrayVec<ContactPoint, 4> {
    let mut result: arrayvec::ArrayVec<ContactPoint, 4> = arrayvec::ArrayVec::new();

    if points.len() <= 4 {
        for p in points {
            result.push(*p);
        }
        return result;
    }

    // Find deepest point
    let deepest_idx = points.iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.penetration.partial_cmp(&b.penetration).unwrap())
        .map(|(i, _)| i)
        .unwrap_or(0);
    result.push(points[deepest_idx]);

    // Find point furthest from first
    let first = points[deepest_idx].local_a;
    let second_idx = points.iter()
        .enumerate()
        .filter(|(i, _)| *i != deepest_idx)
        .max_by(|(_, a), (_, b)| {
            let da = (a.local_a - first).length_squared();
            let db = (b.local_a - first).length_squared();
            da.partial_cmp(&db).unwrap()
        })
        .map(|(i, _)| i)
        .unwrap_or(0);
    result.push(points[second_idx]);

    // Find point furthest from line (first, second)
    let second = points[second_idx].local_a;
    let line_dir = (second - first).normalize_or_zero();
    let third_idx = points.iter()
        .enumerate()
        .filter(|(i, _)| *i != deepest_idx && *i != second_idx)
        .max_by(|(_, a), (_, b)| {
            let da = point_line_dist(a.local_a, first, line_dir);
            let db = point_line_dist(b.local_a, first, line_dir);
            da.partial_cmp(&db).unwrap()
        })
        .map(|(i, _)| i);
    
    if let Some(idx) = third_idx {
        result.push(points[idx]);
    }

    // Find point furthest from triangle plane
    if result.len() >= 3 {
        let p0 = result[0].local_a;
        let p1 = result[1].local_a;
        let p2 = result[2].local_a;
        let tri_normal = (p1 - p0).cross(p2 - p0).normalize_or_zero();
        
        let fourth_idx = points.iter()
            .enumerate()
            .filter(|(i, _)| *i != deepest_idx && *i != second_idx && third_idx != Some(*i))
            .max_by(|(_, a), (_, b)| {
                let da = (a.local_a - p0).dot(tri_normal).abs();
                let db = (b.local_a - p0).dot(tri_normal).abs();
                da.partial_cmp(&db).unwrap()
            })
            .map(|(i, _)| i);
        
        if let Some(idx) = fourth_idx {
            result.push(points[idx]);
        }
    }

    result
}

fn point_line_dist(point: Vec3, line_point: Vec3, line_dir: Vec3) -> f32 {
    let to_point = point - line_point;
    let proj = to_point.dot(line_dir);
    let closest = line_point + proj * line_dir;
    (point - closest).length()
}

#[cfg(test)]
mod tests {
    use super::*;
    use slotmap::SlotMap;

    fn make_handles() -> (SlotMap<BodyHandle, ()>, BodyHandle, BodyHandle) {
        let mut map = SlotMap::with_key();
        let h1 = map.insert(());
        let h2 = map.insert(());
        (map, h1, h2)
    }

    #[test]
    fn test_contact_point_new() {
        let cp = ContactPoint::new(
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(0.9, 0.0, 0.0),
            Vec3::new(-1.0, 0.0, 0.0),
            0.1,
        );
        assert_eq!(cp.penetration, 0.1);
        assert_eq!(cp.normal_impulse, 0.0);
        assert_eq!(cp.tangent_impulse, Vec2::ZERO);
    }

    #[test]
    fn test_contact_point_from_world() {
        let cp = ContactPoint::from_world(
            Vec3::new(2.0, 0.0, 0.0),
            Vec3::new(3.0, 0.0, 0.0),
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(4.0, 0.0, 0.0),
            0.1,
        );
        assert_eq!(cp.local_a, Vec3::new(1.0, 0.0, 0.0));
        assert_eq!(cp.local_b, Vec3::new(-1.0, 0.0, 0.0));
    }

    #[test]
    fn test_manifold_new() {
        let (_map, h1, h2) = make_handles();
        let m = ContactManifold::new(h1, h2);
        assert_eq!(m.body_a, h1);
        assert_eq!(m.body_b, h2);
        assert!(m.is_empty());
    }

    #[test]
    fn test_manifold_add_point() {
        let (_map, h1, h2) = make_handles();
        let mut m = ContactManifold::new(h1, h2);
        let cp = ContactPoint::new(Vec3::ZERO, Vec3::X, Vec3::X, 0.1);
        m.add_point(cp);
        assert_eq!(m.len(), 1);
    }

    #[test]
    fn test_manifold_max_points() {
        let (_map, h1, h2) = make_handles();
        let mut m = ContactManifold::new(h1, h2);
        for i in 0..6 {
            let cp = ContactPoint::new(
                Vec3::new(i as f32, 0.0, 0.0),
                Vec3::new(i as f32, 0.1, 0.0),
                Vec3::Y,
                0.1,
            );
            m.add_point(cp);
        }
        assert_eq!(m.len(), 4);
    }

    #[test]
    fn test_manifold_deepest() {
        let (_map, h1, h2) = make_handles();
        let mut m = ContactManifold::new(h1, h2);
        m.add_point(ContactPoint::new(Vec3::ZERO, Vec3::X, Vec3::X, 0.1));
        m.add_point(ContactPoint::new(Vec3::Y, Vec3::new(1.0, 1.0, 0.0), Vec3::X, 0.5));
        let deepest = m.deepest_point().unwrap();
        assert_eq!(deepest.penetration, 0.5);
    }

    #[test]
    fn test_manifold_refresh() {
        let (_map, h1, h2) = make_handles();
        let mut m = ContactManifold::new(h1, h2);
        m.add_point(ContactPoint::new(Vec3::ZERO, Vec3::X, Vec3::X, 0.1));
        m.add_point(ContactPoint::new(Vec3::Y, Vec3::X, Vec3::X, -0.2));
        m.refresh(0.1);
        assert_eq!(m.len(), 1);
    }

    #[test]
    fn test_generate_contacts() {
        let (_map, h1, h2) = make_handles();
        let m = generate_contacts(
            h1, h2, 
            Vec3::X, 0.1, 
            Vec3::new(1.0, 0.0, 0.0), Vec3::new(0.9, 0.0, 0.0),
            Vec3::ZERO, Vec3::ZERO,
        );
        assert_eq!(m.body_a, h1);
        assert_eq!(m.body_b, h2);
        assert_eq!(m.len(), 1);
    }

    #[test]
    fn test_clip_polygon_inside() {
        let verts = vec![Vec3::ZERO, Vec3::X, Vec3::new(1.0, 1.0, 0.0), Vec3::Y];
        let result = clip_polygon(&verts, Vec3::Y, -1.0);
        assert_eq!(result.len(), 4);
    }

    #[test]
    fn test_clip_polygon_partial() {
        let verts = vec![Vec3::ZERO, Vec3::X, Vec3::new(1.0, 1.0, 0.0), Vec3::Y];
        let result = clip_polygon(&verts, Vec3::Y, 0.5);
        assert!(result.len() >= 2);
    }

    #[test]
    fn test_reduce_manifold_small() {
        let points = vec![
            ContactPoint::new(Vec3::ZERO, Vec3::X, Vec3::X, 0.1),
            ContactPoint::new(Vec3::Y, Vec3::X, Vec3::X, 0.2),
        ];
        let reduced = reduce_manifold(&points);
        assert_eq!(reduced.len(), 2);
    }
}
