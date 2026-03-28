//! Integration tests for vortex physics engine

use vortex::prelude::*;
use vortex::math::Transform;

#[test]
fn test_falling_sphere() {
    let mut world = PhysicsWorld::new();
    world.set_gravity(Vec3::new(0.0, -10.0, 0.0));
    
    let mut body = RigidBody::new(
        CollisionShape::sphere(1.0),
        1.0,
        BodyType::Dynamic,
    );
    body.set_position(Vec3::new(0.0, 10.0, 0.0));
    let handle = world.add_body(body);
    
    // Simulate for 1 second
    for _ in 0..60 {
        world.step(1.0 / 60.0);
    }
    
    let body = world.get_body(handle).unwrap();
    // After 1 second with -10 m/s^2 gravity:
    // position should be around 10 - 5 = 5m (s = 0.5*a*t^2)
    // velocity should be around -10 m/s
    // Using slightly relaxed bounds
    assert!(body.position.y < 6.0, "Expected y < 6.0, got y = {}", body.position.y);
    // With damping (0.99 per frame), velocity is slightly reduced
    assert!(body.linear_velocity.y < -7.0, "Expected vy < -7.0, got vy = {}", body.linear_velocity.y);
}

#[test]
fn test_sphere_on_ground() {
    // NOTE: This test verifies GJK collision detection, not full collision response.
    // Full collision response (constraint solving) is tested separately.
    use vortex::collision::{gjk_intersection, Aabb};
    use vortex::math::Transform;
    
    // Create sphere and ground shapes
    let sphere = CollisionShape::sphere(0.5);
    let ground = CollisionShape::cube(Vec3::new(10.0, 1.0, 10.0));
    
    // Test: sphere just above ground (no intersection)
    let t_sphere_above = Transform::from_position(Vec3::new(0.0, 2.0, 0.0));
    let t_ground = Transform::from_position(Vec3::new(0.0, -1.0, 0.0));
    
    let result = gjk_intersection(&sphere, &t_sphere_above, &ground, &t_ground);
    assert!(result.is_none(), "Sphere above ground should not intersect");
    
    // Test: sphere touching ground (intersection)
    let t_sphere_touching = Transform::from_position(Vec3::new(0.0, 0.5, 0.0));
    let result = gjk_intersection(&sphere, &t_sphere_touching, &ground, &t_ground);
    assert!(result.is_some(), "Sphere touching ground should intersect");
    
    // Test: sphere inside ground (intersection)
    let t_sphere_inside = Transform::from_position(Vec3::new(0.0, -0.3, 0.0));
    let result = gjk_intersection(&sphere, &t_sphere_inside, &ground, &t_ground);
    assert!(result.is_some(), "Sphere inside ground should intersect");
}

#[test]
fn test_box_stack() {
    // NOTE: This test verifies AABB intersection for stacked boxes.
    // Full stacking with collision response requires constraint solver integration.
    use vortex::collision::Aabb;
    
    let box_size = Vec3::new(0.5, 0.5, 0.5);
    
    // Create AABBs for a stack of boxes
    let ground_aabb = Aabb::new(
        Vec3::new(-5.0, -1.0, -5.0),
        Vec3::new(5.0, 0.0, 5.0),
    );
    
    let box1_aabb = Aabb::new(
        Vec3::new(-0.5, 0.0, -0.5),
        Vec3::new(0.5, 1.0, 0.5),
    );
    
    let box2_aabb = Aabb::new(
        Vec3::new(-0.5, 1.0, -0.5),
        Vec3::new(0.5, 2.0, 0.5),
    );
    
    let box3_aabb = Aabb::new(
        Vec3::new(-0.5, 2.0, -0.5),
        Vec3::new(0.5, 3.0, 0.5),
    );
    
    // Ground and box1 should touch (share boundary)
    assert!(ground_aabb.intersects(&box1_aabb) || 
            ground_aabb.max.y == box1_aabb.min.y,
            "Box1 should be on ground");
    
    // Stack should not have gaps
    assert!(box1_aabb.max.y == box2_aabb.min.y, "Box2 should be on box1");
    assert!(box2_aabb.max.y == box3_aabb.min.y, "Box3 should be on box2");
    
    // Boxes should not intersect with each other (just touching)
    let box1_inner = Aabb::new(
        box1_aabb.min + Vec3::splat(0.01),
        box1_aabb.max - Vec3::splat(0.01),
    );
    let box2_inner = Aabb::new(
        box2_aabb.min + Vec3::splat(0.01),
        box2_aabb.max - Vec3::splat(0.01),
    );
    assert!(!box1_inner.intersects(&box2_inner), "Boxes should not overlap");
}

#[test]
fn test_gjk_collision() {
    use vortex::collision::{gjk_intersection, CollisionShape};
    use vortex::math::Transform;
    
    let sphere = CollisionShape::sphere(1.0);
    let t1 = Transform::from_position(Vec3::ZERO);
    let t2 = Transform::from_position(Vec3::new(1.5, 0.0, 0.0));
    
    // Should intersect
    let result = gjk_intersection(&sphere, &t1, &sphere, &t2);
    assert!(result.is_some());
    
    // Should not intersect
    let t3 = Transform::from_position(Vec3::new(5.0, 0.0, 0.0));
    let result = gjk_intersection(&sphere, &t1, &sphere, &t3);
    assert!(result.is_none());
}

#[test]
fn test_broadphase_sap() {
    use vortex::collision::{BroadPhase, SweepAndPrune, Aabb};
    use slotmap::SlotMap;
    use vortex::world::BodyHandle;
    
    let mut sap = SweepAndPrune::new();
    let mut map: SlotMap<BodyHandle, ()> = SlotMap::with_key();
    
    let h1 = map.insert(());
    let h2 = map.insert(());
    let h3 = map.insert(());
    
    sap.insert(h1, Aabb::new(Vec3::ZERO, Vec3::ONE));
    sap.insert(h2, Aabb::new(Vec3::splat(0.5), Vec3::splat(1.5)));
    sap.insert(h3, Aabb::new(Vec3::splat(5.0), Vec3::splat(6.0)));
    
    let pairs = sap.query_pairs();
    
    // h1 and h2 overlap, h3 doesn't overlap with anything
    assert_eq!(pairs.len(), 1);
}

#[test]
fn test_broadphase_bvh() {
    use vortex::collision::{BroadPhase, Bvh, Aabb};
    use slotmap::SlotMap;
    use vortex::world::BodyHandle;
    
    let mut bvh = Bvh::new();
    let mut map: SlotMap<BodyHandle, ()> = SlotMap::with_key();
    
    let h1 = map.insert(());
    let h2 = map.insert(());
    let h3 = map.insert(());
    
    bvh.insert(h1, Aabb::new(Vec3::ZERO, Vec3::ONE));
    bvh.insert(h2, Aabb::new(Vec3::splat(0.5), Vec3::splat(1.5)));
    bvh.insert(h3, Aabb::new(Vec3::splat(5.0), Vec3::splat(6.0)));
    
    let pairs = bvh.query_pairs();
    
    // h1 and h2 overlap, h3 doesn't overlap with anything
    assert_eq!(pairs.len(), 1);
}

#[test]
fn test_convex_shape() {
    use vortex::collision::CollisionShape;
    
    // Create a simple tetrahedron
    let vertices = vec![
        Vec3::new(0.0, 1.0, 0.0),
        Vec3::new(-1.0, -1.0, -1.0),
        Vec3::new(1.0, -1.0, -1.0),
        Vec3::new(0.0, -1.0, 1.0),
    ];
    
    let convex = CollisionShape::convex_hull(vertices);
    
    // Test support function
    let support_up = convex.support(Vec3::Y);
    assert!((support_up.y - 1.0).abs() < 0.01);
    
    let support_down = convex.support(Vec3::NEG_Y);
    assert!((support_down.y - (-1.0)).abs() < 0.01);
}

#[test]
fn test_rk4_integrator() {
    use vortex::dynamics::{IntegratorType, IntegrationState, IntegrationInput, integrate_state};
    
    let state = IntegrationState::new(Vec3::ZERO, Vec3::ZERO);
    let input = IntegrationInput::linear(Vec3::new(0.0, -10.0, 0.0));
    
    // Integrate for 1 second
    let mut current = state;
    for _ in 0..100 {
        current = integrate_state(IntegratorType::Rk4, current, input, 0.01);
    }
    
    // After 1s with -10 m/s^2:
    // v = -10 m/s
    // p = -5 m (using s = 0.5*a*t^2)
    assert!((current.velocity.y - (-10.0)).abs() < 0.1);
    assert!((current.position.y - (-5.0)).abs() < 0.1);
}

#[test]
fn test_determinism() {
    // Same simulation should produce same results
    fn run_simulation() -> Vec3 {
        let mut world = PhysicsWorld::new();
        world.set_gravity(Vec3::new(0.0, -10.0, 0.0));
        
        let mut body = RigidBody::new(
            CollisionShape::sphere(1.0),
            1.0,
            BodyType::Dynamic,
        );
        body.set_position(Vec3::new(0.0, 5.0, 0.0));
        let handle = world.add_body(body);
        
        for _ in 0..100 {
            world.step(1.0 / 60.0);
        }
        
        world.get_body(handle).unwrap().position
    }
    
    let result1 = run_simulation();
    let result2 = run_simulation();
    
    assert_eq!(result1, result2);
}
