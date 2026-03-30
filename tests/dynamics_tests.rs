//! Dynamics and rigid body tests

use glam::{Quat, Vec3};
use vortex::dynamics::*;
use vortex::collision::CollisionShape;
use vortex::constraints::{Joint, DistanceJoint, BallJoint, HingeJoint, SolverConfig};
use vortex::world::BodyHandle;
use slotmap::SlotMap;

// =============================================================================
// RigidBody Tests
// =============================================================================

#[test]
fn test_rigidbody_new_dynamic() {
    let body = RigidBody::new(
        CollisionShape::sphere(1.0),
        1.0,
        BodyType::Dynamic,
    );
    
    assert!(body.is_dynamic());
    assert!(!body.is_static());
    assert!(body.inv_mass > 0.0);
}

#[test]
fn test_rigidbody_new_static() {
    let body = RigidBody::new(
        CollisionShape::sphere(1.0),
        1.0,
        BodyType::Static,
    );
    
    assert!(body.is_static());
    assert!(!body.is_dynamic());
    assert_eq!(body.inv_mass, 0.0);
}

#[test]
fn test_rigidbody_new_kinematic() {
    let body = RigidBody::new(
        CollisionShape::sphere(1.0),
        1.0,
        BodyType::Kinematic,
    );
    
    assert!(body.is_kinematic());
    assert!(!body.is_dynamic());
}

#[test]
fn test_rigidbody_apply_force() {
    let mut body = RigidBody::new(
        CollisionShape::sphere(1.0),
        1.0,
        BodyType::Dynamic,
    );
    
    body.apply_force(Vec3::new(10.0, 0.0, 0.0));
    assert_eq!(body.force, Vec3::new(10.0, 0.0, 0.0));
    
    body.apply_force(Vec3::new(5.0, 0.0, 0.0));
    assert_eq!(body.force, Vec3::new(15.0, 0.0, 0.0));
}

#[test]
fn test_rigidbody_apply_force_static_ignored() {
    let mut body = RigidBody::new(
        CollisionShape::sphere(1.0),
        1.0,
        BodyType::Static,
    );
    
    body.apply_force(Vec3::new(10.0, 0.0, 0.0));
    assert_eq!(body.force, Vec3::ZERO);
}

#[test]
fn test_rigidbody_apply_torque() {
    let mut body = RigidBody::new(
        CollisionShape::sphere(1.0),
        1.0,
        BodyType::Dynamic,
    );
    
    body.apply_torque(Vec3::new(0.0, 5.0, 0.0));
    assert_eq!(body.torque, Vec3::new(0.0, 5.0, 0.0));
}

#[test]
fn test_rigidbody_apply_impulse() {
    let mut body = RigidBody::new(
        CollisionShape::sphere(1.0),
        1.0,
        BodyType::Dynamic,
    );
    
    body.apply_impulse(Vec3::new(10.0, 0.0, 0.0));
    assert!(body.linear_velocity.x > 0.0);
}

#[test]
fn test_rigidbody_apply_impulse_at_point() {
    let mut body = RigidBody::new(
        CollisionShape::sphere(1.0),
        1.0,
        BodyType::Dynamic,
    );
    body.position = Vec3::ZERO;
    
    // Apply impulse at offset point - should induce rotation
    body.apply_impulse_at_point(
        Vec3::new(0.0, 1.0, 0.0),
        Vec3::new(1.0, 0.0, 0.0),
    );
    
    assert!(body.linear_velocity.length() > 0.0);
    assert!(body.angular_velocity.length() > 0.0);
}

#[test]
fn test_rigidbody_clear_forces() {
    let mut body = RigidBody::new(
        CollisionShape::sphere(1.0),
        1.0,
        BodyType::Dynamic,
    );
    
    body.apply_force(Vec3::ONE);
    body.apply_torque(Vec3::ONE);
    body.clear_forces();
    
    assert_eq!(body.force, Vec3::ZERO);
    assert_eq!(body.torque, Vec3::ZERO);
}

#[test]
fn test_rigidbody_sleeping() {
    let mut body = RigidBody::new(
        CollisionShape::sphere(1.0),
        1.0,
        BodyType::Dynamic,
    );
    
    body.linear_velocity = Vec3::ONE;
    body.angular_velocity = Vec3::ONE;
    body.set_sleeping(true);
    
    assert!(body.is_sleeping);
    assert_eq!(body.linear_velocity, Vec3::ZERO);
    assert_eq!(body.angular_velocity, Vec3::ZERO);
}

#[test]
fn test_rigidbody_velocity_at_point() {
    let mut body = RigidBody::new(
        CollisionShape::sphere(1.0),
        1.0,
        BodyType::Dynamic,
    );
    body.position = Vec3::ZERO;
    body.linear_velocity = Vec3::new(1.0, 0.0, 0.0);
    body.angular_velocity = Vec3::new(0.0, 1.0, 0.0);
    
    let v = body.linear_velocity_at(Vec3::new(0.0, 0.0, 1.0));
    // Linear + angular contribution
    assert!(v.x > 1.0);
}

// =============================================================================
// Material Tests
// =============================================================================

#[test]
fn test_material_default() {
    let mat = Material::default();
    assert!(mat.friction >= 0.0);
    assert!(mat.restitution >= 0.0 && mat.restitution <= 1.0);
}

#[test]
fn test_material_rubber() {
    let mat = Material::rubber();
    assert!(mat.restitution > 0.5); // Rubber bounces
}

#[test]
fn test_material_ice() {
    let mat = Material::ice();
    assert!(mat.friction < 0.1); // Ice is slippery
}

#[test]
fn test_material_wood() {
    let mat = Material::wood();
    assert!(mat.friction > 0.3);
    assert!(mat.restitution < 0.5);
}

#[test]
fn test_material_combine() {
    let a = Material::new(0.5, 0.5);
    let b = Material::new(0.3, 0.7);
    let combined = Material::combine(&a, &b);
    
    // Should be geometric mean for friction
    assert!(combined.friction > 0.3 && combined.friction < 0.5);
}

// =============================================================================
// Integrator Tests
// =============================================================================

#[test]
fn test_integrator_euler() {
    let v = integrate_velocity(
        IntegratorType::ExplicitEuler,
        Vec3::ZERO,
        Vec3::new(10.0, 0.0, 0.0),
        1.0,
    );
    assert_eq!(v, Vec3::new(10.0, 0.0, 0.0));
}

#[test]
fn test_integrator_semi_implicit() {
    let state = IntegrationState::new(Vec3::ZERO, Vec3::ZERO);
    let input = IntegrationInput::linear(Vec3::new(0.0, -10.0, 0.0));
    
    let result = integrate_state(IntegratorType::SemiImplicitEuler, state, input, 1.0);
    
    assert_eq!(result.velocity, Vec3::new(0.0, -10.0, 0.0));
    assert_eq!(result.position, Vec3::new(0.0, -10.0, 0.0)); // Uses new velocity
}

#[test]
fn test_integrator_verlet() {
    let p = integrate_position(
        IntegratorType::Verlet,
        Vec3::ZERO,
        Vec3::ZERO,
        Vec3::new(10.0, 0.0, 0.0),
        Vec3::new(10.0, 0.0, 0.0),
        1.0,
    );
    // p + v*dt + 0.5*a*dt^2 = 0 + 0 + 5 = 5
    assert_eq!(p, Vec3::new(5.0, 0.0, 0.0));
}

#[test]
fn test_integrator_rk4() {
    let state = IntegrationState::new(Vec3::ZERO, Vec3::ZERO);
    let input = IntegrationInput::linear(Vec3::new(0.0, -10.0, 0.0));
    
    let result = integrate_state(IntegratorType::Rk4, state, input, 1.0);
    
    assert!((result.velocity.y - (-10.0)).abs() < 0.01);
    assert!((result.position.y - (-5.0)).abs() < 0.01);
}

#[test]
fn test_integration_state_with_rotation() {
    let state = IntegrationState::new(Vec3::ZERO, Vec3::ZERO)
        .with_rotation(Quat::IDENTITY, Vec3::new(0.0, 1.0, 0.0));
    
    let input = IntegrationInput::full(Vec3::ZERO, Vec3::ZERO);
    let result = integrate_state(IntegratorType::SemiImplicitEuler, state, input, 0.1);
    
    // Rotation should have changed due to angular velocity
    assert!(result.orientation.w < 1.0);
}

// =============================================================================
// Force Generator Tests
// =============================================================================

#[test]
fn test_gravity_force() {
    let gravity = Gravity::new(Vec3::new(0.0, -9.81, 0.0));
    let body = RigidBody::new(
        CollisionShape::sphere(1.0),
        2.0,
        BodyType::Dynamic,
    );
    
    let output = gravity.apply_to_body(&body);
    // F = m * g = 2 * 9.81 = 19.62
    assert!((output.force.y - (-19.62)).abs() < 0.01);
}

#[test]
fn test_spring_force() {
    let spring = Spring::new(
        Vec3::new(0.0, 5.0, 0.0), // anchor
        1.0,   // rest_length
        100.0, // stiffness
        10.0,  // damping
    );
    
    let mut body = RigidBody::new(
        CollisionShape::sphere(1.0),
        1.0,
        BodyType::Dynamic,
    );
    body.position = Vec3::ZERO; // 5 units from anchor
    
    let output = spring.apply_to_body(&body);
    // Should pull toward anchor (positive Y)
    assert!(output.force.y > 0.0);
}

#[test]
fn test_drag_force() {
    let drag = Drag::new(0.5, 0.1);
    
    let mut body = RigidBody::new(
        CollisionShape::sphere(1.0),
        1.0,
        BodyType::Dynamic,
    );
    body.linear_velocity = Vec3::new(10.0, 0.0, 0.0);
    
    let output = drag.apply_to_body(&body);
    // Drag opposes velocity
    assert!(output.force.x < 0.0);
}

#[test]
fn test_point_force() {
    let point_force = PointForce::attractor(
        Vec3::new(0.0, 10.0, 0.0), // position
        100.0, // strength
    );
    
    let mut body = RigidBody::new(
        CollisionShape::sphere(1.0),
        1.0,
        BodyType::Dynamic,
    );
    body.position = Vec3::new(0.0, 8.0, 0.0); // 2 units away
    
    let output = point_force.apply_to_body(&body);
    // Should pull toward attractor (positive Y)
    assert!(output.force.y > 0.0);
}

#[test]
fn test_force_registry() {
    let mut registry = ForceRegistry::new();
    registry.add(Gravity::new(Vec3::new(0.0, -10.0, 0.0)));
    
    let body = RigidBody::new(
        CollisionShape::sphere(1.0),
        1.0,
        BodyType::Dynamic,
    );
    
    let total = registry.apply_all_to_body(&body);
    assert!(total.force.y < 0.0);
}

// =============================================================================
// Constraint Tests
// =============================================================================

fn make_handles() -> (SlotMap<BodyHandle, ()>, BodyHandle, BodyHandle) {
    let mut map = SlotMap::with_key();
    let h1 = map.insert(());
    let h2 = map.insert(());
    (map, h1, h2)
}

#[test]
fn test_distance_joint() {
    let (_map, h1, h2) = make_handles();
    
    let joint = DistanceJoint::new(
        h1, h2,
        Vec3::ZERO, Vec3::ZERO,
        2.0, // distance
    );
    
    let (a, b) = joint.bodies();
    assert_eq!(a, h1);
    assert_eq!(b, h2);
}

#[test]
fn test_ball_joint() {
    let (_map, h1, h2) = make_handles();
    
    let joint = BallJoint::new(
        h1, h2,
        Vec3::new(1.0, 0.0, 0.0),
        Vec3::new(-1.0, 0.0, 0.0),
    );
    
    let (a, b) = joint.bodies();
    assert_eq!(a, h1);
    assert_eq!(b, h2);
}

#[test]
fn test_hinge_joint() {
    let (_map, h1, h2) = make_handles();
    
    let joint = HingeJoint::with_axis(
        h1, h2,
        Vec3::ZERO, Vec3::ZERO,
        Vec3::Y, // axis
    );
    
    let (a, b) = joint.bodies();
    assert_eq!(a, h1);
    assert_eq!(b, h2);
}

#[test]
fn test_solver_config() {
    let config = SolverConfig::default();
    assert!(config.velocity_iterations > 0);
    assert!(config.position_iterations > 0);
}
