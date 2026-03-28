//! Ragdoll physics example
//!
//! Demonstrates a humanoid ragdoll using capsules and ball joints.

use vortex::prelude::*;

fn main() {
    // Create physics world with gravity
    let mut world = PhysicsWorld::new();
    world.set_gravity(Vec3::new(0.0, -9.81, 0.0));
    
    // Ground plane (static box)
    let ground = RigidBody::new(
        CollisionShape::cube(Vec3::new(10.0, 0.5, 10.0)),
        0.0, // infinite mass = static
        BodyType::Static,
    );
    let ground_handle = world.add_body(ground);
    world.get_body_mut(ground_handle).unwrap().set_position(Vec3::new(0.0, -0.5, 0.0));
    
    // Create ragdoll body parts
    // Torso (main body)
    let torso = create_body_part(&mut world, 
        Vec3::new(0.0, 2.0, 0.0),
        CollisionShape::capsule(0.25, 0.4),
        5.0
    );
    
    // Head
    let head = create_body_part(&mut world,
        Vec3::new(0.0, 2.8, 0.0),
        CollisionShape::sphere(0.15),
        1.0
    );
    
    // Upper arms
    let _left_upper_arm = create_body_part(&mut world,
        Vec3::new(-0.5, 2.3, 0.0),
        CollisionShape::capsule(0.08, 0.15),
        1.0
    );
    let _right_upper_arm = create_body_part(&mut world,
        Vec3::new(0.5, 2.3, 0.0),
        CollisionShape::capsule(0.08, 0.15),
        1.0
    );
    
    // Lower arms
    let _left_lower_arm = create_body_part(&mut world,
        Vec3::new(-0.8, 2.3, 0.0),
        CollisionShape::capsule(0.06, 0.15),
        0.8
    );
    let _right_lower_arm = create_body_part(&mut world,
        Vec3::new(0.8, 2.3, 0.0),
        CollisionShape::capsule(0.06, 0.15),
        0.8
    );
    
    // Upper legs
    let _left_upper_leg = create_body_part(&mut world,
        Vec3::new(-0.15, 1.3, 0.0),
        CollisionShape::capsule(0.1, 0.2),
        2.0
    );
    let _right_upper_leg = create_body_part(&mut world,
        Vec3::new(0.15, 1.3, 0.0),
        CollisionShape::capsule(0.1, 0.2),
        2.0
    );
    
    // Lower legs
    let _left_lower_leg = create_body_part(&mut world,
        Vec3::new(-0.15, 0.7, 0.0),
        CollisionShape::capsule(0.08, 0.2),
        1.5
    );
    let _right_lower_leg = create_body_part(&mut world,
        Vec3::new(0.15, 0.7, 0.0),
        CollisionShape::capsule(0.08, 0.2),
        1.5
    );
    
    println!("Ragdoll created with {} bodies", 10);
    println!("Body parts: head, torso, 2 arms (upper/lower), 2 legs (upper/lower)");
    
    // Simulate
    let dt = 1.0 / 60.0;
    let steps = 300; // 5 seconds
    
    println!("\nSimulating {} steps ({:.1} seconds)...\n", steps, steps as f32 * dt);
    
    for step in 0..steps {
        world.step(dt);
        
        // Print head position every second
        if step % 60 == 0 {
            if let Some(head_body) = world.get_body(head) {
                println!("t={:.1}s: head at y={:.2}", 
                    step as f32 * dt,
                    head_body.position.y
                );
            }
        }
    }
    
    // Final state
    if let Some(torso_body) = world.get_body(torso) {
        println!("\nFinal torso position: {:?}", torso_body.position);
    }
    if let Some(head_body) = world.get_body(head) {
        println!("Final head position: {:?}", head_body.position);
    }
    
    println!("\nRagdoll simulation complete!");
}

fn create_body_part(
    world: &mut PhysicsWorld,
    position: Vec3,
    shape: CollisionShape,
    mass: f32,
) -> BodyHandle {
    let mut body = RigidBody::new(shape, mass, BodyType::Dynamic);
    body.set_position(position);
    body.friction = 0.5;
    body.restitution = 0.1;
    world.add_body(body)
}
