//! Example: Falling boxes demonstrating basic physics.

use vortex::prelude::*;

fn main() {
    let mut world = PhysicsWorld::new();
    world.set_gravity(Vec3::new(0.0, -9.81, 0.0));

    // Add a stack of boxes
    for i in 0..5 {
        let mut body = RigidBody::new(
            CollisionShape::cube(Vec3::splat(0.5)),
            1.0,
            BodyType::Dynamic,
        );
        body.set_position(Vec3::new(0.0, 1.0 + i as f32 * 1.1, 0.0));
        world.add_body(body);
    }

    // Add a static floor
    let mut floor = RigidBody::new(
        CollisionShape::cube(Vec3::new(10.0, 0.1, 10.0)),
        1.0,
        BodyType::Static,
    );
    floor.set_position(Vec3::new(0.0, -0.1, 0.0));
    world.add_body(floor);

    println!("Simulating {} bodies...", world.body_count());

    // Simulate for 2 seconds at 60 FPS
    let dt = 1.0 / 60.0;
    for frame in 0..120 {
        world.step(dt);

        if frame % 30 == 0 {
            println!("\nFrame {}:", frame);
            for (handle, body) in world.bodies() {
                if body.is_dynamic() {
                    println!("  {:?}: pos={:?}", handle, body.position);
                }
            }
        }
    }

    println!("\nSimulation complete!");
}
