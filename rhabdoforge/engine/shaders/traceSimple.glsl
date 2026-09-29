#ifndef TRACE_SIMPLE_GLSL
#define TRACE_SIMPLE_GLSL

// Ray tracing: single bounce with direct lighting

#include "lighting.glsl"

vec3 trace_simple(Ray r, out float primary_depth) {
    vec3 direction = 1.0 / r.inv_direction;
    HitInfo closest_hit;
    traverse_tlas(r, direction, closest_hit);

    if (!closest_hit.found) {
        primary_depth = -1.0;
        return get_sky_color(direction, true);
    }

    primary_depth = closest_hit.t;

    vec3 surface_color = get_surface_color(closest_hit);
    vec3 hit_pos = r.origin + direction * closest_hit.t;
    vec3 normal = get_surface_normal(closest_hit, direction, hit_pos);

    vec3 result = vec3(0.0);

    // Ambient
    if (enable_ambient) {
        vec3 irradiance = sh_irradiance(normal) * ambient_intensity;
        result += surface_color * irradiance;
    }

    // Direct lighting
    if (enable_direct) {
        uint simple_rng = uint(hit_pos.x * 10000.0) ^ uint(hit_pos.z * 10000.0);
        vec3 incoming_light = calculate_direct_lighting(hit_pos, normal, simple_rng);
        result += surface_color * incoming_light;
    }

    return result;
}

vec3 trace(Ray r, inout uint rng_state, out float primary_depth) {
    return trace_simple(r, primary_depth);
}

#endif // TRACE_SIMPLE_GLSL
