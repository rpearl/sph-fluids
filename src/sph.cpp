#include "sph.h"
#include <sys/time.h>
#include <cstdio>

#include <iostream>

using namespace std;

#define PI_FLOAT				3.141592653589793f

#define SQR(x)					((x) * (x))
#define CUBE(x)					((x) * (x) * (x))
#define POW6(x)					(CUBE(x) * CUBE(x))
#define POW9(x)					(POW6(x) * CUBE(x))

#define OPEN_MP 0

#define PARTICLE_MASS           1.0f


inline float SphFluidSolver::kernel(const Vector3f &r, const float h) {
	return 315.0f / (64.0f * PI_FLOAT * POW9(h)) * CUBE(SQR(h) - dot(r, r));
}

inline Vector3f SphFluidSolver::gradient_kernel(const Vector3f &r, const float h) {
	return -945.0f / (32.0f * PI_FLOAT * POW9(h)) * SQR(SQR(h) - dot(r, r)) * r;
}

inline float SphFluidSolver::laplacian_kernel(const Vector3f &r, const float h) {
	return   945.0f / (32.0f * PI_FLOAT * POW9(h))
	       * (SQR(h) - dot(r, r)) * (7.0f * dot(r, r) - 3.0f * SQR(h));
}

inline Vector3f SphFluidSolver::gradient_pressure_kernel(const Vector3f &r, const float h) {
	if (dot(r, r) < SQR(0.001f)) {
		return Vector3f(0.0f);
	}

	return -45.0f / (PI_FLOAT * POW6(h)) * SQR(h - length(r)) * normalize(r);
}

inline float SphFluidSolver::laplacian_viscosity_kernel(const Vector3f &r, const float h) {
	return 45.0f / (PI_FLOAT * POW6(h)) * (h - length(r));
}

inline void SphFluidSolver::add_density(Particle &particle, Particle &neighbour) {
	if (particle.id > neighbour.id) {
		return;
	}

	Vector3f r = particle.position - neighbour.position;
	if (dot(r, r) > SQR(core_radius)) {
		return;
	}

    float common = kernel(r, core_radius);
    particle.density += PARTICLE_MASS * common;
	neighbour.density += PARTICLE_MASS * common;
}

void SphFluidSolver::sum_density(GridElement &grid_element, Particle &particle) {
	auto &plist = grid_element.particles;
	for (auto piter = plist.begin(); piter != plist.end(); piter++) {
		add_density(particle, particles[*piter]);
	}
}

inline void SphFluidSolver::sum_all_density(int i, int j, int k, Particle &particle) {
	for (int z = k - 1; z <= k + 1; z++) {
		for (int y = j - 1; y <= j + 1; y++) {
			for (int x = i - 1; x <= i + 1; x++) {
				if (   (x < 0) || (x >= grid_width)
					|| (y < 0) || (y >= grid_height)
					|| (z < 0) || (z >= grid_depth)) {
					continue;
				}

				sum_density(grid(x, y, z), particle);
			}
		}
	}
}

void SphFluidSolver::update_densities(int i, int j, int k) {
	GridElement &grid_element = grid(i, j, k);

	auto &plist = grid_element.particles;
	for (auto piter = plist.begin(); piter != plist.end(); piter++) {
		sum_all_density(i, j, k, particles[*piter]);
	}
}

inline void SphFluidSolver::add_forces(Particle &particle, Particle &neighbour) {
	if (particle.id >= neighbour.id) {
		return;
	}

	Vector3f r = particle.position - neighbour.position;
	if (dot(r, r) > SQR(core_radius)) {
		return;
	}

	/* Compute the pressure force. */
	Vector3f common = 0.5f * material.gas_constant
			* ((particle.density - material.rest_density) + (neighbour.density - material.rest_density))
	        * gradient_pressure_kernel(r, core_radius);
	particle.force += -PARTICLE_MASS / neighbour.density * common;
	neighbour.force -= -PARTICLE_MASS / particle.density * common;

	/* Compute the viscosity force. */
	common = material.mu * (neighbour.velocity - particle.velocity)
	         * laplacian_viscosity_kernel(r, core_radius);
	particle.force += PARTICLE_MASS / neighbour.density * common;
	neighbour.force -= PARTICLE_MASS / particle.density * common;

	/* Compute the gradient of the color field. */
	common = gradient_kernel(r, core_radius);
	particle.color_gradient += PARTICLE_MASS / neighbour.density * common;
	neighbour.color_gradient -= PARTICLE_MASS / particle.density * common;

	/* Compute the laplacian of the color field. */
	float value = laplacian_kernel(r, core_radius);
	particle.color_laplacian += PARTICLE_MASS / neighbour.density * value;
	neighbour.color_laplacian += PARTICLE_MASS / particle.density * value;
}

void SphFluidSolver::sum_forces(GridElement &grid_element, Particle &particle) {
	auto  &plist = grid_element.particles;
	for (auto piter = plist.begin(); piter != plist.end(); piter++) {
		add_forces(particle, particles[*piter]);
	}
}

void SphFluidSolver::sum_all_forces(int i, int j, int k, Particle &particle) {
	for (int z = k - 1; z <= k + 1; z++) {
		for (int y = j - 1; y <= j + 1; y++) {
			for (int x = i - 1; x <= i + 1; x++) {
				if (   (x < 0) || (x >= grid_width)
					|| (y < 0) || (y >= grid_height)
					|| (z < 0) || (z >= grid_depth)) {
					continue;
				}

				sum_forces(grid(x, y, z), particle);
			}
		}
	}
}

void SphFluidSolver::update_forces(int i, int j, int k) {
	GridElement &grid_element = grid(i, j, k);
	auto &plist = grid_element.particles;
	for (auto piter = plist.begin(); piter != plist.end(); piter++) {
		sum_all_forces(i, j, k, particles[*piter]);
	}
}

inline void SphFluidSolver::update_particle(Particle &particle) {
	if (length(particle.color_gradient) > 0.001f) {
		particle.force +=   -material.sigma * particle.color_laplacian
		                  * normalize(particle.color_gradient);
	}

	Vector3f acceleration =   particle.force / particle.density
	               - material.point_damping * particle.velocity / PARTICLE_MASS;
	particle.velocity += timestep * acceleration;

	particle.position += timestep * particle.velocity;
}

void SphFluidSolver::update_particles(int i, int j, int k) {
	GridElement &grid_element = grid(i, j, k);

	auto &plist = grid_element.particles;
	for (auto piter = plist.begin(); piter != plist.end(); piter++) {
		update_particle(particles[*piter]);
	}
}

inline void SphFluidSolver::reset_particle(Particle &particle) {
	particle.density = 0.0f;
	particle.force = Vector3f(0.0f);
	particle.color_gradient = Vector3f(0.0f);
	particle.color_laplacian = 0.0f;
}

void SphFluidSolver::reset_particles() {
	for (int k = 0; k < grid_depth; k++) {
		for (int j = 0; j < grid_height; j++) {
			for (int i = 0; i < grid_width; i++) {
				GridElement &grid_element = grid(i, j, k);

				auto &plist = grid_element.particles;
				for (auto piter = plist.begin(); piter != plist.end(); piter++) {
					reset_particle(particles[*piter]);
				}
			}
		}
	}
}

void SphFluidSolver::update_grid() {
	for (int k = 0; k < grid_depth; k++) {
		for (int j = 0; j < grid_height; j++) {
			for (int i = 0; i < grid_width; i++) {
				grid(i, j, k).particles.clear();
			}
		}
	}
	for (uint16_t i = 0; i < particle_count; i++) {
		add_to_grid(i);
	}
}

void SphFluidSolver::update_densities() {
	timeval tv1, tv2;

	gettimeofday(&tv1, NULL);

#if OPEN_MP
	#pragma omp parallel for
#endif
	for (int k = 0; k < grid_depth; k++) {
		for (int j = 0; j < grid_height; j++) {
			for (int i = 0; i < grid_width; i++) {
				update_densities(i, j, k);
			}
		}
	}

	gettimeofday(&tv2, NULL);
	int time = 1000 * (tv2.tv_sec - tv1.tv_sec) + (tv2.tv_usec - tv1.tv_usec) / 1000;
	printf("TIME[update_densities]: %dms\n", time);
}

void SphFluidSolver::update_forces() {
	timeval tv1, tv2;

	gettimeofday(&tv1, NULL);

#if OPEN_MP
	#pragma omp parallel for
#endif
	for (int k = 0; k < grid_depth; k++) {
		for (int j = 0; j < grid_height; j++) {
			for (int i = 0; i < grid_width; i++) {
				update_forces(i, j, k);
			}
		}
	}

	gettimeofday(&tv2, NULL);
	int time = 1000 * (tv2.tv_sec - tv1.tv_sec) + (tv2.tv_usec - tv1.tv_usec) / 1000;
	printf("TIME[update_forces]   : %dms\n", time);
}

void SphFluidSolver::update_particles() {
	timeval tv1, tv2;

	gettimeofday(&tv1, NULL);

#if OPEN_MP
	#pragma omp parallel for
#endif
	for (int k = 0; k < grid_depth; k++) {
		for (int j = 0; j < grid_height; j++) {
			for (int i = 0; i < grid_width; i++) {
				update_particles(i, j, k);
			}
		}
	}

	gettimeofday(&tv2, NULL);
	int time = 1000 * (tv2.tv_sec - tv1.tv_sec) + (tv2.tv_usec - tv1.tv_usec) / 1000;
	printf("TIME[update_particles]: %dms\n", time);
}

void SphFluidSolver::update(void(*inter_hook)(), void(*post_hook)()) {
	reset_particles();

    update_densities();
    update_forces();

    /* User supplied hook, e.g. for adding custom forces (gravity, ...). */
	if (inter_hook != NULL) {
		inter_hook();
	}

    update_particles();

    /* User supplied hook, e.g. for handling collisions. */
	if (post_hook != NULL) {
		post_hook();
	}

	update_grid();
}

void SphFluidSolver::init_particles(Particle *particles, int count) {
	this->particle_count = count;
	this->particles = particles;
	grid_elements = new GridElement[grid_width * grid_height * grid_depth];

	for (int x = 0; x < count; x++) {
		add_to_grid(x);
	}
}

GridElement &SphFluidSolver::grid(int i, int j, int k) {
	return grid_elements[grid_index(i, j, k)];
}

inline int SphFluidSolver::grid_index(int i, int j, int k) {
	return grid_width * (k * grid_height + j) + i;
}

inline void SphFluidSolver::add_to_grid(uint16_t idx) {
	Particle &particle = particles[idx];
	int i = (int)(particle.position.x / core_radius);
	int j = (int)(particle.position.y / core_radius);
	int k = (int)(particle.position.z / core_radius);
	grid_elements[grid_index(i, j, k)].particles.push_back(idx);
}

