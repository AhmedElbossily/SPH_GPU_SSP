// Copyright ETH Zurich, IWF

// This file is part of iwf_mfree_gpu_3d.

// iwf_mfree_gpu_3d is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
//(at your option) any later version.

// iwf_mfree_gpu_3d is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.

// You should have received a copy of the GNU General Public License
// along with mfree_iwf.  If not, see <http://www.gnu.org/licenses/>.

#include "solid_state_processes.h"

struct Point
{
	float_t x;
	float_t y;
	float_t z;
};

void generate_circular_arrangement(float_t n, float_t r, float_t z, std::vector<Point> &points)
{

	if (r == 0)
	{
		Point point;
		point.x = r;
		point.y = r;
		point.z = z;
		points.push_back(point);
	}

	for (int i = 0; i < n; i++)
	{
		double angle = 2 * M_PI * i / n;
		Point point;
		point.x = r * cos(angle);
		point.y = r * sin(angle);
		point.z = z;
		points.push_back(point);
	}
}

float_t distance(Point a, Point b)
{

	float deff_x = a.x - b.x;
	float deff_y = a.y - b.y;
	return sqrt(deff_x * deff_x + deff_y * deff_y);
}

void generate_circular_points(float_t diameter, float_t dz, float_t zz, std::vector<Point> &points)
{
	float_t r = 0.0;
	while (r < diameter / 2.0)
	{
		int n = static_cast<int>((2 * M_PI * r) / dz);
		generate_circular_arrangement(n, r, zz, points);
		r += dz;
	}
}

void generate_circular_points_hollow(float_t outer_diameter, float_t inner_diameter, float_t dz, float_t zz, std::vector<Point> &points)
{
	float_t r = 0.0;
	while (r < outer_diameter / 2.0)
	{
		if (r < inner_diameter / 2.0)
		{
			r += dz;
			continue;
		}
		int n = static_cast<int>((2 * M_PI * r) / dz);
		generate_circular_arrangement(n, r, zz, points);
		r += dz;
	}
}

void generate_grid_points(int nx, int ny, float_t dz, float_t zz, std::vector<Point> &points)
{
	for (int i = 0; i < nx; ++i)
	{
		for (int j = 0; j < ny; ++j)
		{
			float_t x = -nx / 2.0 * dz + i * dz;
			float_t y = -ny / 2.0 * dz + j * dz;
			points.push_back(Point{x, y, zz});
		}
	}
}

float_t find_lowest_point_z(const std::vector<Point> &points)
{
	if (points.empty())
	{
		throw std::runtime_error("The points vector is empty.");
	}

	Point lowest_point = points[0];
	for (const auto &point : points)
	{
		if (point.z < lowest_point.z)
		{
			lowest_point = point;
		}
	}

	return lowest_point.z;
}

particle_gpu *setup_RFSSW(int nbox, grid_base **grid)
{
	phys_constants phys = make_phys_constants();
	corr_constants corr = make_corr_constants();
	trml_constants trml_wp = make_trml_constants();
	trml_constants trml_tool = make_trml_constants();
	joco_constants joco = make_joco_constants();

	// Constants
	constexpr float_t ms = 1.0;
	global_Vsf = 10.;
	constexpr float_t dz = 0.4;
	global_dz = dz;
	constexpr float_t hdx = 1.3;

	// dimensions of the workpiece
	constexpr float_t wp_width = 25.0;
	constexpr float_t wp_length = 25.0;
	constexpr float_t wp_thickness = 5.0 + 2. * dz;
	constexpr float_t probe_diameter = 6.0;
	global_probe_raduis = probe_diameter / 2.0;
	constexpr float_t shoulder_inner_diameter = probe_diameter;
	constexpr float_t shoulder_outer_diameter = 9.0;
	global_shoulder_raduis = shoulder_outer_diameter / 2.0;
	constexpr float_t ring_inner_diameter = shoulder_outer_diameter;
	constexpr float_t ring_outer_diameter = 17.0;
	global_ring_raduis = ring_outer_diameter / 2.0;
	constexpr float_t probe_hight = 10.0;
	constexpr float_t shoulder_hight = 10.0;
	constexpr float_t ring_hight = 5.0;
	constexpr float_t back_plate_diameter = ring_outer_diameter;
	constexpr float_t back_plate_hight = 0.0;
	constexpr float_t depth = 2.6;
	int nx = static_cast<int>(wp_width / dz) + 1;
	int ny = static_cast<int>(wp_length / dz) + 1;

	// BC
	global_shoulder_velocity = -1.25 * global_Vsf;
	global_rtf = -1.25;

	global_wz = 2700 * 0.104719755 * global_Vsf;
	glm::vec3 w(0.0, 0.0, global_wz);

	// physical constants
	phys.E = 71.7e9;
	phys.nu = 0.33;
	phys.rho0 = 2830.0 * 1.0e-6 * ms;
	phys.G = phys.E / (2. * (1. + phys.nu));
	phys.K = 2.0 * phys.G * (1 + phys.nu) / (3 * (1 - 2 * phys.nu));
	phys.mass = dz * dz * dz * phys.rho0;

	// Johnson Cook Constants substrate
	joco.A = 450.821e6; //	450.821 MPa
	joco.B = 0.;		//	108.537 MPA
	joco.C = 0.027;		//	0.027
	joco.m = 0.981;		//	0.981
	joco.n = 0.;		//	0.045
	joco.Tref = 20.;	//	323
	joco.Tmelt = 630.;	//	488
	joco.eps_dot_ref = 1;
	joco.clamp_temp = 1.;

	// Thermal Constants
	trml_wp.cp = (860. * 1.0e6) / ms;					  // Heat Capacity
	trml_wp.tq = 0.9;									  // Taylor-Quinney Coefficient
	trml_wp.k = 153. * 1.0e6 * global_Vsf;				  // Thermal Conduction
	trml_wp.alpha = trml_wp.k / (phys.rho0 * trml_wp.cp); // Thermal diffusivity
	trml_wp.eta = 0.9;

	// Thermal Constants steel
	float_t steel_rho = 7850.0 * 1.0e-6 * ms;
	trml_tool.cp = (560.0 * 1.0e6) / ms;						// Heat Capacity
	trml_tool.tq = 0.9;											// Taylor-Quinney Coefficient
	trml_tool.k = 33.0 * 1.0e6 * global_Vsf;					// Thermal Conduction
	trml_tool.alpha = trml_tool.k / (phys.rho0 * trml_tool.cp); // Thermal diffusivity
	trml_tool.eta = 0.9;

	// Corrector Constants
	corr.alpha = 1.;
	corr.beta = 1.;
	corr.eta = 0.1;
	corr.xspheps = 0.5;
	corr.stresseps = 0.3;
	float_t h1 = 1. / (hdx * dz);
	float_t q = dz * h1;
	float_t fac = (M_1_PI)*h1 * h1 * h1;
	corr.wdeltap = fac * (1 - 1.5 * q * q * (1 - 0.5 * q));

	std::vector<Point> points;
	float_t zz = 0;
	/* 	while (zz < wp_thickness + shoulder_hight + back_plate_hight)
		{
			if (zz < back_plate_hight)
			{
				generate_circular_points(back_plate_diameter, dz, zz, points);
			}
			else if (zz < back_plate_hight + wp_thickness)
			{
				generate_grid_points(nx, ny, dz, zz, points);
			}
			else if (zz < back_plate_hight + wp_thickness + ring_hight)
			{
				generate_circular_points(ring_outer_diameter, dz, zz, points);
			}
			else
			{
				generate_circular_points(shoulder_outer_diameter, dz, zz, points);
			}
			zz += dz;
		}
	*/

	while (zz < wp_thickness + shoulder_hight + back_plate_hight)
	{
		if (zz < back_plate_hight)
		{
			generate_circular_points(back_plate_diameter, dz, zz, points);
		}
		else if (zz < back_plate_hight + wp_thickness)
		{
			generate_grid_points(nx, ny, dz, zz, points);
		}
		else
		{
			generate_circular_points_hollow(ring_outer_diameter, 0, dz, zz, points);
		}
		zz += dz;
	}

	int n = points.size();
	float_t lowest_point_z = find_lowest_point_z(points);

	*grid = new grid_gpu_green(n, make_float3_t(-26., -26., -1.), make_float3_t(+26., +26., +30.), hdx * dz);

	printf("calculating with %d\n", n);

	float4_t *pos = new float4_t[n];
	float4_t *vel = new float4_t[n];
	float_t *h = new float_t[n];
	float_t *rho = new float_t[n];
	float_t *T = new float_t[n];
	float_t *tool_p = new float_t[n];
	float_t *fixed = new float_t[n];

	for (int i = 0; i < n; i++)
	{
		float_t radius = sqrt(points[i].x * points[i].x + points[i].y * points[i].y);
		pos[i] = {points[i].x, points[i].y, points[i].z, 0};
		rho[i] = phys.rho0;
		h[i] = hdx * dz;
		vel[i].x = 0;
		vel[i].y = 0.;
		vel[i].z = 0.;
		T[i] = joco.Tref;
		tool_p[i] = 0.0;
		fixed[i] = 0.0;

		if (pos[i].z == lowest_point_z)
		{
			fixed[i] = 1;
		}

		// shoulder 2
		if (pos[i].z > back_plate_hight + wp_thickness && radius <= shoulder_outer_diameter / 2.0)
		{
			glm::vec3 r(pos[i].x, pos[i].y, 0.0);
			glm::vec3 v = glm::cross(w, r);
			vel[i] = {v.x, v.y, 0.0};

			tool_p[i] = 1.0;
			rho[i] = steel_rho;
			fixed[i] = 2;
		}

		// probe 3
		if (pos[i].z > back_plate_hight + wp_thickness && radius <= probe_diameter / 2.0)
		{

			glm::vec3 r(pos[i].x, pos[i].y, 0.0);
			glm::vec3 v = glm::cross(w, r);
			vel[i] = {v.x, v.y, 0.0};

			tool_p[i] = 1.0;
			rho[i] = steel_rho;
			fixed[i] = 3;
		}

		// ring 4
		if (pos[i].z > back_plate_hight + wp_thickness && radius > shoulder_outer_diameter / 2.0)
		{

			vel[i] = {0.,0., 0.0};
			tool_p[i] = 1.0;
			rho[i] = steel_rho;
			fixed[i] = 4;
		}
	
	}

	for (int i = 0; i < n; i++)
	{
		if (tool_p[i] == 0.0)
		{
			global_shoulder_contact_surface = std::max(global_shoulder_contact_surface, pos[i].z);
			global_top_surface = global_shoulder_contact_surface;
			global_probe_contact_surface = global_shoulder_contact_surface;
		}
	}

	actions_setup_corrector_constants(corr);
	actions_setup_physical_constants(phys);
	actions_setup_thermal_constants_wp(trml_wp);
	actions_setup_thermal_constants_tool(trml_tool);
	actions_setup_johnson_cook_constants(joco);

	interactions_setup_corrector_constants(corr);
	interactions_setup_physical_constants(phys);
	interactions_setup_geometry_constants(*grid);
	interactions_setup_thermal_constants_workpiece(trml_wp);
	interactions_setup_thermal_constants_tool(trml_tool);

	particle_gpu *particles = new particle_gpu(pos, vel, rho, T, h, fixed, tool_p, n);

	global_time_dt = 1.565015e-08;
	global_time_final = 0.1;

	assert(check_cuda_error());
	return particles;
}
