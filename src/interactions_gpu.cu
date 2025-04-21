//Copyright ETH Zurich, IWF

//This file is part of iwf_mfree_gpu_3d.

//iwf_mfree_gpu_3d is free software: you can redistribute it and/or modify
//it under the terms of the GNU General Public License as published by
//the Free Software Foundation, either version 3 of the License, or
//(at your option) any later version.

//iwf_mfree_gpu_3d is distributed in the hope that it will be useful,
//but WITHOUT ANY WARRANTY; without even the implied warranty of
//MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
//GNU General Public License for more details.

//You should have received a copy of the GNU General Public License
//along with mfree_iwf.  If not, see <http://www.gnu.org/licenses/>.

#include "interactions_gpu.h"

#include <thrust/device_vector.h>

#include "eigen_solver.cuh"
#include "kernels.cuh"

//physical constants on device
__constant__ static phys_constants physics;
__constant__ static corr_constants correctors;
__constant__ static geom_constants geometry;
__constant__ static trml_constants trml;
__constant__ static trml_constants trml_tool;

//thermal constants on host
static trml_constants thermals_workpiece;
static trml_constants thermals_tool;

//is there thermal conduction in workpiece (and/or into tool?)
static bool m_thermal_workpiece = false;
static bool m_thermal_tool = false;

//texture objects for fast access to read only attributes in interactions
cudaTextureObject_t pos_tex;
cudaTextureObject_t vel_tex;
cudaTextureObject_t h_tex;
cudaTextureObject_t rho_tex;
cudaTextureObject_t p_tex;
cudaTextureObject_t T_tex;
cudaTextureObject_t tool_particle_tex;

//texture objects for fast access to hashing information
cudaTextureObject_t hashes_tex;
cudaTextureObject_t cells_start_tex;
cudaTextureObject_t cells_end_tex;

#ifdef USE_DOUBLE
template <typename T>
__inline__ __device__ T fetch_double(cudaTextureObject_t t, int i) {
    int2 v = tex1Dfetch<int2>(t, i);
    return __hiloint2double(v.y, v.x);
}

/* static __inline__ __device__ double2 fetch_double(texture<int4, 1> t, int i) {
	int4 v = tex1Dfetch(t,i);
	return make_double2(__hiloint2double(v.y, v.x), __hiloint2double(v.w, v.z));
} */
template <typename T>
__inline__ __device__ T fetch_double2(cudaTextureObject_t t, int i) {
    int4 v1 = tex1Dfetch<int4>(t, 2 * i + 0);
    int4 v2 = tex1Dfetch<int4>(t, 2 * i + 1);

    return make_double4(
        __hiloint2double(v1.y, v1.x),
        __hiloint2double(v1.w, v1.z),
        __hiloint2double(v2.y, v2.x),
        __hiloint2double(v2.w, v2.z)
    );
}
#endif
__device__ __forceinline__ void hash(int i, int j, int k, int &idx) {
    idx = i * geometry.ny * geometry.nz + j * geometry.nz + k;
}

__device__ __forceinline__ void unhash(int &i, int &j, int &k, int idx) {
    i = idx / (geometry.nz * geometry.ny);
    j = (idx - i * geometry.ny * geometry.nz) / geometry.nz;
    k = idx % geometry.nz;
}

void setup_texture_objects(particle_gpu *particles, int *cells_start, int *cells_end, int num_cell) {
    cudaResourceDesc resDesc = {};
    cudaTextureDesc texDesc = {};
    texDesc.readMode = cudaReadModeElementType;

    // Setup pos_tex
    resDesc.resType = cudaResourceTypeLinear;
    resDesc.res.linear.devPtr = particles->pos;
    resDesc.res.linear.sizeInBytes = sizeof(float_t) * particles->N * 4;
#ifdef USE_DOUBLE
    resDesc.res.linear.desc = cudaCreateChannelDesc<int4>();
#else
    resDesc.res.linear.desc = cudaCreateChannelDesc<float4>();
#endif
    cudaCreateTextureObject(&pos_tex, &resDesc, &texDesc, nullptr);

    // Setup vel_tex
    resDesc.res.linear.devPtr = particles->vel;
    resDesc.res.linear.sizeInBytes = sizeof(float_t) * particles->N * 4;
#ifdef USE_DOUBLE
    resDesc.res.linear.desc = cudaCreateChannelDesc<int4>();
#else
    resDesc.res.linear.desc = cudaCreateChannelDesc<float4>();
#endif
    cudaCreateTextureObject(&vel_tex, &resDesc, &texDesc, nullptr);

    // Setup h_tex
    resDesc.res.linear.devPtr = particles->h;
    resDesc.res.linear.sizeInBytes = sizeof(float_t) * particles->N;
#ifdef USE_DOUBLE
    resDesc.res.linear.desc = cudaCreateChannelDesc<int2>();
#else
    resDesc.res.linear.desc = cudaCreateChannelDesc<float>();
#endif
    cudaCreateTextureObject(&h_tex, &resDesc, &texDesc, nullptr);

    // Setup rho_tex
    resDesc.res.linear.devPtr = particles->rho;
    resDesc.res.linear.sizeInBytes = sizeof(float_t) * particles->N;
#ifdef USE_DOUBLE
    resDesc.res.linear.desc = cudaCreateChannelDesc<int2>();
#else
    resDesc.res.linear.desc = cudaCreateChannelDesc<float>();
#endif
    cudaCreateTextureObject(&rho_tex, &resDesc, &texDesc, nullptr);

    // Setup p_tex
    resDesc.res.linear.devPtr = particles->p;
    resDesc.res.linear.sizeInBytes = sizeof(float_t) * particles->N;
#ifdef USE_DOUBLE
    resDesc.res.linear.desc = cudaCreateChannelDesc<int2>();
#else
    resDesc.res.linear.desc = cudaCreateChannelDesc<float>();
#endif
    cudaCreateTextureObject(&p_tex, &resDesc, &texDesc, nullptr);

    // Setup T_tex
    resDesc.res.linear.devPtr = particles->T;
    resDesc.res.linear.sizeInBytes = sizeof(float_t) * particles->N;
#ifdef USE_DOUBLE
    resDesc.res.linear.desc = cudaCreateChannelDesc<int2>();
#else
    resDesc.res.linear.desc = cudaCreateChannelDesc<float>();
#endif
    cudaCreateTextureObject(&T_tex, &resDesc, &texDesc, nullptr);

    // Setup tool_particle_tex
    resDesc.res.linear.devPtr = particles->tool_particle;
    resDesc.res.linear.sizeInBytes = sizeof(float_t) * particles->N;
#ifdef USE_DOUBLE
    resDesc.res.linear.desc = cudaCreateChannelDesc<int2>();
#else
    resDesc.res.linear.desc = cudaCreateChannelDesc<float>();
#endif
    cudaCreateTextureObject(&tool_particle_tex, &resDesc, &texDesc, nullptr);

    // Setup hashes_tex
    resDesc.res.linear.devPtr = particles->hash;
    resDesc.res.linear.sizeInBytes = sizeof(int) * particles->N;
    resDesc.res.linear.desc = cudaCreateChannelDesc<int>();
    cudaCreateTextureObject(&hashes_tex, &resDesc, &texDesc, nullptr);

    // Setup cells_start_tex
    resDesc.res.linear.devPtr = cells_start;
    resDesc.res.linear.sizeInBytes = sizeof(int) * num_cell;
    resDesc.res.linear.desc = cudaCreateChannelDesc<int>();
    cudaCreateTextureObject(&cells_start_tex, &resDesc, &texDesc, nullptr);

    // Setup cells_end_tex
    resDesc.res.linear.devPtr = cells_end;
    resDesc.res.linear.sizeInBytes = sizeof(int) * num_cell;
    resDesc.res.linear.desc = cudaCreateChannelDesc<int>();
    cudaCreateTextureObject(&cells_end_tex, &resDesc, &texDesc, nullptr);
}

void cleanup_texture_objects() {
    cudaDestroyTextureObject(pos_tex);
    cudaDestroyTextureObject(vel_tex);
    cudaDestroyTextureObject(h_tex);
    cudaDestroyTextureObject(rho_tex);
    cudaDestroyTextureObject(p_tex);
    cudaDestroyTextureObject(T_tex);
    cudaDestroyTextureObject(tool_particle_tex);
    cudaDestroyTextureObject(hashes_tex);
    cudaDestroyTextureObject(cells_start_tex);
    cudaDestroyTextureObject(cells_end_tex);
}

__global__ void do_interactions_heat(cudaTextureObject_t pos_tex, cudaTextureObject_t h_tex, cudaTextureObject_t T_tex,
                                     cudaTextureObject_t tool_particle_tex, cudaTextureObject_t hashes_tex,
                                     cudaTextureObject_t cells_start_tex, cudaTextureObject_t cells_end_tex,
                                     cudaTextureObject_t rho_tex, float_t *T_t, int N, float_t alpha_wp, float_t alpha_tool) {
    unsigned int pidx = blockIdx.x * blockDim.x + threadIdx.x;
    if (pidx >= N) return;

    //load geometrical constants
    int nx = geometry.nx;
    int ny = geometry.ny;
    int nz = geometry.nz;

    //load physical constants
    float_t mass = physics.mass;

    //load particle data at pidx
    float4_t pi = texfetch4<float4_t>(pos_tex, pidx);
    float_t hi = texfetch1<float_t>(h_tex, pidx);
    float_t Ti = texfetch1<float_t>(T_tex, pidx);

    float_t is_tool_particle = texfetch1<float_t>(tool_particle_tex, pidx);
    float_t alpha = (is_tool_particle == 1.) ? alpha_tool : alpha_wp;

    //unhash and look for neighbor boxes
    int hashi = tex1Dfetch<int>(hashes_tex, pidx);
    int gi, gj, gk;
    unhash(gi, gj, gk, hashi);

    int low_i = gi - 2 < 0 ? 0 : gi - 2;
    int low_j = gj - 2 < 0 ? 0 : gj - 2;
    int low_k = gk - 2 < 0 ? 0 : gk - 2;

    int high_i = gi + 3 > nx ? nx : gi + 3;
    int high_j = gj + 3 > ny ? ny : gj + 3;
    int high_k = gk + 3 > nz ? nz : gk + 3;

    float_t T_ti = 0.;

    for (int ii = low_i; ii < high_i; ii++) {
        for (int jj = low_j; jj < high_j; jj++) {
            for (int kk = low_k; kk < high_k; kk++) {
                int idx;
                hash(ii, jj, kk, idx);

                int c_start = tex1Dfetch<int>(cells_start_tex, idx);
                int c_end = tex1Dfetch<int>(cells_end_tex, idx);

                if (c_start == 0xffffffff) continue;

                for (int iter = c_start; iter < c_end; iter++) {
                    float4_t pj = texfetch4<float4_t>(pos_tex, iter);
                    float_t Tj = texfetch1<float_t>(T_tex, iter);
                    float_t rhoj = texfetch1<float_t>(rho_tex, iter);

                    float_t w2_pse = lapl_pse(pi, pj, hi); // Laplacian by PSE-method

                    T_ti += (Tj - Ti) * w2_pse * mass / rhoj;
                }
            }
        }
    }

    T_t[pidx] = alpha * T_ti;
}

__global__ void do_interactions_monaghan(cudaTextureObject_t pos_tex, cudaTextureObject_t T_tex,cudaTextureObject_t vel_tex, cudaTextureObject_t h_tex,
                                         cudaTextureObject_t rho_tex, cudaTextureObject_t p_tex, cudaTextureObject_t tool_particle_tex,
                                         cudaTextureObject_t hashes_tex, cudaTextureObject_t cells_start_tex, cudaTextureObject_t cells_end_tex,
                                         const float_t *__restrict__ blanked, const mat3x3_t *__restrict__ S, const mat3x3_t *__restrict__ R,
                                         mat3x3_t *__restrict__ v_der, mat3x3_t *__restrict__ S_der, float_t *__restrict__ T_t,
                                         float3_t *__restrict__ pos_t, float3_t *__restrict__ vel_t, unsigned int N) {
   int pidx = blockIdx.x * blockDim.x + threadIdx.x;

	if (pidx >= N) return;
	if (blanked[pidx] == 1.) return;

	float_t is_tool_particle_i =  texfetch1<float_t>(tool_particle_tex, pidx);

	//load physical constants
	float_t mass = physics.mass;
	float_t K    = physics.K;
#ifdef Thermal_Conduction_Brookshaw
	float_t thermal_alpha = (is_tool_particle_i == 0.) ? trml.alpha : trml_tool.alpha;
#endif

	//load correction constants
	float_t wdeltap = correctors.wdeltap;
	float_t alpha   = correctors.alpha;
	float_t beta    = correctors.beta;
	float_t eta     = correctors.eta;
	float_t eps     = correctors.xspheps;

	//load geometrical constants
	int nx = geometry.nx;
	int ny = geometry.ny;
	int nz = geometry.nz;

	//load particle data at pidx
	float4_t pi   = texfetch4<float4_t>(pos_tex, pidx);
	float4_t vi   = texfetch4<float4_t>(vel_tex, pidx);

	mat3x3_t Si   = S[pidx];
	mat3x3_t Ri   = R[pidx];
	float_t  hi   = texfetch1<float_t>(h_tex,pidx);
	float_t  rhoi = texfetch1<float_t>(rho_tex,pidx);
	float_t  prsi = texfetch1<float_t>(p_tex,pidx);

	//printf("pidx: %d, rho: %f, h: %f\n", pidx, rhoi, hi);

#ifdef Thermal_Conduction_Brookshaw
	float_t Ti = 0.;
	// particle_i temperature
	if (thermal_alpha != 0.) {
		Ti = texfetch1<float_t>(T_tex,pidx);
	}
#endif

	float_t rhoi21 = 1./(rhoi*rhoi);

	//unhash and look for neighbor boxes
	int hashi = tex1Dfetch<int>(hashes_tex,pidx);
	int gi,gj,gk;
	unhash(gi, gj, gk, hashi);

	//find neighboring boxes (take care not to iterate beyond size of cell lists structure)
	int low_i  = gi-2 < 0 ? 0 : gi-2;
	int low_j  = gj-2 < 0 ? 0 : gj-2;
	int low_k  = gk-2 < 0 ? 0 : gk-2;

	int high_i = gi+3 > nx ? nx : gi+3;
	int high_j = gj+3 > ny ? ny : gj+3;
	int high_k = gk+3 > nz ? nz : gk+3;

	//init vars to be written at pidx
	mat3x3_t vi_der(0.);
	mat3x3_t Si_der(0.);
	float3_t vi_t   = make_float3_t(0.,0.,0.);
	float3_t vi_adv_t   = make_float3_t(0.,0.,0.);
	float3_t xi_t   = make_float3_t(0.,0.,0.);

#ifdef Thermal_Conduction_Brookshaw
	float_t T_lapl = 0.;							// Laplacian of temperature field
#endif

#ifdef CSPM
	mat3x3_t B(0.);

	if (is_tool_particle_i == 0) {

		//iterate over neighboring boxes
		for (int ii = low_i; ii < high_i; ii++) {
			for (int jj = low_j; jj < high_j; jj++) {
				for (int kk = low_k; kk < high_k; kk++) {
					int idx;
					hash(ii,jj,kk,idx);

					// iterate over particles contained in a neighboring box
					int c_start = tex1Dfetch<int>(cells_start_tex, idx);
					int c_end   = ttex1Dfetch<int>(cells_end_tex,   idx);

					if (c_start ==  0xffffffff) continue;

					for (int iter = c_start; iter < c_end; iter++) {

						if (blanked[iter] == 1.) {
							continue;
						}

						float_t is_tool_particle_j = texfetch1<float_t>(tool_particle_tex,iter);
						if (is_tool_particle_j != 0.) {
							continue;
						}

						float4_t pj   = texfetch4<float4_t>(pos_tex,iter);
						float_t  rhoj = texfetch1<float_t>(rho_tex,iter);

						const float_t volj = mass/rhoj;

						float4_t ww = cubic_spline(pi, pj, hi);

						float_t w_x = ww.y;
						float_t w_y = ww.z;
						float_t w_z = ww.w;

						const float_t delta_x = pi.x - pj.x;
						const float_t delta_y = pi.y - pj.y;
						const float_t delta_z = pi.z - pj.z;

						//copute CSPM / Randles Libersky Correction Matrix
						B[0][0]-= volj * delta_x * w_x;
						B[1][0]-= volj * delta_x * w_y;
						B[2][0]-= volj * delta_x * w_z;

						B[0][1]-= volj * delta_y * w_x;
						B[1][1]-= volj * delta_y * w_y;
						B[2][1]-= volj * delta_y * w_z;

						B[0][2]-= volj * delta_z * w_x;
						B[1][2]-= volj * delta_z * w_y;
						B[2][2]-= volj * delta_z * w_z;
					}
				}
			}
		}
	}

	//save invert
	mat3x3_t invB(1.);
    float_t det_B = glm::determinant(B);
    if (det_B > 1e-8) {
    	invB = glm::inverse(B);
    }
#endif

	//iterate over neighboring boxes
	for (int ii = low_i; ii < high_i; ii++) {
		for (int jj = low_j; jj < high_j; jj++) {
			for (int kk = low_k; kk < high_k; kk++) {
				int idx;
				hash(ii,jj,kk,idx);

				// iterate over particles contained in a neighboring box
				int c_start = tex1Dfetch<int>(cells_start_tex, idx);
				int c_end   = tex1Dfetch<int>(cells_end_tex,   idx);

				if (c_start ==  0xffffffff) continue;

				for (int iter = c_start; iter < c_end; iter++) {

					if (blanked[iter] == 1.) {
						continue;
					}
					//load vars at neighbor particle
					float4_t pj   = texfetch4<float4_t>(pos_tex,iter);
					float4_t vj   = texfetch4<float4_t>(vel_tex,iter);

					mat3x3_t Sj   = S[iter];
					mat3x3_t Rj   = R[iter];
					float_t  hj   = texfetch1<float_t>(h_tex,iter);
					float_t  rhoj = texfetch1<float_t>(rho_tex,iter);
					float_t  prsj = texfetch1<float_t>(p_tex,iter);
#ifdef Thermal_Conduction_Brookshaw
					float_t Tj = 0.;
					if (thermal_alpha != 0.) {
						Tj = texfetch1<float_t>(T_tex,iter);
					}
#endif
					float_t is_tool_particle_j = texfetch1<float_t>(tool_particle_tex,iter);

					float_t volj   = mass/rhoj;
					float_t rhoj21 = 1./(rhoj*rhoj);

					//compute kernel
					float4_t ww = cubic_spline(pi, pj, hi);

					//correct by CSPM matrix if def'd
#ifndef CSPM
					float_t w   = ww.x;
					float_t w_x = ww.y;
					float_t w_y = ww.z;
					float_t w_z = ww.w;
#else
					float_t w   = ww.x;
					float_t w_x = (ww.y * invB[0][0] + ww.z * invB[1][0] + ww.w * invB[2][0]);
					float_t w_y = (ww.y * invB[0][1] + ww.z * invB[1][1] + ww.w * invB[2][1]);
					float_t w_z = (ww.y * invB[0][2] + ww.z * invB[1][2] + ww.w * invB[2][2]);
#endif

					if ((is_tool_particle_i == 0.0) && (is_tool_particle_j == 0.0)) {

						//derive vel
						vi_der[0][0] += (vj.x-vi.x)*w_x*volj;
						vi_der[0][1] += (vj.x-vi.x)*w_y*volj;
						vi_der[0][2] += (vj.x-vi.x)*w_z*volj;

						vi_der[1][0] += (vj.y-vi.y)*w_x*volj;
						vi_der[1][1] += (vj.y-vi.y)*w_y*volj;
						vi_der[1][2] += (vj.y-vi.y)*w_z*volj;

						vi_der[2][0] += (vj.z-vi.z)*w_x*volj;
						vi_der[2][1] += (vj.z-vi.z)*w_y*volj;
						vi_der[2][2] += (vj.z-vi.z)*w_z*volj;

						float_t Rxx = 0.;
						float_t Ryy = 0.;
						float_t Rzz = 0.;

						float_t Rxy = 0.;
						float_t Rxz = 0.;
						float_t Ryz = 0.;

						//compute artificial stress
						if (wdeltap > 0) {
							float_t  fab = w/wdeltap;
							fab *= fab;		//to the power of 4
							fab *= fab;

							Rxx = fab*(Ri[0][0] + Rj[0][0]);
							Rxy = fab*(Ri[0][1] + Rj[0][1]);
							Ryy = fab*(Ri[1][1] + Rj[1][1]);
							Rxz = fab*(Ri[0][2] + Rj[0][2]);
							Ryz = fab*(Ri[1][2] + Rj[1][2]);
							Rzz = fab*(Ri[2][2] + Rj[2][2]);
						}

						//derive stress
						Si_der[0][0] += mass*((Si[0][0]-prsi)*rhoi21 + (Sj[0][0]-prsj)*rhoj21 + Rxx)*w_x;
						Si_der[0][1] += mass*(Si[0][1]*rhoi21 + Sj[0][1]*rhoj21 + Rxy)*w_y;
						Si_der[0][2] += mass*(Si[0][2]*rhoi21 + Sj[0][2]*rhoj21+ Rxz)*w_z;

						Si_der[1][0] += mass*(Si[1][0]*rhoi21 + Sj[1][0]*rhoj21 + Rxy)*w_x;
						Si_der[1][1] += mass*((Si[1][1]-prsi)*rhoi21 + (Sj[1][1]-prsj)*rhoj21 + Ryy)*w_y;
						Si_der[1][2] += mass*(Si[1][2]*rhoi21 + Sj[1][2]*rhoj21 + Ryz)*w_z;

						Si_der[2][0] += mass*(Si[2][0]*rhoi21 + Sj[2][0]*rhoj21 + Rxz)*w_x;
						Si_der[2][1] += mass*(Si[2][1]*rhoi21 + Sj[2][1]*rhoj21 + Ryz)*w_y;
						Si_der[2][2] += mass*((Si[2][2]-prsi)*rhoi21 + (Sj[2][2]-prsj)*rhoj21 + Rzz)*w_z;

						//artificial viscosity
						float_t xij = pi.x - pj.x;
						float_t yij = pi.y - pj.y;
						float_t zij = pi.z - pj.z;

						float_t vijx = vi.x - vj.x;
						float_t vijy = vi.y - vj.y;
						float_t vijz = vi.z - vj.z;

						float_t vijposij = vijx*xij + vijy*yij + vijz*zij;
						float_t rhoij = 0.5*(rhoi+rhoj);

						if (vijposij < 0.) {
							float_t ci   = sqrtf(K/rhoi);
							float_t cj   = sqrtf(K/rhoj);

							float_t cij = 0.5*(ci+cj);
							float_t hij = 0.5*(hi+hj);

							float_t r2ij = xij*xij + yij*yij + zij*zij;
							float_t muij = (hij*vijposij)/(r2ij + eta*eta*hij*hij);
							float_t piij = (-alpha*cij*muij + beta*muij*muij)/rhoij;

							vi_t.x += -mass*piij*w_x;
							vi_t.y += -mass*piij*w_y;
							vi_t.z += -mass*piij*w_z;
						}

						//add xsph correction
						xi_t.x += -eps*w*mass/rhoij*vijx;
						xi_t.y += -eps*w*mass/rhoij*vijy;
						xi_t.z += -eps*w*mass/rhoij*vijz;
					}

#ifdef Thermal_Conduction_Brookshaw
					//thermal, 3D Brookshaw
					if (thermal_alpha != 0.) {
						float4_t pj   = texfetch4<float4_t>(pos_tex,iter);
						float_t xij = pi.x - pj.x;
						float_t yij = pi.y - pj.y;
						float_t zij = pi.z - pj.z;
						float_t rij = sqrt(xij*xij + yij*yij + zij*zij);
						if (rij > 1e-8) {
							float_t eijx = xij/rij;
							float_t eijy = yij/rij;
							float_t eijz = zij/rij;
							float_t rij1 = 1./rij;
							T_lapl += 2.0*(mass/rhoj)*(Ti-Tj)*rij1*(eijx*w_x + eijy*w_y + eijz*w_z);
						}
					}
#endif
				}
			}
		}
	}


	//write back
	S_der[pidx] = Si_der;
	v_der[pidx] = vi_der;

	pos_t[pidx] = xi_t;
	vel_t[pidx] = vi_t;

#ifdef Thermal_Conduction_Brookshaw
	if (thermal_alpha != 0.) {
		T_t[pidx] = thermal_alpha*T_lapl;
	}
#endif
}

void interactions_monaghan(particle_gpu *particles,  int *cells_start,  int *cells_end, int num_cell) {
    setup_texture_objects(particles, cells_start, cells_end, num_cell);

    const unsigned int block_size = BLOCK_SIZE;
    dim3 dG((particles->N + block_size - 1) / block_size);
    dim3 dB(block_size);

    particle_gpu *p = particles;
    int N = p->N;

    check_cuda_error("before interactions monaghan\n");


    do_interactions_monaghan<<<dG, dB>>>(pos_tex, T_tex, vel_tex, h_tex, rho_tex, p_tex, tool_particle_tex, hashes_tex, cells_start_tex, cells_end_tex,
                                         particles->blanked, particles->S, particles->R, p->v_der, p->S_der, p->T_t, p->pos_t, p->vel_t, p->N);

    cleanup_texture_objects();

    check_cuda_error("interactions monaghan\n");
}

void interactions_heat_pse(particle_gpu *particles,  int *cells_start,  int *cells_end, int num_cell) {
    if (!m_thermal_workpiece) return;

    setup_texture_objects(particles, cells_start, cells_end, num_cell);

    const unsigned int block_size = BLOCK_SIZE;
    dim3 dG((particles->N + block_size - 1) / block_size);
    dim3 dB(block_size);

    particle_gpu *p = particles;
    int N = p->N;

    do_interactions_heat<<<dG, dB>>>(pos_tex, h_tex, T_tex, tool_particle_tex, hashes_tex, cells_start_tex, cells_end_tex, rho_tex,
                                     p->T_t, p->N, thermals_workpiece.alpha, thermals_tool.alpha);

    cleanup_texture_objects();
}

void interactions_setup_geometry_constants(grid_base *g) {
    geom_constants geometry_h;
    geometry_h.nx = g->nx();
    geometry_h.ny = g->ny();
    geometry_h.nz = g->nz();
    geometry_h.bbmin_x = g->bbmin_x();
    geometry_h.bbmin_y = g->bbmin_y();
    geometry_h.bbmin_z = g->bbmin_z();
    geometry_h.dx = g->dx();
    cudaMemcpyToSymbol(geometry, &geometry_h, sizeof(geom_constants), 0, cudaMemcpyHostToDevice);
}

void interactions_setup_physical_constants(phys_constants physics_h) {
    cudaMemcpyToSymbol(physics, &physics_h, sizeof(phys_constants), 0, cudaMemcpyHostToDevice);
    if (physics_h.mass == 0 || isnan(physics_h.mass)) {
        printf("WARNING: invalid mass set!\n");
    }
}

void interactions_setup_corrector_constants(corr_constants correctors_h) {
    cudaMemcpyToSymbol(correctors, &correctors_h, sizeof(corr_constants), 0, cudaMemcpyHostToDevice);
}

void interactions_setup_thermal_constants_workpiece(trml_constants trml_h) {
    thermals_workpiece = trml_h;
    m_thermal_workpiece = trml_h.alpha != 0.;
#if defined(Thermal_Conduction_Brookshaw) || defined(Thermal_Conduction_PSE)
    if (m_thermal_workpiece) {
        printf("considering thermal diffusion in workpiece\n");
#if !(defined(Thermal_Conduction_Brookshaw) || defined(Thermal_Conduction_Brookshaw))
        printf("warning! heat conduction constants set but no heat conduction algorithm active!");
#endif
    }
    printf("Diffusitvity workpiece: %e\n", trml_h.alpha);
#endif

    cudaMemcpyToSymbol(trml, &trml_h, sizeof(trml_constants), 0, cudaMemcpyHostToDevice);
    check_cuda_error("error copying thermal constants.\n");
}

void interactions_setup_thermal_constants_tool(trml_constants trml_h, tool_3d_gpu *tool) {
    thermals_tool = trml_h;
    m_thermal_tool = trml_h.alpha != 0.;
#if defined(Thermal_Conduction_Brookshaw) || defined(Thermal_Conduction_PSE)
    if (m_thermal_tool) {
        tool->set_thermal(true);
        printf("considering thermal diffusion from workpiece into tool\n");
#if !(defined(Thermal_Conduction_Brookshaw) || defined(Thermal_Conduction_Brookshaw))
        printf("warning! heat conduction constants set but no heat conduction algorithm active!");
#endif
    }
    printf("Diffusitvity tool: %e\n", trml_h.alpha);
#endif

    cudaMemcpyToSymbol(trml_tool, &trml_h, sizeof(trml_constants), 0, cudaMemcpyHostToDevice);
    check_cuda_error("error copying thermal constants.\n");
}

void interactions_setup_thermal_constants_tool(trml_constants trml_h) {
    thermals_tool = trml_h;
    cudaMemcpyToSymbol(trml_tool, &trml_h, sizeof(trml_constants), 0, cudaMemcpyHostToDevice);
    check_cuda_error("error copying thermal constants.\n");
}