// =====================================================================================================
//  backend.cu  -  lid-driven cavity LES, Eulerian impulse scheme, on the GPU.  The whole solver is this file.
//
//  How to read it:
//    Part 1  settings that live on the GPU, and small helpers (array indexing, grid spacing, lid speed)
//    Part 2  GPU kernels: each one is the work for ONE face or ONE cell; the GPU runs it for all of them at once
//    Part 3  CPU side: grid, pressure solver set-up, and the time step  (start with step() near the end)
//    Part 4  the functions Python calls (python_api.py)
//
//  Arrays are flat. A cell (x, y, z) is element  x + nx*(y + ny*z)  (x changes fastest), like NumPy's [z, y, x].
//  Staggered (MAC) grid: u lives on x-faces  (nx+1, ny, nz),  v on y-faces  (nx, ny+1, nz),
//                        w on z-faces  (nx, ny, nz+1).  The faces on the walls are always 0.
//  "m" is the impulse, "u" the velocity, "phi" the gauge potential:  u = m - grad(phi).
//
//  Build: nvcc -O2 -std=c++17 --fmad=false -shared backend.cu -o cavity_solver.dll -lcublas
// =====================================================================================================
#ifdef CPU_TEST
#include "cpu_test.h"            // only for testing without a GPU (runs every kernel as a plain loop). Ignore.
#else
#include <cuda_runtime.h>
#include <cublas_v2.h>
// RUN(kernel, n, args...): start n GPU threads, each runs `kernel` once (thread number = thread_index()).
#define RUN(kernel, n, ...) kernel<<<((n)+255)/256, 256>>>(__VA_ARGS__)
#endif
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>
#include <algorithm>
#include <stdexcept>

using std::vector;
const int GMAX = 1024;           // max cells per axis
const int X = 0, Y = 1, Z = 2;   // axis numbers

// =====================================================================================================
//  PART 1.  Settings on the GPU, and helpers
// =====================================================================================================
__constant__ int NX, NY, NZ;                 // cells per axis
__constant__ int LID_SHAPED;                 // 1: lid speed falls to 0 at the lid's edges, 0: uniform lid
__device__ float FACE[3][GMAX+1];            // face positions along each axis
__device__ float CENTRE[3][GMAX];            // cell-centre positions
__device__ float WIDTH[3][GMAX];             // cell widths
__device__ float SPACING[3][GMAX+1];         // centre-to-centre distance; at a wall: half a wall cell
__device__ float LENGTH[3];                  // box size

__device__ int thread_index() { return blockIdx.x*blockDim.x + threadIdx.x; }

// flat index of u, v, w and of a cell
__device__ int iu(int x, int y, int z) { return x + (NX+1)*(y + NY*z); }
__device__ int iv(int x, int y, int z) { return x + NX*(y + (NY+1)*z); }
__device__ int iw(int x, int y, int z) { return x + NX*(y + NY*z); }
__device__ int ic(int x, int y, int z) { return x + NX*(y + NY*z); }

__device__ float width(int axis, int i)   { return WIDTH[axis][i]; }
__device__ float spacing(int axis, int i) { return SPACING[axis][i]; }

// lid speed shape at fraction (fx, fz) of the lid: [1 - s^18]^2 in x and in z, s = 2*fraction - 1
__device__ float lid_edge(float s) { float s2 = s*s, s4 = s2*s2, s8 = s4*s4, s16 = s8*s8, a = 1.0f - s16*s2; return a*a; }
__device__ float lid_shape(float fx, float fz) {
    if (!LID_SHAPED) return 1.0f;
    return lid_edge(2.0f*fx - 1.0f)*lid_edge(2.0f*fz - 1.0f);
}
__device__ float lid_at_u_face(int x, int z) { return lid_shape(FACE[X][x]/LENGTH[X], CENTRE[Z][z]/LENGTH[Z]); }
__device__ float lid_at_cell(int x, int z)   { return lid_shape(CENTRE[X][x]/LENGTH[X], CENTRE[Z][z]/LENGTH[Z]); }

// velocity component c (0 u, 1 v, 2 w) averaged to the centre of cell (x, y, z)
__device__ float centred(const float* u, const float* v, const float* w, int c, int x, int y, int z) {
    if (c == 0) return 0.5f*(u[iu(x, y, z)] + u[iu(x+1, y, z)]);
    if (c == 1) return 0.5f*(v[iv(x, y, z)] + v[iv(x, y+1, z)]);
    return 0.5f*(w[iw(x, y, z)] + w[iw(x, y, z+1)]);
}

// g[i][j] = d u_j / d x_i at the centre of cell (x, y, z). Next to a wall, a mirrored "ghost" value one wall
// cell away: -q (no-slip), or 2*lid - q under the moving lid.
__device__ void cell_gradient(const float* u, const float* v, const float* w, float lid, int x, int y, int z,
                              float g[3][3]) {
    float dxm = x > 0 ? spacing(X, x) : width(X, 0),  dxp = x+1 < NX ? spacing(X, x+1) : width(X, NX-1);
    float dym = y > 0 ? spacing(Y, y) : width(Y, 0),  dyp = y+1 < NY ? spacing(Y, y+1) : width(Y, NY-1);
    float dzm = z > 0 ? spacing(Z, z) : width(Z, 0),  dzp = z+1 < NZ ? spacing(Z, z+1) : width(Z, NZ-1);
    for (int c = 0; c < 3; c++) {
        float q = centred(u, v, w, c, x, y, z);
        float xm = x > 0    ? centred(u, v, w, c, x-1, y, z) : -q;
        float xp = x+1 < NX ? centred(u, v, w, c, x+1, y, z) : -q;
        float ym = y > 0    ? centred(u, v, w, c, x, y-1, z) : -q;
        float yp = y+1 < NY ? centred(u, v, w, c, x, y+1, z) : 2*(c == 0 ? lid*lid_at_cell(x, z) : 0) - q;
        float zp = z+1 < NZ ? centred(u, v, w, c, x, y, z+1) : -q;
        float zm = z > 0    ? centred(u, v, w, c, x, y, z-1) : -q;
        g[0][c] = (xp - xm)/(dxp + dxm);
        g[1][c] = (yp - ym)/(dyp + dym);
        g[2][c] = (zp - zm)/(dzp + dzm);
    }
    g[0][0] = (u[iu(x+1, y, z)] - u[iu(x, y, z)])/width(X, x);
    g[1][1] = (v[iv(x, y+1, z)] - v[iv(x, y, z)])/width(Y, y);
    g[2][2] = (w[iw(x, y, z+1)] - w[iw(x, y, z)])/width(Z, z);
}

// =====================================================================================================
//  PART 2.  GPU kernels
// =====================================================================================================

// ---------------------------------------------------------------- inviscid impulse term I(u, m)
// I = (Lamb vector u x omega) - grad(K),  K = u.m.  The Lamb vector is built from the vorticity on cell edges.
// Fourth order away from the walls: each edge term is a "narrow" stencil (q = 1, neighbouring nodes) plus a
// small correction from a "wide" stencil (q = 3, nodes 3 apart); next to the walls it falls back to 2nd order.

// nodes lo, hi of the q-stencil around edge e on an axis with n cells; false if it sticks out of the box
__device__ bool stencil(int e, int q, int n, int& lo, int& hi) {
    lo = e - (q+1)/2;
    hi = e + (q-1)/2;
    return lo >= 0 && hi <= n-1;
}
// distance between the two stencil nodes
__device__ float node_distance(int axis, int e, int q, int lo, int hi) {
    if (q == 1) return spacing(axis, e);
    return CENTRE[axis][hi] - CENTRE[axis][lo];
}
// value at the edge from the two stencil nodes (cell-width weighted)
__device__ float edge_value(int axis, int e, int q, int lo, int hi, float flo, float fhi) {
    if (q == 1) return (width(axis, lo)*flo + width(axis, hi)*fhi)/(2.0f*spacing(axis, e));
    return (width(axis, lo)*flo + width(axis, hi)*fhi)/(width(axis, lo) + width(axis, hi));
}
// width that the edge contributes to its face
__device__ float edge_weight_width(int axis, int e, int q, int n) {
    int lo, hi;
    if (q == 1 || !stencil(e, q, n, lo, hi)) return spacing(axis, e);
    return 0.5f*(width(axis, lo) + width(axis, hi));
}
// 0 at the wall, rising to 1 over K cells: switches the 4th-order correction on away from the walls
__device__ float wall_ramp(int e, int n, float K) {
    float d = (float)(e < n-e ? e : n-e) - 1.0f;
    return fminf(fmaxf(d/K, 0.0f), 1.0f);
}
// stencil weight: narrow 1 + s/8, wide -s/8 (4th order); narrow 1, wide 0 (2nd order)
__device__ float stencil_weight(int q, int order4, float s) {
    if (q == 1) return order4 ? 1.0f + s*0.125f : 1.0f;
    return -s*0.125f;
}

// vorticity on the three kinds of edge: oz = dv/dx - du/dy (edge along z), ox = dw/dy - dv/dz, oy = du/dz - dw/dx
__device__ float omega_z(const float* u, const float* v, int x, int y, int z, int q) {
    int xl, xh, yl, yh;
    if (!stencil(x, q, NX, xl, xh) || !stencil(y, q, NY, yl, yh)) return 0.0f;
    return (v[iv(xh, y, z)] - v[iv(xl, y, z)])/node_distance(X, x, q, xl, xh)
         - (u[iu(x, yh, z)] - u[iu(x, yl, z)])/node_distance(Y, y, q, yl, yh);
}
__device__ float omega_x(const float* v, const float* w, int x, int y, int z, int q) {
    int yl, yh, zl, zh;
    if (!stencil(y, q, NY, yl, yh) || !stencil(z, q, NZ, zl, zh)) return 0.0f;
    return (w[iw(x, yh, z)] - w[iw(x, yl, z)])/node_distance(Y, y, q, yl, yh)
         - (v[iv(x, y, zh)] - v[iv(x, y, zl)])/node_distance(Z, z, q, zl, zh);
}
__device__ float omega_y(const float* u, const float* w, int x, int y, int z, int q) {
    int xl, xh, zl, zh;
    if (!stencil(x, q, NX, xl, xh) || !stencil(z, q, NZ, zl, zh)) return 0.0f;
    return (u[iu(x, y, zh)] - u[iu(x, y, zl)])/node_distance(Z, z, q, zl, zh)
         - (w[iw(xh, y, z)] - w[iw(xl, y, z)])/node_distance(X, x, q, xl, xh);
}

// edge vorticity times edge velocity, for the six (vorticity, velocity) pairings of the Lamb vector.
// Name: lamb_<vorticity>_<axis the velocity is interpolated along>.  0 outside the box.
__device__ float lamb_z_x(const float* u, const float* v, int x, int y, int z, int q, int o4, float K) {   // oz * v
    int xl, xh;
    if (y < 0 || y > NY || !stencil(x, q, NX, xl, xh)) return 0.0f;
    float wq = stencil_weight(q, o4, wall_ramp(y, NY, K)*wall_ramp(x, NX, K));
    if (wq == 0.0f) return 0.0f;
    float o = omega_z(u, v, x, y, z, q);
    if (o == 0.0f) return 0.0f;
    return wq*o*edge_value(X, x, q, xl, xh, v[iv(xl, y, z)], v[iv(xh, y, z)]);
}
__device__ float lamb_z_y(const float* u, const float* v, int x, int y, int z, int q, int o4, float K) {   // oz * u
    int yl, yh;
    if (x < 0 || x > NX || !stencil(y, q, NY, yl, yh)) return 0.0f;
    float wq = stencil_weight(q, o4, wall_ramp(y, NY, K)*wall_ramp(x, NX, K));
    if (wq == 0.0f) return 0.0f;
    float o = omega_z(u, v, x, y, z, q);
    if (o == 0.0f) return 0.0f;
    return wq*o*edge_value(Y, y, q, yl, yh, u[iu(x, yl, z)], u[iu(x, yh, z)]);
}
__device__ float lamb_x_y(const float* v, const float* w, int x, int y, int z, int q, int o4, float K) {   // ox * w
    int yl, yh;
    if (z < 0 || z > NZ || !stencil(y, q, NY, yl, yh)) return 0.0f;
    float wq = stencil_weight(q, o4, wall_ramp(z, NZ, K)*wall_ramp(y, NY, K));
    if (wq == 0.0f) return 0.0f;
    float o = omega_x(v, w, x, y, z, q);
    if (o == 0.0f) return 0.0f;
    return wq*o*edge_value(Y, y, q, yl, yh, w[iw(x, yl, z)], w[iw(x, yh, z)]);
}
__device__ float lamb_x_z(const float* v, const float* w, int x, int y, int z, int q, int o4, float K) {   // ox * v
    int zl, zh;
    if (y < 0 || y > NY || !stencil(z, q, NZ, zl, zh)) return 0.0f;
    float wq = stencil_weight(q, o4, wall_ramp(z, NZ, K)*wall_ramp(y, NY, K));
    if (wq == 0.0f) return 0.0f;
    float o = omega_x(v, w, x, y, z, q);
    if (o == 0.0f) return 0.0f;
    return wq*o*edge_value(Z, z, q, zl, zh, v[iv(x, y, zl)], v[iv(x, y, zh)]);
}
__device__ float lamb_y_x(const float* u, const float* w, int x, int y, int z, int q, int o4, float K) {   // oy * w
    int xl, xh;
    if (z < 0 || z > NZ || !stencil(x, q, NX, xl, xh)) return 0.0f;
    float wq = stencil_weight(q, o4, wall_ramp(z, NZ, K)*wall_ramp(x, NX, K));
    if (wq == 0.0f) return 0.0f;
    float o = omega_y(u, w, x, y, z, q);
    if (o == 0.0f) return 0.0f;
    return wq*o*edge_value(X, x, q, xl, xh, w[iw(xl, y, z)], w[iw(xh, y, z)]);
}
__device__ float lamb_y_z(const float* u, const float* w, int x, int y, int z, int q, int o4, float K) {   // oy * u
    int zl, zh;
    if (x < 0 || x > NX || !stencil(z, q, NZ, zl, zh)) return 0.0f;
    float wq = stencil_weight(q, o4, wall_ramp(z, NZ, K)*wall_ramp(x, NX, K));
    if (wq == 0.0f) return 0.0f;
    float o = omega_y(u, w, x, y, z, q);
    if (o == 0.0f) return 0.0f;
    return wq*o*edge_value(Z, z, q, zl, zh, u[iu(x, y, zl)], u[iu(x, y, zh)]);
}

// K = u.m at the centre of cell (x, y, z)
__device__ float K_cell(const float* u, const float* v, const float* w,
                        const float* mx, const float* my, const float* mz, int x, int y, int z) {
    int a = iu(x, y, z), b = iv(x, y, z), c = iw(x, y, z), c1 = iw(x, y, z+1);
    return 0.5f*((u[a]*mx[a] + u[a+1]*mx[a+1]) + (v[b]*my[b] + v[b+NX]*my[b+NX]) + (w[c]*mz[c] + w[c1]*mz[c1]));
}

// one thread per u-face: x-component of I
__global__ void inviscid_x(const float* u, const float* v, const float* w, const float* mx, const float* my,
                           const float* mz, int o4, float K, float* out) {
    int i = thread_index();
    if (i >= (NX+1)*NY*NZ) return;
    int x = i % (NX+1), y = (i / (NX+1)) % NY, z = i / ((NX+1)*NY);
    if (x == 0 || x == NX) { out[i] = 0.0f; return; }
    float L = 0.0f;
    for (int q = 1; q <= (o4 ? 3 : 1); q += 2) {
        int up = (q+1)/2, down = (q-1)/2;
        float c = edge_weight_width(X, x, q, NX)/(2.0f*spacing(X, x));
        L += c*(lamb_z_x(u, v, x, y+up, z, q, o4, K) + lamb_z_x(u, v, x, y-down, z, q, o4, K));
        L -= c*(lamb_y_x(u, w, x, y, z+up, q, o4, K) + lamb_y_x(u, w, x, y, z-down, q, o4, K));
    }
    L -= (K_cell(u, v, w, mx, my, mz, x, y, z) - K_cell(u, v, w, mx, my, mz, x-1, y, z))/spacing(X, x);
    out[i] = L;
}
// one thread per v-face: y-component of I
__global__ void inviscid_y(const float* u, const float* v, const float* w, const float* mx, const float* my,
                           const float* mz, int o4, float K, float* out) {
    int i = thread_index();
    if (i >= NX*(NY+1)*NZ) return;
    int x = i % NX, y = (i / NX) % (NY+1), z = i / (NX*(NY+1));
    if (y == 0 || y == NY) { out[i] = 0.0f; return; }
    float L = 0.0f;
    for (int q = 1; q <= (o4 ? 3 : 1); q += 2) {
        int up = (q+1)/2, down = (q-1)/2;
        float c = edge_weight_width(Y, y, q, NY)/(2.0f*spacing(Y, y));
        L -= c*(lamb_z_y(u, v, x+up, y, z, q, o4, K) + lamb_z_y(u, v, x-down, y, z, q, o4, K));
        L += c*(lamb_x_y(v, w, x, y, z+up, q, o4, K) + lamb_x_y(v, w, x, y, z-down, q, o4, K));
    }
    L -= (K_cell(u, v, w, mx, my, mz, x, y, z) - K_cell(u, v, w, mx, my, mz, x, y-1, z))/spacing(Y, y);
    out[i] = L;
}
// one thread per w-face: z-component of I
__global__ void inviscid_z(const float* u, const float* v, const float* w, const float* mx, const float* my,
                           const float* mz, int o4, float K, float* out) {
    int i = thread_index();
    if (i >= NX*NY*(NZ+1)) return;
    int x = i % NX, y = (i / NX) % NY, z = i / (NX*NY);
    if (z == 0 || z == NZ) { out[i] = 0.0f; return; }
    float L = 0.0f;
    for (int q = 1; q <= (o4 ? 3 : 1); q += 2) {
        int up = (q+1)/2, down = (q-1)/2;
        float c = edge_weight_width(Z, z, q, NZ)/(2.0f*spacing(Z, z));
        L -= c*(lamb_x_z(v, w, x, y+up, z, q, o4, K) + lamb_x_z(v, w, x, y-down, z, q, o4, K));
        L += c*(lamb_y_z(u, w, x+up, y, z, q, o4, K) + lamb_y_z(u, w, x-down, y, z, q, o4, K));
    }
    L -= (K_cell(u, v, w, mx, my, mz, x, y, z) - K_cell(u, v, w, mx, my, mz, x, y, z-1))/spacing(Z, z);
    out[i] = L;
}

// ---------------------------------------------------------------- one RK3 stage for one component
// out = [a0*m0 +] b*( m + dt*( nu*Laplacian(m) + I ) ).   comp: 0 u-faces, 1 v-faces, 2 w-faces.
// Laplacian in flux form; on a wall the flux uses the velocity at the face (no-slip, or the lid speed on top).
__global__ void rk3_stage(int comp, const float* m, const float* m0, const float* vel, const float* I,
                          float nu, float lid, float dt, float a0, float b, int has_base, float* out) {
    int sx = NX + (comp == 0), sy = NY + (comp == 1), sz = NZ + (comp == 2);     // array size of this component
    int i = thread_index();
    if (i >= sx*sy*sz) return;
    int x = i % sx, y = (i / sx) % sy, z = i / (sx*sy);
    if ((comp == 0 && (x == 0 || x == NX)) || (comp == 1 && (y == 0 || y == NY)) || (comp == 2 && (z == 0 || z == NZ))) {
        out[i] = 0.0f;                                     // face on a wall: no flow through it
        return;
    }
    float c = m[i];
    float wall = vel[i];                                   // velocity at this face, for the wall fluxes
    int dy = sx, dz = sx*sy;                               // index steps in y and z
    float lap = 0.0f;
    // x direction
    if (comp == 0) {
        lap += ((m[i+1] - c)/width(X, x) - (c - m[i-1])/width(X, x-1))/spacing(X, x);
    } else {
        float east = x < sx-1 ? (m[i+1] - c)/spacing(X, x+1) : 0.0f;
        float west = x > 0    ? (c - m[i-1])/spacing(X, x)   : 0.0f;
        if (x == 0)    west = wall/spacing(X, 0);
        if (x == sx-1) east = -wall/spacing(X, NX);
        lap += (east - west)/width(X, x);
    }
    // y direction (the lid is the top wall, y = NY)
    if (comp == 1) {
        lap += ((m[i+dy] - c)/width(Y, y) - (c - m[i-dy])/width(Y, y-1))/spacing(Y, y);
    } else {
        float north = y < sy-1 ? (m[i+dy] - c)/spacing(Y, y+1) : 0.0f;
        float south = y > 0    ? (c - m[i-dy])/spacing(Y, y)   : 0.0f;
        if (y == 0) south = wall/spacing(Y, 0);
        if (y == sy-1) {
            float U = comp == 0 ? lid*lid_at_u_face(x, z) : 0.0f;
            north = (U - wall)/spacing(Y, NY);
        }
        lap += (north - south)/width(Y, y);
    }
    // z direction
    if (comp == 2) {
        lap += ((m[i+dz] - c)/width(Z, z) - (c - m[i-dz])/width(Z, z-1))/spacing(Z, z);
    } else {
        float top    = z < sz-1 ? (m[i+dz] - c)/spacing(Z, z+1) : 0.0f;
        float bottom = z > 0    ? (c - m[i-dz])/spacing(Z, z)   : 0.0f;
        if (z == 0)    bottom = wall/spacing(Z, 0);
        if (z == sz-1) top = -wall/spacing(Z, NZ);
        lap += (top - bottom)/width(Z, z);
    }
    float rhs = nu*lap + 0.0f;
    float q = c + dt*(rhs + I[i]);
    if (has_base) q = a0*m0[i] + b*q;
    out[i] = q;
}

// ---------------------------------------------------------------- projection
// rhs = volume * div(m) of every cell (double precision)
__global__ void divergence(const float* u, const float* v, const float* w, double* rhs) {
    int i = thread_index();
    if (i >= NX*NY*NZ) return;
    int x = i % NX, y = (i / NX) % NY, z = i / (NX*NY);
    double dx = width(X, x), dy = width(Y, y), dz = width(Z, z);
    rhs[i] = ((double)u[iu(x+1, y, z)] - (double)u[iu(x, y, z)])*(dy*dz)
           + ((double)v[iv(x, y+1, z)] - (double)v[iv(x, y, z)])*(dx*dz)
           + ((double)w[iw(x, y, z+1)] - (double)w[iw(x, y, z)])*(dx*dy);
}
// r = rhs - L(phi): how far phi is from solving the pressure equation (L = sum over the 6 cell faces of
// face area * (phi_neighbour - phi)/spacing; no flux through walls)
__global__ void pressure_residual(const double* phi, const double* rhs, double* r) {
    int i = thread_index();
    if (i >= NX*NY*NZ) return;
    int x = i % NX, y = (i / NX) % NY, z = i / (NX*NY);
    double dx = width(X, x), dy = width(Y, y), dz = width(Z, z), c = phi[i];
    int sy = NX, sz = NX*NY;
    double s = 0.0;
    if (x+1 < NX) s += (dy*dz)*(phi[i+1] - c)/(double)spacing(X, x+1);
    if (x > 0)    s -= (dy*dz)*(c - phi[i-1])/(double)spacing(X, x);
    if (y+1 < NY) s += (dx*dz)*(phi[i+sy] - c)/(double)spacing(Y, y+1);
    if (y > 0)    s -= (dx*dz)*(c - phi[i-sy])/(double)spacing(Y, y);
    if (z+1 < NZ) s += (dx*dy)*(phi[i+sz] - c)/(double)spacing(Z, z+1);
    if (z > 0)    s -= (dx*dy)*(c - phi[i-sz])/(double)spacing(Z, z);
    r[i] = rhs[i] - s;
}
// out = m - grad(phi) on the faces of component comp; wall faces 0
__global__ void subtract_gradient(int comp, const float* m, const double* phi, float* out) {
    int sx = NX + (comp == 0), sy = NY + (comp == 1), sz = NZ + (comp == 2);
    int i = thread_index();
    if (i >= sx*sy*sz) return;
    int x = i % sx, y = (i / sx) % sy, z = i / (sx*sy);
    double d;
    if (comp == 0)      { if (x == 0 || x == NX) { out[i] = 0.0f; return; } d = (phi[ic(x, y, z)] - phi[ic(x-1, y, z)])/(double)spacing(X, x); }
    else if (comp == 1) { if (y == 0 || y == NY) { out[i] = 0.0f; return; } d = (phi[ic(x, y, z)] - phi[ic(x, y-1, z)])/(double)spacing(Y, y); }
    else                { if (z == 0 || z == NZ) { out[i] = 0.0f; return; } d = (phi[ic(x, y, z)] - phi[ic(x, y, z-1)])/(double)spacing(Z, z); }
    out[i] = m[i] - (float)d;
}
// zero the wall faces of component comp
__global__ void zero_walls(int comp, float* f) {
    int sx = NX + (comp == 0), sy = NY + (comp == 1), sz = NZ + (comp == 2);
    int i = thread_index();
    if (i >= sx*sy*sz) return;
    int x = i % sx, y = (i / sx) % sy, z = i / (sx*sy);
    if ((comp == 0 && (x == 0 || x == NX)) || (comp == 1 && (y == 0 || y == NY)) || (comp == 2 && (z == 0 || z == NZ)))
        f[i] = 0.0f;
}

// ---------------------------------------------------------------- simple array operations
__global__ void blend(const float* m, const float* u, float r, float one_minus_r, float* out, int n) {
    int i = thread_index(); if (i >= n) return;
    out[i] = r*m[i] + one_minus_r*u[i];                    // gauge reset: m0 = r m + (1-r) u
}
__global__ void lincomb(double a, const double* x, double b, const double* y, double* out, int n) {
    int i = thread_index(); if (i >= n) return;
    out[i] = a*x[i] + b*y[i];                              // out = a x + b y
}
__global__ void scaled(double a, const double* x, double* out, int n) {
    int i = thread_index(); if (i >= n) return;
    out[i] = a*x[i];
}
__global__ void add_to(double* out, const double* x, int n) {
    int i = thread_index(); if (i >= n) return;
    out[i] = out[i] + x[i];
}
__global__ void subtract_constant(double* out, double c, int n) {
    int i = thread_index(); if (i >= n) return;
    out[i] = out[i] - c;
}
__global__ void to_float_scaled(const double* x, double s, float* out, int n) {
    int i = thread_index(); if (i >= n) return;
    out[i] = (float)(x[i]*s);
}
__global__ void multiply(float* x, const float* y, int n) {
    int i = thread_index(); if (i >= n) return;
    x[i] = x[i]*y[i];
}
__global__ void add_float_scaled(double* out, const float* x, double s, int n) {
    int i = thread_index(); if (i >= n) return;
    out[i] = out[i] + (double)x[i]*s;
}
__global__ void abs_difference(const float* a, const float* b, float* out, int n) {
    int i = thread_index(); if (i >= n) return;
    out[i] = fabsf(a[i] - b[i]);
}
__global__ void first_index_of(const float* x, float value, int n, int* first) {
    int i = thread_index(); if (i >= n) return;
    if (x[i] == value) atomicMin(first, i);
}

// ---------------------------------------------------------------- reductions (max / sum over an array)
// Each of `threads` threads combines every threads-th element; the CPU then combines the partial results.
// op: 0 max, 1 sum, 2 max |x|, 3 sum x^2.  "a != a" is true only for NaN, so NaN wins a max (blow-up check).
__device__ double combine(int op, double a, double b) {
    if (op == 0 || op == 2) return (a != a || a > b) ? a : b;
    return a + b;
}
template <class T>
__global__ void reduce(const T* x, int n, int op, int threads, double* partial) {
    int t = thread_index();
    if (t >= threads) return;
    double r = (op == 0) ? -INFINITY : 0.0;
    for (int i = t; i < n; i += threads) {
        double v = (double)x[i];
        if (op == 2) v = fabs(v);
        if (op == 3) v = v*v;
        r = combine(op, r, v);
    }
    partial[t] = r;
}

// ---------------------------------------------------------------- output
// per cell: kinetic energy, enstrophy and divergence^2, each times the cell volume
__global__ void cell_diagnostics(const float* u, const float* v, const float* w, float lid,
                                 double* ke, double* ens, double* div2) {
    int i = thread_index();
    if (i >= NX*NY*NZ) return;
    int x = i % NX, y = (i / NX) % NY, z = i / (NX*NY);
    double vol = (double)width(X, x)*(double)width(Y, y)*(double)width(Z, z);
    double a = centred(u, v, w, 0, x, y, z), b = centred(u, v, w, 1, x, y, z), c = centred(u, v, w, 2, x, y, z);
    ke[i] = 0.5*(a*a + b*b + c*c)*vol;
    float g[3][3];
    cell_gradient(u, v, w, lid, x, y, z, g);
    float ox = g[1][2] - g[2][1], oy = g[2][0] - g[0][2], oz = g[0][1] - g[1][0];
    ens[i] = (double)(ox*ox + oy*oy + oz*oz)*vol;
    double d = (double)(u[iu(x+1, y, z)] - u[iu(x, y, z)])/(double)width(X, x)
             + (double)(v[iv(x, y+1, z)] - v[iv(x, y, z)])/(double)width(Y, y)
             + (double)(w[iw(x, y, z+1)] - w[iw(x, y, z)])/(double)width(Z, z);
    div2[i] = d*d*vol;
}
// per cell: CFL rate  max|u|/dx + max|v|/dy + max|w|/dz  over the cell's faces
__global__ void cfl_rate(const float* u, const float* v, const float* w, float* rate) {
    int i = thread_index();
    if (i >= NX*NY*NZ) return;
    int x = i % NX, y = (i / NX) % NY, z = i / (NX*NY);
    rate[i] = fmaxf(fabsf(u[iu(x, y, z)]), fabsf(u[iu(x+1, y, z)]))/width(X, x)
            + fmaxf(fabsf(v[iv(x, y, z)]), fabsf(v[iv(x, y+1, z)]))/width(Y, y)
            + fmaxf(fabsf(w[iw(x, y, z)]), fabsf(w[iw(x, y, z+1)]))/width(Z, z);
}
// per cell: Q criterion (Q L^2/U^2) and a colour (w/U, or |u|/U)
__global__ void q_criterion(const float* u, const float* v, const float* w, float lid, float qscale,
                            float inv_U, int colour_w, float* q, float* colour) {
    int i = thread_index();
    if (i >= NX*NY*NZ) return;
    int x = i % NX, y = (i / NX) % NY, z = i / (NX*NY);
    float g[3][3];
    cell_gradient(u, v, w, lid, x, y, z, g);
    float s = 0.0f;
    for (int a = 0; a < 3; a++) for (int b = 0; b < 3; b++) s += g[a][b]*g[b][a];
    q[i] = -0.5f*s*qscale;
    float a = centred(u, v, w, 0, x, y, z), b = centred(u, v, w, 1, x, y, z), c = centred(u, v, w, 2, x, y, z);
    colour[i] = colour_w ? c*inv_U : sqrtf(a*a + b*b + c*c)*inv_U;
}
// the mid-span plane z = LZ/2 lies between cell layers k1 and k2 (the same layer if nz is odd)
__device__ void midspan_layers(int& k1, int& k2) { k2 = NZ/2; k1 = (NZ % 2) ? k2 : k2-1; }
// mid-span plane of u, v, w and |vorticity|: out = [u | v | w | |omega|], each nx*ny
__global__ void midplane(const float* u, const float* v, const float* w, float lid, float* out) {
    int i = thread_index();
    if (i >= NX*NY) return;
    int x = i % NX, y = i / NX, k1, k2;
    midspan_layers(k1, k2);
    float val[2][4];
    for (int s = 0; s < 2; s++) {
        int z = s ? k2 : k1;
        for (int c = 0; c < 3; c++) val[s][c] = centred(u, v, w, c, x, y, z);
        float g[3][3];
        cell_gradient(u, v, w, lid, x, y, z, g);
        float a = g[1][2] - g[2][1], b = g[2][0] - g[0][2], c = g[0][1] - g[1][0];
        val[s][3] = sqrtf(a*a + b*b + c*c);
    }
    for (int c = 0; c < 4; c++) out[c*NX*NY + i] = (k1 == k2) ? val[1][c] : 0.5f*(val[0][c] + val[1][c]);
}
// statistics: add weight * (u, v, uu, vv, ww, uv) on the mid-span plane to the running sums
__global__ void add_statistics(const float* u, const float* v, const float* w, double weight, double* sums) {
    int i = thread_index();
    if (i >= NX*NY) return;
    int x = i % NX, y = i / NX, k1, k2, n = NX*NY;
    midspan_layers(k1, k2);
    double a = 0.5*((double)centred(u, v, w, 0, x, y, k1) + (double)centred(u, v, w, 0, x, y, k2));
    double b = 0.5*((double)centred(u, v, w, 1, x, y, k1) + (double)centred(u, v, w, 1, x, y, k2));
    double c = 0.5*((double)centred(u, v, w, 2, x, y, k1) + (double)centred(u, v, w, 2, x, y, k2));
    sums[i]       += weight*a;      sums[n+i]   += weight*b;      sums[2*n+i] += weight*(a*a);
    sums[3*n+i]   += weight*(b*b);  sums[4*n+i] += weight*(c*c);  sums[5*n+i] += weight*(a*b);
}
// centreline sample on the mid-span plane: u(y) on x = L/2, v(x) on y = L/2, and u^2, v^2, w^2 (5 rows of nx)
__global__ void centreline_sample(const float* u, const float* v, const float* w, double* out) {
    int j = thread_index();
    if (j >= NX) return;
    int mid = NX/2, k1, k2;
    midspan_layers(k1, k2);
    auto mean4 = [&](int c, int xa, int ya, int xb, int yb) {      // average over 2 cells and 2 layers
        return 0.5*(0.5*((double)centred(u, v, w, c, xa, ya, k1) + (double)centred(u, v, w, c, xa, ya, k2))
                  + 0.5*((double)centred(u, v, w, c, xb, yb, k1) + (double)centred(u, v, w, c, xb, yb, k2)));
    };
    double a = mean4(0, mid-1, j, mid, j);       // u at x = L/2, height j
    double b = mean4(1, j, mid-1, j, mid);       // v at y = L/2, position j
    double c = mean4(2, mid-1, j, mid, j);       // w at x = L/2, height j
    out[j] = a; out[NX+j] = b; out[2*NX+j] = a*a; out[3*NX+j] = b*b; out[4*NX+j] = c*c;
}
// probe values: u and v at the centre of each probe cell
__global__ void probe_values(const float* u, const float* v, const int* cells, int n, float* out) {
    int p = thread_index();
    if (p >= n) return;
    int x = cells[3*p], y = cells[3*p+1], z = cells[3*p+2];
    out[2*p]   = 0.5f*(u[iu(x, y, z)] + u[iu(x+1, y, z)]);
    out[2*p+1] = 0.5f*(v[iv(x, y, z)] + v[iv(x, y+1, z)]);
}

// ---------------------------------------------------------------- kick (random smooth disturbance)
// one 3-point smoothing pass along `axis` of an (sx, sy, sz) array, wrapping around at the ends
__global__ void smooth(const double* in, double* out, int axis, int sx, int sy, int sz) {
    int i = thread_index();
    if (i >= sx*sy*sz) return;
    int x = i % sx, y = (i / sx) % sy, z = i / (sx*sy);
    int lo, hi;
    if (axis == 0)      { lo = (x+sx-1)%sx + sx*(y + sy*z);  hi = (x+1)%sx + sx*(y + sy*z); }
    else if (axis == 1) { lo = x + sx*((y+sy-1)%sy + sy*z);  hi = x + sx*((y+1)%sy + sy*z); }
    else                { lo = x + sx*(y + sy*((z+sz-1)%sz)); hi = x + sx*(y + sy*((z+1)%sz)); }
    out[i] = 0.25*in[lo] + 0.5*in[i] + 0.25*in[hi];
}
__global__ void add_noise(const float* f, const double* noise, double s, float* out, int n) {
    int i = thread_index(); if (i >= n) return;
    out[i] = f[i] + (float)(noise[i]*s);
}
__global__ void add_difference(float* m, const float* a, const float* b, int n) {
    int i = thread_index(); if (i >= n) return;
    m[i] = m[i] + (a[i] - b[i]);                           // m += a - b
}

// =====================================================================================================
//  PART 3.  CPU side
// =====================================================================================================
static std::string last_error;
static void fail(const std::string& msg) { throw std::runtime_error(msg); }
static void check(cudaError_t e) { if (e != cudaSuccess) fail(std::string("CUDA error: ") + cudaGetErrorString(e)); }

// ---------------------------------------------------------------- GPU memory
static vector<void*> all_gpu_arrays;                       // everything allocated, freed by solver_destroy
template <class T> static T* gpu_array(long long n) {
    void* p = nullptr;
    check(cudaMalloc(&p, sizeof(T)*std::max(n, 1LL)));
    check(cudaMemset(p, 0, sizeof(T)*std::max(n, 1LL)));
    all_gpu_arrays.push_back(p);
    return (T*)p;
}
template <class T> static void to_gpu(T* dst, const T* src, long long n) { check(cudaMemcpy(dst, src, sizeof(T)*n, cudaMemcpyHostToDevice)); }
template <class T> static void to_cpu(T* dst, const T* src, long long n) { check(cudaMemcpy(dst, src, sizeof(T)*n, cudaMemcpyDeviceToHost)); }
template <class T> static void gpu_copy(T* dst, const T* src, long long n) { check(cudaMemcpy(dst, src, sizeof(T)*n, cudaMemcpyDeviceToDevice)); }

// ---------------------------------------------------------------- the solver's data
struct Config {                          // must match Config in python_api.py, field by field
    int nx, ny, nz, lid_shaped, impulse_order, adaptive_dt, dt_lag, phi_extrapolate, fdm_max_passes, grid_stretch, n_probes;
    double length, u_lid, nu, lz, lid_ramp_t, reinit_r, impulse_order4_ramp, grid_dx_wall, grid_max_aspect;
    double dt_fixed, dt_max, cfl_target, diffusion_limit, fdm_rtol;
    double probe_x[8], probe_y[8];
};
struct Report {                          // must match Report in python_api.py
    long long steps;
    double time, dt, cfl, phi_rel;
    int dt_reason, phi_passes;           // dt_reason: 0 fixed, 1 DT_MAX, 2 CFL, 3 viscous
};
struct Diagnostics {                     // must match Diagnostics in python_api.py
    double kinetic_energy, enstrophy, umax, div_rms, cfl_rate, gauge_ratio, gauge_x, gauge_y;
};
struct Axis { int n; double length; vector<double> face, centre, width, spacing; };
struct History { double t, scale; double* dphi; };          // projection warm start (see guess_phi)

struct Solver {
    Config c;
    int nx, ny, nz, n_u, n_v, n_w, n_cell, size[3];   // size[comp] = number of faces of component comp
    Axis ax[3];                                        // grid per axis (X, Y, Z)
    double volume;
    float viscous_rate;                                // max over cells of nu * sum 4/(width * spacing)
    // state: velocity u[3], impulse m[3] (one array per component), potential phi
    float* u[3]; float* m[3]; double* phi;
    // work arrays
    float *m0[3], *I[3], *new_m[2][3], *new_u[2][3], *tmp_f[3], *cell_f;
    double *phi0, *phi_stage[2], *base, *guess, *rhs, *res, *hist_buf[4], *cell_d[3], *noise[2], *partial, *stats;
    int *probe_cells, *first;
    float *probe_out, *plane;
    // pressure solver: eigenvectors S and S^T per axis (as matrices), 1/eigenvalue per cell
    float *S[3], *ST[3], *inv_lambda, *t1, *t2;
    cublasHandle_t blas;
    // clock
    double t = 0, last_dt = 0, last_cfl = 0, phi_rel = 0, prev_rates[2] = {0, 0};
    long long steps = 0;
    int dt_reason = 0, phi_passes = 0;
    bool have_prev_rates = false;
    vector<History> hist;
    double lid(double time) const {
        if (c.lid_ramp_t > 0) return c.u_lid*std::min(1.0, std::max(time, 0.0)/c.lid_ramp_t);
        return c.u_lid;
    }
};

// ---------------------------------------------------------------- sums and maxima of GPU arrays
// op: 0 max, 1 sum, 2 max |x|, 3 sum x^2   (two GPU passes: n -> 65536 -> 256 partial results, then the CPU)
template <class T> static double reduce_gpu(Solver& s, const T* x, int n, int op) {
    RUN(reduce<T>, 65536, x, n, op, 65536, s.partial);
    int op2 = (op == 0 || op == 2) ? 0 : 1;
    RUN(reduce<double>, 256, s.partial, 65536, op2, 256, s.partial + 65536);
    double part[256];
    to_cpu(part, s.partial + 65536, 256);
    double r = (op2 == 0) ? -INFINITY : 0.0;
    for (int k = 0; k < 256; k++) r = (op2 == 0) ? ((r != r || r > part[k]) ? r : part[k]) : r + part[k];
    return r;
}

// ---------------------------------------------------------------- grid  (same as grid.py)
static vector<double> linspace(double a, double b, int count) {
    vector<double> v(count);
    double step = (b - a)/(count - 1);
    for (int i = 0; i < count; i++) v[i] = i*step + a;
    v[count-1] = b;
    return v;
}
// n+1 face positions on [0, length], clustered towards both walls with first cell = dx_wall (tanh stretching)
static vector<double> stretched_faces(int n, double length, double dx_wall) {
    if (dx_wall >= length/n*(1 - 1e-12)) return linspace(0.0, length, n+1);
    vector<double> xi = linspace(0.0, 1.0, n+1), f(n+1);
    auto faces = [&](double beta) {
        for (int i = 0; i <= n; i++) f[i] = length*0.5*(1.0 + std::tanh(beta*(2*xi[i] - 1))/std::tanh(beta));
    };
    double lo = 1e-8, hi = 30.0;                       // find beta by bisection
    for (int it = 0; it < 200; it++) {
        double mid = 0.5*(lo + hi);
        faces(mid);
        if (f[1] - f[0] < dx_wall) hi = mid; else lo = mid;
    }
    faces(0.5*(lo + hi));
    f[0] = 0.0; f[n] = length; f[n/2] = 0.5*length;
    return f;
}
static double cell_ratio(const vector<double>& f) {     // largest cell / smallest cell
    double big = 0, small = 1e300;
    for (size_t i = 1; i < f.size(); i++) { big = std::max(big, f[i] - f[i-1]); small = std::min(small, f[i] - f[i-1]); }
    return big/small;
}
static Axis make_axis(const Config& c, int n, double length) {
    vector<double> f;
    if (!c.grid_stretch) f = stretched_faces(n, length, 1e300);
    else {
        double d = std::min(c.grid_dx_wall, length/n);
        auto ratio = [&](double dw) { return cell_ratio(stretched_faces(n, length, dw)); };
        if (c.grid_max_aspect > 0 && ratio(d) > c.grid_max_aspect) {   // smallest wall cell within the ratio cap
            double lo = d, hi = length/n;
            for (int it = 0; it < 60; it++) {
                double mid = 0.5*(lo + hi);
                if (ratio(mid) > c.grid_max_aspect) lo = mid; else hi = mid;
            }
            d = hi;
        }
        f = stretched_faces(n, length, d);
    }
    Axis a;
    a.n = n; a.face = f; a.width.resize(n); a.centre.resize(n); a.spacing.resize(n+1);
    for (int i = 0; i < n; i++) { a.width[i] = f[i+1] - f[i]; a.centre[i] = 0.5*(f[i+1] + f[i]); }
    for (int i = 1; i < n; i++) a.spacing[i] = a.centre[i] - a.centre[i-1];
    a.spacing[0] = 0.5*a.width[0];
    a.spacing[n] = 0.5*a.width[n-1];
    a.length = f[n] - f[0];
    // round every value through float once, so the GPU (float) and the CPU (double) use identical numbers
    for (vector<double>* v : {&a.face, &a.centre, &a.width, &a.spacing})
        for (double& q : *v) q = (double)(float)q;
    return a;
}
static void upload_grid(Solver& s) {
    static float face[3][GMAX+1], centre[3][GMAX], width_[3][GMAX], spacing_[3][GMAX+1], length[3];
    for (int a = 0; a < 3; a++) {
        const Axis& g = s.ax[a];
        for (int i = 0; i <= g.n; i++) { face[a][i] = (float)g.face[i]; spacing_[a][i] = (float)g.spacing[i]; }
        for (int i = 0; i < g.n; i++)  { centre[a][i] = (float)g.centre[i]; width_[a][i] = (float)g.width[i]; }
        length[a] = (float)g.length;
    }
    check(cudaMemcpyToSymbol(FACE, face, sizeof face));
    check(cudaMemcpyToSymbol(CENTRE, centre, sizeof centre));
    check(cudaMemcpyToSymbol(WIDTH, width_, sizeof width_));
    check(cudaMemcpyToSymbol(SPACING, spacing_, sizeof spacing_));
    check(cudaMemcpyToSymbol(LENGTH, length, sizeof length));
    check(cudaMemcpyToSymbol(NX, &s.nx, sizeof(int)));
    check(cudaMemcpyToSymbol(NY, &s.ny, sizeof(int)));
    check(cudaMemcpyToSymbol(NZ, &s.nz, sizeof(int)));
    check(cudaMemcpyToSymbol(LID_SHAPED, &s.c.lid_shaped, sizeof(int)));
}

// ---------------------------------------------------------------- pressure solver set-up
// Eigenvalues/vectors of a symmetric matrix A (n x n, row-major) by Jacobi rotations: repeatedly rotate two
// rows/columns so that one off-diagonal entry becomes 0, until A is diagonal. On return A's diagonal holds
// the eigenvalues and the columns of V the eigenvectors.
static void jacobi_eigen(int n, vector<double>& A, vector<double>& V) {
    auto a = [&](int i, int j) -> double& { return A[(size_t)i*n + j]; };
    auto v = [&](int i, int j) -> double& { return V[(size_t)i*n + j]; };
    V.assign((size_t)n*n, 0.0);
    for (int i = 0; i < n; i++) v(i, i) = 1.0;
    double total = 0;
    for (double q : A) total += q*q;
    for (int sweep = 0; sweep < 100; sweep++) {
        double off = 0;
        for (int i = 0; i < n; i++) for (int j = 0; j < n; j++) if (i != j) off += a(i, j)*a(i, j);
        if (off <= 1e-28*total) return;
        for (int p = 0; p < n; p++)
            for (int q = p+1; q < n; q++) {
                if (a(p, q) == 0.0) continue;
                double theta = (a(q, q) - a(p, p))/(2*a(p, q));
                double t = (theta >= 0 ? 1.0 : -1.0)/(std::fabs(theta) + std::sqrt(theta*theta + 1));
                double cs = 1/std::sqrt(t*t + 1), sn = t*cs;
                for (int k = 0; k < n; k++) { double x = a(k, p), y = a(k, q); a(k, p) = cs*x - sn*y; a(k, q) = sn*x + cs*y; }
                for (int k = 0; k < n; k++) { double x = a(p, k), y = a(q, k); a(p, k) = cs*x - sn*y; a(q, k) = sn*x + cs*y; }
                for (int k = 0; k < n; k++) { double x = v(k, p), y = v(k, q); v(k, p) = cs*x - sn*y; v(k, q) = sn*x + cs*y; }
            }
    }
    fail("pressure solver set-up: eigenvalues did not converge");
}
// Fast diagonalisation (same as FDMPoisson): per axis, the 1-D pressure operator K and the cell widths W give
// eigenvectors S with K S = W S diag(lambda). The 3-D operator is then diagonal in the S basis: transform,
// divide by lambda_x + lambda_y + lambda_z, transform back.
static void setup_pressure_solver(Solver& s) {
    vector<double> lambda[3];
    int zero_mode[3];
    for (int a = 0; a < 3; a++) {
        const Axis& g = s.ax[a];
        int n = g.n;
        vector<double> A((size_t)n*n, 0.0), V;
        for (int i = 0; i < n-1; i++) {                  // face i+1 connects cells i and i+1
            double k = 1.0/g.spacing[i+1];
            A[i*n+i] -= k; A[(i+1)*n+i+1] -= k; A[i*n+i+1] += k; A[(i+1)*n+i] += k;
        }
        for (int i = 0; i < n; i++)                      // symmetric form W^-1/2 K W^-1/2
            for (int j = 0; j < n; j++) A[i*n+j] /= std::sqrt(g.width[i]*g.width[j]);
        jacobi_eigen(n, A, V);
        lambda[a].resize(n);
        for (int i = 0; i < n; i++) lambda[a][i] = A[i*n+i];
        zero_mode[a] = (int)(std::max_element(lambda[a].begin(), lambda[a].end()) - lambda[a].begin());
        lambda[a][zero_mode[a]] = 0.0;                   // the constant mode (phi + constant is the same solution)
        vector<float> S((size_t)n*n), ST((size_t)n*n);
        for (int i = 0; i < n; i++)
            for (int j = 0; j < n; j++) {
                float e = (float)(V[i*n+j]/std::sqrt(g.width[i]));   // back from the symmetric form
                S[i*n+j] = e;
                ST[j*n+i] = e;
            }
        s.S[a] = gpu_array<float>((long long)n*n);  to_gpu(s.S[a], S.data(), (long long)n*n);
        s.ST[a] = gpu_array<float>((long long)n*n); to_gpu(s.ST[a], ST.data(), (long long)n*n);
    }
    vector<float> inv(s.n_cell);
    for (int z = 0; z < s.nz; z++)
        for (int y = 0; y < s.ny; y++)
            for (int x = 0; x < s.nx; x++) {
                bool zero = (x == zero_mode[X] && y == zero_mode[Y] && z == zero_mode[Z]);
                inv[x + s.nx*(y + s.ny*z)] = zero ? 0.0f : (float)(1.0/(lambda[Z][z] + lambda[Y][y] + lambda[X][x]));
            }
    s.inv_lambda = gpu_array<float>(s.n_cell);
    to_gpu(s.inv_lambda, inv.data(), s.n_cell);
}

// ---------------------------------------------------------------- pressure solve
// C = A @ B for row-major matrices (A: rows x k, B: k x cols), `batch` times with the given element strides.
// (cuBLAS works column-major, so it is asked for C^T = B^T A^T, which is the same memory.)
static void matmul(Solver& s, const float* A, const float* B, float* C, int rows, int cols, int k,
                   long long strideA = 0, long long strideB = 0, long long strideC = 0, int batch = 1) {
    const float one = 1.0f, zero = 0.0f;
    if (cublasSgemmStridedBatched(s.blas, CUBLAS_OP_N, CUBLAS_OP_N, cols, rows, k, &one, B, cols, strideB,
                                  A, k, strideA, &zero, C, cols, strideC, batch) != CUBLAS_STATUS_SUCCESS)
        fail("cuBLAS matrix multiply failed");
}
// phi += (approximate) solution of L phi = r: transform r to the eigenvector basis, divide, transform back.
// The transforms run in float, so r is scaled to rms 1 first; the outer loop (solve_pressure) corrects the error.
static void pressure_pass(Solver& s, const double* r, double* phi) {
    int n = s.n_cell, nx = s.nx, ny = s.ny, nz = s.nz, plane = nx*ny;
    double rms = std::sqrt(reduce_gpu(s, r, n, 3)/n);
    if (!(rms > 0)) return;
    RUN(to_float_scaled, n, r, 1.0/rms, s.t1, n);
    matmul(s, s.t1, s.S[X], s.t2, nz*ny, nx, nx);                            // along x
    matmul(s, s.ST[Y], s.t2, s.t1, ny, nx, ny, 0, plane, plane, nz);         // along y (each z layer)
    matmul(s, s.ST[Z], s.t1, s.t2, nz, plane, nz);                           // along z
    RUN(multiply, n, s.t2, s.inv_lambda, n);                                 // divide by the eigenvalues
    matmul(s, s.S[Z], s.t2, s.t1, nz, plane, nz);                            // back along z
    matmul(s, s.S[Y], s.t1, s.t2, ny, nx, ny, 0, plane, plane, nz);          // back along y
    matmul(s, s.t2, s.ST[X], s.t1, nz*ny, nx, nx);                           // back along x
    RUN(add_float_scaled, n, phi, s.t1, rms, n);
}
// solve L phi = rhs until |rhs - L phi| <= FDM_RTOL |rhs|, starting from guess (or 0); phi gets zero mean
static void solve_pressure(Solver& s, const double* rhs, const double* guess, double* phi) {
    int n = s.n_cell;
    double rhs_norm = std::sqrt(reduce_gpu(s, rhs, n, 3));
    s.phi_passes = 0;
    if (!std::isfinite(rhs_norm)) fail("pressure solve: right-hand side is NaN/inf (the run has blown up)");
    if (rhs_norm == 0.0) { check(cudaMemset(phi, 0, sizeof(double)*n)); s.phi_rel = 0; return; }
    const double* r = rhs;                                   // current residual
    double rel = 1.0;
    check(cudaMemset(phi, 0, sizeof(double)*n));
    if (guess) {
        gpu_copy(phi, guess, n);
        RUN(pressure_residual, n, phi, rhs, s.res);
        r = s.res;
        rel = std::sqrt(reduce_gpu(s, r, n, 3))/rhs_norm;
        if (rel >= 1.0) { check(cudaMemset(phi, 0, sizeof(double)*n)); r = rhs; rel = 1.0; }   // guess was useless
    }
    while (rel > s.c.fdm_rtol && s.phi_passes < s.c.fdm_max_passes) {
        pressure_pass(s, r, phi);
        s.phi_passes++;
        RUN(pressure_residual, n, phi, rhs, s.res);
        r = s.res;
        rel = std::sqrt(reduce_gpu(s, r, n, 3))/rhs_norm;
        if (!std::isfinite(rel)) fail("pressure solve: residual is NaN/inf");
    }
    if (rel > s.c.fdm_rtol) {
        char msg[200];
        std::snprintf(msg, sizeof msg, "pressure solve: residual %.2e > FDM_RTOL after %d passes", rel, s.phi_passes);
        fail(msg);
    }
    s.phi_rel = rel;
    RUN(subtract_constant, n, phi, reduce_gpu(s, phi, n, 1)/n, n);
}
// make m divergence-free:  out = m - grad(phi)  with  div(grad(phi)) = div(m).  Writes phi.
static void project(Solver& s, float* const m[3], const double* guess, double* phi, float* const out[3]) {
    for (int k = 0; k < 3; k++) RUN(zero_walls, s.size[k], k, m[k]);
    RUN(divergence, s.n_cell, m[0], m[1], m[2], s.rhs);
    solve_pressure(s, s.rhs, guess, phi);
    for (int k = 0; k < 3; k++) RUN(subtract_gradient, s.size[k], k, m[k], phi, out[k]);
}
// warm start for the next projection: phi changes smoothly in time, so extrapolate the last changes
// (dphi = b*dt*dphi/dt of earlier stages) to time tau and rescale to this stage. False if no history.
static bool guess_phi(Solver& s, double tau, double scale, double* out) {
    if (!s.c.phi_extrapolate || s.hist.empty()) return false;
    vector<int> order(s.hist.size());
    for (size_t k = 0; k < order.size(); k++) order[k] = (int)k;
    std::stable_sort(order.begin(), order.end(), [&](int a, int b) {     // nearest in time first, later first on ties
        double da = std::fabs(s.hist[a].t - tau), db = std::fabs(s.hist[b].t - tau);
        return da != db ? da < db : a > b;
    });
    const History& h1 = s.hist[order[0]];
    const History* h2 = nullptr;
    for (size_t k = 1; k < order.size() && !h2; k++)
        if (std::fabs(s.hist[order[k]].t - h1.t) > 1e-12*std::max(1.0, std::fabs(h1.t))) h2 = &s.hist[order[k]];
    double w2 = h2 ? (tau - h1.t)/(h2->t - h1.t) : 0.0, w1 = 1.0 - w2;
    if (h2) RUN(lincomb, s.n_cell, scale*w1/h1.scale, h1.dphi, scale*w2/h2->scale, h2->dphi, out, s.n_cell);
    else    RUN(scaled, s.n_cell, scale*w1/h1.scale, h1.dphi, out, s.n_cell);
    return true;
}

// ---------------------------------------------------------------- the time step
// One SSP-RK3 step of the Eulerian impulse scheme (same as _trial_step_eul_impulse in the Python version).
static void step(Solver& s) {
    int n = s.n_cell;
    // 1. gauge reset: m0 = r m + (1-r) u,  phi0 = r phi
    for (int k = 0; k < 3; k++)
        RUN(blend, s.size[k], s.m[k], s.u[k], (float)s.c.reinit_r, (float)(1.0 - s.c.reinit_r), s.m0[k], s.size[k]);
    RUN(scaled, n, s.c.reinit_r, s.phi, s.phi0, n);

    // 2. time step from the CFL rate (largest |u|/dx summed over x, y, z) and the viscous rate
    RUN(cfl_rate, n, s.u[0], s.u[1], s.u[2], s.cell_f);
    double rates[2] = {(double)(float)reduce_gpu(s, s.cell_f, n, 0), (double)s.viscous_rate};
    double use[2] = {rates[0], rates[1]};
    if (s.c.dt_lag && s.have_prev_rates) { use[0] = s.prev_rates[0]; use[1] = s.prev_rates[1]; }   // previous step's
    s.prev_rates[0] = rates[0]; s.prev_rates[1] = rates[1]; s.have_prev_rates = true;
    if (!std::isfinite(use[0])) fail("velocity is NaN/inf: the solution has blown up");
    double dt;
    if (!s.c.adaptive_dt) { dt = s.c.dt_fixed; s.dt_reason = 0; }
    else {
        double caps[3] = {s.c.dt_max, s.c.cfl_target/std::max(use[0], 1e-300), s.c.diffusion_limit/std::max(use[1], 1e-300)};
        int best = 0;
        for (int k = 1; k < 3; k++) if (caps[k] < caps[best]) best = k;
        dt = caps[best]; s.dt_reason = best + 1;
    }
    if (s.c.dt_lag && s.last_dt > 0) dt = std::min(dt, 1.1*s.last_dt);

    // 3. three stages.  Stage k: m_new = [a0 m0 +] b (m + dt R(u, m)),  then u_new = m_new - grad(phi)
    const double a0s[3] = {0.0, 0.75, 1.0/3.0}, bs[3] = {1.0, 0.25, 2.0/3.0};
    const double times[3] = {s.t, s.t + dt, s.t + 0.5*dt};
    float* cur_u[3] = {s.u[0], s.u[1], s.u[2]};
    float* cur_m[3] = {s.m0[0], s.m0[1], s.m0[2]};
    double* cur_phi = s.phi0;
    for (int stage = 0; stage < 3; stage++) {
        double a0 = a0s[stage], b = bs[stage], ts = times[stage];
        float** nm = s.new_m[stage % 2];
        float** nu = s.new_u[stage % 2];
        int o4 = s.c.impulse_order == 4;
        float K = (float)std::max(s.c.impulse_order4_ramp, 1.0);
        // inviscid term
        RUN(inviscid_x, s.n_u, cur_u[0], cur_u[1], cur_u[2], cur_m[0], cur_m[1], cur_m[2], o4, K, s.I[0]);
        RUN(inviscid_y, s.n_v, cur_u[0], cur_u[1], cur_u[2], cur_m[0], cur_m[1], cur_m[2], o4, K, s.I[1]);
        RUN(inviscid_z, s.n_w, cur_u[0], cur_u[1], cur_u[2], cur_m[0], cur_m[1], cur_m[2], o4, K, s.I[2]);
        // viscosity + RK3 combination
        for (int k = 0; k < 3; k++)
            RUN(rk3_stage, s.size[k], k, cur_m[k], s.m0[k], cur_u[k], s.I[k], (float)s.c.nu, (float)s.lid(ts),
                (float)dt, (float)a0, (float)b, stage > 0 ? 1 : 0, nm[k]);
        // the gauge potential carried by the RK combination, and a warm start for the projection
        if (a0 != 0.0) RUN(lincomb, n, a0, s.phi0, b, cur_phi, s.base, n);
        else           RUN(scaled, n, b, cur_phi, s.base, n);
        if (guess_phi(s, ts, b*dt, s.guess)) RUN(add_to, n, s.guess, s.base, n);
        else gpu_copy(s.guess, s.base, n);
        double* new_phi = s.phi_stage[stage % 2];
        project(s, nm, s.guess, new_phi, nu);
        // remember dphi = phi - base for the next warm starts (last three)
        if (s.c.phi_extrapolate) {
            double* buf = nullptr;
            for (int k = 0; k < 4 && !buf; k++) {
                bool used = false;
                for (const History& h : s.hist) used |= (h.dphi == s.hist_buf[k]);
                if (!used) buf = s.hist_buf[k];
            }
            RUN(lincomb, n, 1.0, new_phi, -1.0, s.base, buf, n);
            s.hist.push_back({ts, b*dt, buf});
            if (s.hist.size() > 3) s.hist.erase(s.hist.begin());
        }
        for (int k = 0; k < 3; k++) { cur_u[k] = nu[k]; cur_m[k] = nm[k]; }
        cur_phi = new_phi;
    }

    // 4. the result becomes the new state (swap pointers: the old state arrays become work arrays)
    for (int k = 0; k < 3; k++) {
        std::swap(s.u[k], s.new_u[0][k]);                   // stage 3 wrote set 0
        std::swap(s.m[k], s.new_m[0][k]);
    }
    std::swap(s.phi, s.phi_stage[0]);
    s.t += dt;
    s.steps++;
    s.last_dt = dt;
    s.last_cfl = dt*use[0];
    check(cudaGetLastError());
}

// smooth random kick (rms amp*U), made divergence-free; the impulse gets the same change, so the gauge is kept
static void kick(Solver& s, const double* const noise[3], double amp, int passes) {
    for (int k = 0; k < 3; k++) {
        int n = s.size[k], sx = s.nx + (k == 0), sy = s.ny + (k == 1), sz = s.nz + (k == 2);
        to_gpu(s.noise[0], noise[k], n);
        double* a = s.noise[0];
        double* b = s.noise[1];
        for (int p = 0; p < std::max(1, passes); p++)
            for (int axis = 2; axis >= 0; axis--) {         // z, y, x: the axis order of the Python version
                RUN(smooth, n, a, b, axis, sx, sy, sz);
                std::swap(a, b);
            }
        double rms = std::sqrt(reduce_gpu(s, a, n, 3)/n);
        RUN(add_noise, n, s.u[k], a, amp*s.c.u_lid/std::max(rms, 1e-30), s.tmp_f[k], n);
    }
    project(s, s.tmp_f, nullptr, s.phi_stage[0], s.new_u[0]);    // phi itself is left unchanged
    for (int k = 0; k < 3; k++) {
        RUN(add_difference, s.size[k], s.m[k], s.new_u[0][k], s.u[k], s.size[k]);
        std::swap(s.u[k], s.new_u[0][k]);
    }
    check(cudaGetLastError());
}

// =====================================================================================================
//  PART 4.  Functions called from Python (python_api.py)
// =====================================================================================================
#ifdef _WIN32
#define EXPORT extern "C" __declspec(dllexport)
#else
#define EXPORT extern "C"
#endif
// run f; on an error remember the message (solver_last_error) and return 1
template <class F> static int safely(F f) {
    try { f(); return 0; }
    catch (const std::exception& e) { last_error = e.what(); return 1; }
}
static Solver* the_solver = nullptr;                       // one solver at a time
static Solver& S() { if (!the_solver) fail("no solver: call solver_create first"); return *the_solver; }
static void fill(Report* r) {
    Solver& s = S();
    r->steps = s.steps; r->time = s.t; r->dt = s.last_dt; r->cfl = s.last_cfl; r->phi_rel = s.phi_rel;
    r->dt_reason = s.dt_reason; r->phi_passes = s.phi_passes;
}
EXPORT int solver_destroy() {
    return safely([&] {
        if (the_solver) cublasDestroy(the_solver->blas);
        for (void* p : all_gpu_arrays) cudaFree(p);
        all_gpu_arrays.clear();
        delete the_solver;
        the_solver = nullptr;
    });
}
EXPORT int solver_create(const Config* config) {
    return safely([&] {
        solver_destroy();
        the_solver = new Solver();
        Solver& s = *the_solver;
        s.c = *config;
        s.nx = s.c.nx; s.ny = s.c.ny; s.nz = s.c.nz;
        if (s.nx != s.ny || s.nx % 2 || s.nx < 6) fail("nx = ny must be even and >= 6");
        if (s.nz < 5) fail("need nz >= 5");
        if (std::max(s.nx, s.nz) > GMAX) fail("at most 1024 cells per axis");
        if (s.c.impulse_order != 2 && s.c.impulse_order != 4) fail("IMPULSE_ORDER must be 2 or 4");
        if ((double)(s.nx+1)*s.ny*(s.nz+1) > 2e9) fail("grid too large");
        s.n_u = (s.nx+1)*s.ny*s.nz; s.n_v = s.nx*(s.ny+1)*s.nz; s.n_w = s.nx*s.ny*(s.nz+1); s.n_cell = s.nx*s.ny*s.nz;
        s.size[0] = s.n_u; s.size[1] = s.n_v; s.size[2] = s.n_w;
        // grid
        s.ax[X] = make_axis(s.c, s.nx, s.c.length);
        s.ax[Y] = make_axis(s.c, s.ny, s.c.length);
        s.ax[Z] = make_axis(s.c, s.nz, s.nz*(s.c.lz/s.nz));
        s.volume = s.ax[X].length*s.ax[Y].length*s.ax[Z].length;
        upload_grid(s);
        // viscous rate: largest over cells of nu * sum over axes of 4/(width * smaller spacing)
        float vmax = 0.0f;
        auto geo = [&](int a, int i) { return 4.0f/((float)s.ax[a].width[i]*std::min((float)s.ax[a].spacing[i], (float)s.ax[a].spacing[i+1])); };
        for (int z = 0; z < s.nz; z++) for (int y = 0; y < s.ny; y++) for (int x = 0; x < s.nx; x++)
            vmax = std::max(vmax, (float)s.c.nu*(geo(X, x) + geo(Y, y) + geo(Z, z)));
        s.viscous_rate = vmax;
        // arrays
        for (int k = 0; k < 3; k++) {
            s.u[k] = gpu_array<float>(s.size[k]);   s.m[k] = gpu_array<float>(s.size[k]);
            s.m0[k] = gpu_array<float>(s.size[k]);  s.I[k] = gpu_array<float>(s.size[k]);
            s.tmp_f[k] = gpu_array<float>(s.size[k]);
            for (int j = 0; j < 2; j++) { s.new_m[j][k] = gpu_array<float>(s.size[k]); s.new_u[j][k] = gpu_array<float>(s.size[k]); }
            s.cell_d[k] = gpu_array<double>(s.n_cell);
        }
        for (double** p : {&s.phi, &s.phi0, &s.phi_stage[0], &s.phi_stage[1], &s.base, &s.guess, &s.rhs, &s.res})
            *p = gpu_array<double>(s.n_cell);
        for (int k = 0; k < 4; k++) s.hist_buf[k] = gpu_array<double>(s.n_cell);
        int biggest = std::max(s.n_u, std::max(s.n_v, s.n_w));
        s.noise[0] = gpu_array<double>(biggest);
        s.noise[1] = gpu_array<double>(biggest);
        s.cell_f = gpu_array<float>(s.n_cell);
        s.partial = gpu_array<double>(65536 + 256);
        s.stats = gpu_array<double>(6LL*s.nx*s.ny);
        s.plane = gpu_array<float>(4LL*s.nx*s.ny);
        s.first = gpu_array<int>(1);
        s.t1 = gpu_array<float>(s.n_cell); s.t2 = gpu_array<float>(s.n_cell);
        if (cublasCreate(&s.blas) != CUBLAS_STATUS_SUCCESS) fail("could not start cuBLAS");
        setup_pressure_solver(s);
        // probes: the cell containing (x, y) on the mid-span layer
        vector<int> cells;
        for (int p = 0; p < s.c.n_probes; p++) {
            auto cell_of = [](const Axis& g, double frac) {
                int k = (int)(std::upper_bound(g.face.begin(), g.face.end(), frac*g.length) - g.face.begin()) - 1;
                return std::min(g.n - 1, std::max(0, k));
            };
            cells.push_back(cell_of(s.ax[X], s.c.probe_x[p]));
            cells.push_back(cell_of(s.ax[Y], s.c.probe_y[p]));
            cells.push_back(s.nz/2);
        }
        s.probe_cells = gpu_array<int>(std::max(1, (int)cells.size()));
        if (!cells.empty()) to_gpu(s.probe_cells, cells.data(), (long long)cells.size());
        s.probe_out = gpu_array<float>(16);
        check(cudaDeviceSynchronize());
    });
}
// initial state: v, w = given noise (NULL = none), m = that field, u = its divergence-free part
EXPORT int solver_initialize(const double* v_noise, const double* w_noise) {
    return safely([&] {
        Solver& s = S();
        const double* src[3] = {nullptr, v_noise, w_noise};
        for (int k = 0; k < 3; k++) {
            vector<float> f(s.size[k], 0.0f);
            if (src[k]) for (int i = 0; i < s.size[k]; i++) f[i] = (float)src[k][i];
            to_gpu(s.m[k], f.data(), s.size[k]);
        }
        project(s, s.m, nullptr, s.phi, s.u);
        s.t = 0; s.steps = 0; s.last_dt = 0; s.hist.clear(); s.have_prev_rates = false;
        check(cudaDeviceSynchronize());
    });
}
EXPORT int solver_step(Report* r) {
    return safely([&] { step(S()); fill(r); });
}
EXPORT int solver_kick(const double* nu, const double* nv, const double* nw, double amp, int passes) {
    return safely([&] { const double* n[3] = {nu, nv, nw}; kick(S(), n, amp, passes); });
}
EXPORT int solver_report(Report* r) { return safely([&] { fill(r); }); }
EXPORT int solver_diagnostics(Diagnostics* d) {
    return safely([&] {
        Solver& s = S();
        int n = s.n_cell;
        float lid = (float)s.lid(s.t);
        RUN(cell_diagnostics, n, s.u[0], s.u[1], s.u[2], lid, s.cell_d[0], s.cell_d[1], s.cell_d[2]);
        d->kinetic_energy = reduce_gpu(s, s.cell_d[0], n, 1)/s.volume;
        d->enstrophy = 0.5*reduce_gpu(s, s.cell_d[1], n, 1)/s.volume;
        d->div_rms = std::sqrt(reduce_gpu(s, s.cell_d[2], n, 1)/s.volume);
        double umax = 0;
        for (int k = 0; k < 3; k++) umax = std::max(umax, reduce_gpu(s, s.u[k], s.size[k], 2));
        d->umax = umax;
        RUN(cfl_rate, n, s.u[0], s.u[1], s.u[2], s.cell_f);
        d->cfl_rate = reduce_gpu(s, s.cell_f, n, 0);
        // gauge: largest |m - u| and where it is
        double best = -1; int best_comp = 0, best_i = 0;
        for (int k = 0; k < 3; k++) {
            RUN(abs_difference, s.size[k], s.m[k], s.u[k], s.tmp_f[k], s.size[k]);
            double val = reduce_gpu(s, s.tmp_f[k], s.size[k], 0);
            if (val > best) {
                int big = 2147483647;
                to_gpu(s.first, &big, 1);
                RUN(first_index_of, s.size[k], s.tmp_f[k], (float)val, s.size[k], s.first);
                to_cpu(&best_i, s.first, 1);
                best = val; best_comp = k;
            }
        }
        int sx = s.nx + (best_comp == 0), sy = s.ny + (best_comp == 1);
        int ix = best_i % sx, iy = (best_i / sx) % sy;
        d->gauge_ratio = best/std::max(umax, 1e-30);
        d->gauge_x = (best_comp == 0 ? s.ax[X].face[ix] : s.ax[X].centre[std::min(ix, s.nx-1)])/s.c.length;
        d->gauge_y = (best_comp == 1 ? s.ax[Y].face[iy] : s.ax[Y].centre[std::min(iy, s.ny-1)])/s.c.length;
    });
}
// grid of one axis (0 x, 1 y, 2 z): faces (n+1) and centres (n)
EXPORT int solver_grid(int axis, double* faces, double* centres) {
    return safely([&] {
        const Axis& g = S().ax[axis];
        std::copy(g.face.begin(), g.face.end(), faces);
        std::copy(g.centre.begin(), g.centre.end(), centres);
    });
}
// fields: 0 u, 1 v, 2 w, 3 mx, 4 my, 5 mz (float), 6 phi (double)
EXPORT int solver_get_field(int field, void* out) {
    return safely([&] {
        Solver& s = S();
        if (field < 3) to_cpu((float*)out, s.u[field], s.size[field]);
        else if (field < 6) to_cpu((float*)out, s.m[field-3], s.size[field-3]);
        else to_cpu((double*)out, s.phi, s.n_cell);
    });
}
EXPORT int solver_set_field(int field, const void* in) {
    return safely([&] {
        Solver& s = S();
        if (field < 3) to_gpu(s.u[field], (const float*)in, s.size[field]);
        else if (field < 6) to_gpu(s.m[field-3], (const float*)in, s.size[field-3]);
        else to_gpu(s.phi, (const double*)in, s.n_cell);
        s.hist.clear(); s.have_prev_rates = false;
    });
}
EXPORT int solver_set_clock(double t, long long steps, double last_dt) {
    return safely([&] { Solver& s = S(); s.t = t; s.steps = steps; s.last_dt = last_dt; });
}
EXPORT int solver_probes(float* out) {                     // u, v per probe, now
    return safely([&] {
        Solver& s = S();
        if (s.c.n_probes == 0) return;
        RUN(probe_values, s.c.n_probes, s.u[0], s.u[1], s.probe_cells, s.c.n_probes, s.probe_out);
        to_cpu(out, s.probe_out, 2LL*s.c.n_probes);
    });
}
// add weight * (mid-span u, v, uu, vv, ww, uv) to the statistics and return the centreline sample (5 x nx)
EXPORT int solver_add_statistics(double weight, double* centreline) {
    return safely([&] {
        Solver& s = S();
        RUN(add_statistics, s.nx*s.ny, s.u[0], s.u[1], s.u[2], weight, s.stats);
        RUN(centreline_sample, s.nx, s.u[0], s.u[1], s.u[2], s.cell_d[0]);
        to_cpu(centreline, s.cell_d[0], 5LL*s.nx);
    });
}
EXPORT int solver_get_statistics(double* sums) { return safely([&] { to_cpu(sums, S().stats, 6LL*S().nx*S().ny); }); }
EXPORT int solver_set_statistics(const double* sums) { return safely([&] { to_gpu(S().stats, sums, 6LL*S().nx*S().ny); }); }
EXPORT int solver_midplane(float* out) {                   // [u | v | w | |omega|] on the mid-span plane
    return safely([&] {
        Solver& s = S();
        RUN(midplane, s.nx*s.ny, s.u[0], s.u[1], s.u[2], (float)s.lid(s.t), s.plane);
        to_cpu(out, s.plane, 4LL*s.nx*s.ny);
    });
}
EXPORT int solver_q_volume(int colour_w, float* q, float* colour) {
    return safely([&] {
        Solver& s = S();
        double L_over_U = s.c.length/s.c.u_lid;
        RUN(q_criterion, s.n_cell, s.u[0], s.u[1], s.u[2], (float)s.lid(s.t), (float)(L_over_U*L_over_U),
            (float)(1.0/s.c.u_lid), colour_w, s.cell_f, s.tmp_f[0]);
        to_cpu(q, s.cell_f, s.n_cell);
        to_cpu(colour, s.tmp_f[0], s.n_cell);
    });
}
EXPORT const char* solver_last_error() { return last_error.c_str(); }
