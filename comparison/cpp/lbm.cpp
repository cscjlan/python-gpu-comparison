#include "lbm.hpp"

// This is the python API
// These redirect to the actual type generic implementation
extern "C" {
void LBM_compute_edf_f32(dim3 *blocks, dim3 *threads, int nx, int ny, float *f,
                         float *rho, float *u, int *nodetype, float *ex,
                         float *ey, float *w, float es) {
    lbm::compute_edf(blocks, threads, nx, ny, f, rho, u, nodetype, ex, ey, w,
                     es);
}
void LBM_compute_edf_f64(dim3 *blocks, dim3 *threads, int nx, int ny, double *f,
                         double *rho, double *u, int *nodetype, double *ex,
                         double *ey, double *w, double es) {
    lbm::compute_edf(blocks, threads, nx, ny, f, rho, u, nodetype, ex, ey, w,
                     es);
}
void LBM_collide_f32(dim3 *blocks, dim3 *threads, int nx, int ny, float *f,
                     float *rho, float *u, int *nodetype, float *tau,
                     float *Fg) {
    lbm::collide(blocks, threads, nx, ny, f, rho, u, nodetype, tau, Fg);
}
void LBM_collide_f64(dim3 *blocks, dim3 *threads, int nx, int ny, double *f,
                     double *rho, double *u, int *nodetype, double *tau,
                     double *Fg) {
    lbm::collide(blocks, threads, nx, ny, f, rho, u, nodetype, tau, Fg);
}
void LBM_stream_and_bounce_f32(dim3 *blocks, dim3 *threads, int nx, int ny,
                               float *f, int *nodetype, float *ex, float *ey) {
    lbm::stream_and_bounce(blocks, threads, nx, ny, nx, ny, f, nodetype, ex,
                           ey);
}
void LBM_stream_and_bounce_f64(dim3 *blocks, dim3 *threads, int nx, int ny,
                               double *f, int *nodetype, double *ex,
                               double *ey) {
    lbm::stream_and_bounce(blocks, threads, nx, ny, nx, ny, f, nodetype, ex,
                           ey);
}
void LBM_compute_macro_vars_f32(dim3 *blocks, dim3 *threads, int nx, int ny,
                                float *f, float *rho, float *u, int *nodetype,
                                float *ex, float *ey) {
    lbm::compute_macro_vars(blocks, threads, nx, ny, f, rho, u, nodetype, ex,
                            ey);
}
void LBM_compute_macro_vars_f64(dim3 *blocks, dim3 *threads, int nx, int ny,
                                double *f, double *rho, double *u,
                                int *nodetype, double *ex, double *ey) {
    lbm::compute_macro_vars(blocks, threads, nx, ny, f, rho, u, nodetype, ex,
                            ey);
}

void *LBM_malloc(size_t bytes) { return gpu::allocate(bytes); }
void LBM_memcpy(void *dst, void *src, size_t bytes) {
    return gpu::memcpy(dst, src, bytes);
}
void LBM_free(void *ptr) { return gpu::free(ptr); }
void LBM_synchronize() { return gpu::synchronize(); }
}
