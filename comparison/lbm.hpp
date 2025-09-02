#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <hip/hip_runtime.h>
#include <type_traits>

#define LAUNCH_KERNEL(kernel, ...)                                             \
    gpu::launch_kernel(#kernel, __FILE__, __LINE__, kernel, __VA_ARGS__)
#define HIP_ERRCHK(result) gpu::hip_errchk(result, __FILE__, __LINE__)

// Here we have generic GPU related boilerplate
namespace gpu {
inline void hip_errchk(hipError_t result, const char *file, int32_t line) {
    if (result != hipSuccess) {
        std::printf("\n\n%s in %s at line %d\n", hipGetErrorString(result),
                    file, line);
        exit(EXIT_FAILURE);
    }
}

inline void *allocate(size_t num_bytes) {
    void *ptr = nullptr;
    HIP_ERRCHK(hipMalloc(&ptr, num_bytes));
    if (ptr == nullptr) {
        std::fprintf(stderr, "GPU malloc allocated a nullptr\n");
        std::abort();
    }

    return ptr;
}

inline void free(void *ptr) { HIP_ERRCHK(hipFree(ptr)); }

inline void memcpy(void *dst, const void *src, size_t num_bytes) {
    HIP_ERRCHK(hipMemcpy(dst, src, num_bytes, hipMemcpyDefault));
}

inline void synchronize() { HIP_ERRCHK(hipDeviceSynchronize()); }

template <typename F, typename... Args>
__global__ void loop_kernel(F f, int nx, int ny, Args... args) {
    const auto tidx = threadIdx.x + blockIdx.x * blockDim.x;
    const auto tidy = threadIdx.y + blockIdx.y * blockDim.y;
    const auto stridex = blockDim.x * gridDim.x;
    const auto stridey = blockDim.y * gridDim.y;

    const auto num_values = nx * ny;
    for (auto y = tidy; y < ny; y += stridey) {
        const auto offset = y * nx;
        for (auto x = tidx; x < nx; x += stridex) {
            f(offset + x, num_values, args...);
        }
    }
}

void hip_errchk(hipError_t result, const char *file, int32_t line);

template <typename... Args>
void launch_kernel(const char *kernel_name, const char *file, int32_t line,
                   void (*kernel)(Args...), dim3 blocks, dim3 threads,
                   size_t num_bytes_shared_mem, hipStream_t stream,
                   const char *function_name, Args... args) {
#if !NDEBUG
    int32_t device = 0;
    HIP_ERRCHK(hipGetDevice(&device));

    // Helper lambda for querying device attributes
    auto get_device_attribute = [&device](hipDeviceAttribute_t attribute) {
        int32_t value = 0;
        HIP_ERRCHK(hipDeviceGetAttribute(&value, attribute, device));
        return value;
    };

    // Get maximum allowed size of block for each dimension
    const dim3 max_threads(
        get_device_attribute(
            hipDeviceAttribute_t::hipDeviceAttributeMaxBlockDimX),
        get_device_attribute(
            hipDeviceAttribute_t::hipDeviceAttributeMaxBlockDimY),
        get_device_attribute(
            hipDeviceAttribute_t::hipDeviceAttributeMaxBlockDimZ));

    // Get maximum allowed size of grid for each dimension
    const dim3 max_blocks(
        get_device_attribute(
            hipDeviceAttribute_t::hipDeviceAttributeMaxGridDimX),
        get_device_attribute(
            hipDeviceAttribute_t::hipDeviceAttributeMaxGridDimY),
        get_device_attribute(
            hipDeviceAttribute_t::hipDeviceAttributeMaxGridDimZ));

    // Maximum threads per block in total (i.e. x * y * z)
    const int32_t max_threads_per_block = get_device_attribute(
        hipDeviceAttribute_t::hipDeviceAttributeMaxThreadsPerBlock);

    // Maximum number of bytes of shared memory per block
    const int32_t max_shared_memory_per_block = get_device_attribute(
        hipDeviceAttribute_t::hipDeviceAttributeMaxSharedMemoryPerBlock);

    auto error_print_prelude = [&kernel_name, &function_name]() {
        std::fprintf(
            stderr,
            "Bad launch parameters for kernel \"%s\" with lambda \"%s\"\n",
            kernel_name, function_name);
    };
    // Helper lambda for asserting dim3 launch variable is within allowed limits
    auto assert_within_limits =
        [&error_print_prelude](const char *name, int32_t value, int32_t min,
                               int32_t max) {
            if (not(min <= value && value <= max)) {
                error_print_prelude();
                std::fprintf(stderr, "%s (%d) not within limits [%d, %d]\n",
                             name, value, min, max);
                exit(EXIT_FAILURE);
            }
        };

    assert_within_limits("threads.x", threads.x, 1, max_threads.x);
    assert_within_limits("threads.y", threads.y, 1, max_threads.y);
    assert_within_limits("threads.z", threads.z, 1, max_threads.z);
    assert_within_limits("blocks.x", blocks.x, 1, max_blocks.x);
    assert_within_limits("blocks.y", blocks.y, 1, max_blocks.y);
    assert_within_limits("blocks.z", blocks.z, 1, max_blocks.z);
    assert_within_limits("block size", threads.x * threads.y * threads.z, 1,
                         max_threads_per_block);

    // Requested amount of shared memory must be below the limit queried above
    if (num_bytes_shared_mem > max_shared_memory_per_block) {
        error_print_prelude();
        std::fprintf(stderr, "Shared memory request too large: %ld > %d\n",
                     num_bytes_shared_mem, max_shared_memory_per_block);
        exit(EXIT_FAILURE);
    }

    // Reset the error variable to success
    [[maybe_unused]] auto result = hipGetLastError();
#endif

    kernel<<<blocks, threads, num_bytes_shared_mem, stream>>>(args...);

#if !NDEBUG
    // Quoting from HIP documentation
    // (https://rocm.docs.amd.com/projects/HIP/en/latest/how-to/hip_runtime_api/error_handling.html)
    //
    //  > hipGetLastError() returns the returned error code of the last HIP
    //    runtime API call even if it’s hipSuccess, while cudaGetLastError
    //    returns the error returned by any of the preceding CUDA APIs in the
    //    same host thread. hipGetLastError() behavior will be matched with
    //    cudaGetLastError in ROCm release 7.0.
    //
    // Because of this, using the Cuda recommended pattern of cathcing kernel
    // errors by first synchronizing with the device, then calling
    // hipGetLastError doesn't work. Until ROCm 7.0, HIP will overwrite the
    // error code returned by the kernel with success from hipDeviceSynchronize.
    // This means hipGetLastError can only be used to catch launch parameter
    // errors, i.e. errors that happen during the kernel launch, like too many
    // threads per block. Any errors that happen during the asynchronous kernel
    // execution are missed. To be able to catch even the kernel launch errors,
    // one must not synchronize first, if using ROCm < 7.0, or the errors will
    // be overwritten.

#if defined(__NVCC__) || (defined(__clang__) && defined(__CUDA__))
    result = hipDeviceSynchronize();
#endif
    result = hipGetLastError();
    if (result != hipSuccess) {
        printf("Error with kernel \"%s\" executing lambda \"%s\" in %s at line "
               "%d\n%s: %s\n",
               kernel_name, function_name, file, line, hipGetErrorName(result),
               hipGetErrorString(result));
        exit(EXIT_FAILURE);
    }
#endif
}

void *allocate(size_t num_bytes);
void free(void *ptr);
void memcpy(void *dst, const void *src, size_t num_bytes);
void synchronize();
} // namespace gpu

// Here we have the device functions that are executed on the GPU
namespace lbm {
template <typename... Args>
void compute_edf(dim3 *blocks, dim3 *threads, Args... args) {
    LAUNCH_KERNEL(
        gpu::loop_kernel, *blocks, *threads, 0, 0, "compute_edf",
        [] __device__(int index, int num_values, auto *f, auto *rho, auto *u,
                      int *nodetype, auto *ex, auto *ey, auto *w, auto es) {
            using ft =
                std::remove_cv_t<std::remove_reference_t<decltype(f[0])>>;

            const auto s = static_cast<ft>(nodetype[index] <= 0);

            static constexpr auto N = 9;
            for (auto i = 0; i < N; i++) {
                const auto u0 = u[index];
                const auto u1 = u[index + num_values];
                const auto exi = ex[i];
                const auto eyi = ey[i];

                const auto ux2 = u0 * u0;
                const auto uy2 = u1 * u1;
                const auto u2 = ux2 + uy2;
                const auto euxy = exi * eyi * u0 * u1;
                const auto euxx = exi * exi * ux2;
                const auto euyy = eyi * eyi * uy2;
                const auto eu2 = static_cast<ft>(2.0f) * euxy + euxx + euyy;

                const auto inv_es_sq = static_cast<ft>(1.0f) / (es * es);
                const auto term_order1 = inv_es_sq * (exi * u0 + eyi * u1);
                const auto term_order2 =
                    static_cast<ft>(0.5f) * inv_es_sq * (inv_es_sq * eu2 - u2);

                const auto f_old = f[index + i * num_values];
                const auto f_new =
                    w[i] * rho[index] *
                    (static_cast<ft>(1.0f) + term_order1 + term_order2);

                f[index + i * num_values] =
                    s * f_new + (static_cast<ft>(1.0f) - s) * f_old;
            }
        },
        args...);
}

template <typename... Args>
void collide(dim3 *blocks, dim3 *threads, Args... args) {
    LAUNCH_KERNEL(
        gpu::loop_kernel, *blocks, *threads, 0, 0, "collide",
        [] __device__(int index, int num_values, auto *f, auto *rho, auto *u,
                      int *nodetype, auto *tau, auto *Fg) {
            using ft =
                std::remove_cv_t<std::remove_reference_t<decltype(f[0])>>;

            const auto s = static_cast<ft>(nodetype[index] <= 0);

            const auto rho_i = rho[index];
            const auto tau_i = tau[index];
            const auto tau_per_rho =
                std::min(tau_i / rho_i, std::numeric_limits<ft>::max());

            const int i = index + 0 * num_values;
            const int j = index + 1 * num_values;

            const auto u0 = u[i] + s * Fg[i] * tau_per_rho;
            const auto u1 = u[j] + s * Fg[j] * tau_per_rho;
            u[i] = u0;
            u[j] = u1;

            const auto u0_sq = u0 * u0;
            const auto u1_sq = u1 * u1;
            const auto u0_sq_p_u1_sq = static_cast<ft>(0.5f) * (u0_sq + u1_sq);
            const auto u0_p_u1 = u0 + u1;
            const auto u0_m_u1 = u0 - u1;
            const auto u0_p_u1_sq = static_cast<ft>(1.5f) * u0_p_u1 * u0_p_u1;
            const auto u0_m_u1_sq = static_cast<ft>(1.5f) * u0_m_u1 * u0_m_u1;

            ft f_updated[9] = {
                -u0_sq - u1_sq + static_cast<ft>(0.33333333f),
                -static_cast<ft>(0.5f) * u1_sq + u0_sq + u0,
                -static_cast<ft>(0.5f) * u0_sq + u1_sq + u1,
                -u0_sq_p_u1_sq + u0_p_u1 + u0_p_u1_sq,
                -u0_sq_p_u1_sq + u0_m_u1 + u0_m_u1_sq,
                -static_cast<ft>(0.5f) * u1_sq + u0_sq - u0,
                -static_cast<ft>(0.5f) * u0_sq + u1_sq - u1,
                -u0_sq_p_u1_sq - u0_p_u1 + u0_p_u1_sq,
                -u0_sq_p_u1_sq - u0_m_u1 + u0_m_u1_sq,
            };

            const auto rho_per_three = static_cast<ft>(0.333333f) * rho_i;
            const auto inv_tau = std::min(static_cast<ft>(1.0f) / tau_i,
                                          std::numeric_limits<ft>::max());
            const auto tau_m_1 = tau_i - static_cast<ft>(1.0f);

            static constexpr size_t N = 9;
            // clang-format off
            static constexpr ft multipliers[N] = {
                static_cast<ft>(2.00f),
                static_cast<ft>(1.00f),
                static_cast<ft>(1.00f),
                static_cast<ft>(0.25f),
                static_cast<ft>(0.25f),
                static_cast<ft>(1.00f),
                static_cast<ft>(1.00f),
                static_cast<ft>(0.25f),
                static_cast<ft>(0.25f),
            };
            // clang-format on
            for (size_t i = 0; i < N; i++) {
                const auto f_eq =
                    multipliers[i] * rho_per_three *
                    (f_updated[i] + static_cast<ft>(0.333333333f));
                const auto f_old = f[index + i * num_values];
                const auto f_new = inv_tau * (tau_m_1 * f_old + f_eq);

                f_updated[i] = s * f_new + (static_cast<ft>(1.0f) - s) * f_old;
            }

            // clang-format off
            // Update 1-8, such that pairs are swapped:
            // 0 <--> 0
            // 1 <--> 5
            // 2 <--> 6
            // 3 <--> 7
            // 4 <--> 8
            f[index + 0 * num_values] = s * f_updated[0] + (static_cast<ft>(1.0f) - s) * f_updated[0];
            f[index + 1 * num_values] = s * f_updated[5] + (static_cast<ft>(1.0f) - s) * f_updated[1];
            f[index + 2 * num_values] = s * f_updated[6] + (static_cast<ft>(1.0f) - s) * f_updated[2];
            f[index + 3 * num_values] = s * f_updated[7] + (static_cast<ft>(1.0f) - s) * f_updated[3];
            f[index + 4 * num_values] = s * f_updated[8] + (static_cast<ft>(1.0f) - s) * f_updated[4];
            f[index + 5 * num_values] = s * f_updated[1] + (static_cast<ft>(1.0f) - s) * f_updated[5];
            f[index + 6 * num_values] = s * f_updated[2] + (static_cast<ft>(1.0f) - s) * f_updated[6];
            f[index + 7 * num_values] = s * f_updated[3] + (static_cast<ft>(1.0f) - s) * f_updated[7];
            f[index + 8 * num_values] = s * f_updated[4] + (static_cast<ft>(1.0f) - s) * f_updated[8];
            // clang-format on
        },
        args...);
}

template <typename... Args>
void stream_and_bounce(dim3 *blocks, dim3 *threads, Args... args) {
    LAUNCH_KERNEL(
        gpu::loop_kernel, *blocks, *threads, 0, 0, "stream_and_bounce",
        [] __device__(int index, int num_values, int nx, int ny, auto *f,
                      int *nodetype, auto *ex, auto *ey) {
            using ft =
                std::remove_cv_t<std::remove_reference_t<decltype(f[0])>>;

            const auto s1 = static_cast<ft>(nodetype[index] <= 0);
            // i over ny, j over nx
            const auto i = index / nx;
            const auto j = index % nx;

            for (auto k = 1; k < 5; k++) {
                const auto next_i = (ny + i - static_cast<int>(ey[k])) % ny;
                const auto next_j = (nx + j + static_cast<int>(ex[k])) % nx;
                const auto index2 = next_i * nx + next_j;
                const auto s2 = static_cast<ft>(nodetype[index2] <= 0);

                const auto linear_index1 = index + (k + 4) * num_values;
                const auto linear_index2 = index2 + k * num_values;
                const auto f1 = f[linear_index1];
                const auto f2 = f[linear_index2];

                // s == 0 or s == 1
                const auto s = s1 * s2;
                f[linear_index1] = s * f2 + (static_cast<ft>(1.0f) - s) * f1;
                f[linear_index2] = s * f1 + (static_cast<ft>(1.0f) - s) * f2;
            }
        },
        args...);
}

template <typename... Args>
void compute_macro_vars(dim3 *blocks, dim3 *threads, Args... args) {
    LAUNCH_KERNEL(
        gpu::loop_kernel, *blocks, *threads, 0, 0, "compute_macro_vars",
        [] __device__(int index, int num_values, auto *f, auto *rho, auto *u,
                      int *nodetype, auto *ex, auto *ey) {
            using ft =
                std::remove_cv_t<std::remove_reference_t<decltype(f[0])>>;

            auto rho_i = static_cast<ft>(0.0f);
            auto f_dot_ex = static_cast<ft>(0.0f);
            auto f_dot_ey = static_cast<ft>(0.0f);
            for (int i = 0; i < 9; i++) {
                const auto fi = f[index + i * num_values];
                rho_i += fi;
                f_dot_ex += ex[i] * fi;
                f_dot_ey += ey[i] * fi;
            }

            const auto s = static_cast<ft>(nodetype[index] <= 0);
            const auto inv_rho = std::min(static_cast<ft>(1.0f) / rho_i,
                                          std::numeric_limits<ft>::max());

            rho[index] = s * rho_i;
            u[index + 0 * num_values] = s * f_dot_ex * inv_rho;
            u[index + 1 * num_values] = s * f_dot_ey * inv_rho;
        },
        args...);
}
} // namespace lbm

// This is the python API
extern "C" {
void LBM_compute_edf_f32(dim3 *blocks, dim3 *threads, int nx, int ny, float *f,
                         float *rho, float *u, int *nodetype, float *ex,
                         float *ey, float *w, float es);
void LBM_compute_edf_f64(dim3 *blocks, dim3 *threads, int nx, int ny, double *f,
                         double *rho, double *u, int *nodetype, double *ex,
                         double *ey, double *w, double es);
void LBM_collide_f32(dim3 *blocks, dim3 *threads, int nx, int ny, float *f,
                     float *rho, float *u, int *nodetype, float *tau,
                     float *Fg);
void LBM_collide_f64(dim3 *blocks, dim3 *threads, int nx, int ny, double *f,
                     double *rho, double *u, int *nodetype, double *tau,
                     double *Fg);
void LBM_stream_and_bounce_f32(dim3 *blocks, dim3 *threads, int nx, int ny,
                               float *f, int *nodetype, float *ex, float *ey);
void LBM_stream_and_bounce_f64(dim3 *blocks, dim3 *threads, int nx, int ny,
                               double *f, int *nodetype, double *ex,
                               double *ey);
void LBM_compute_macro_vars_f32(dim3 *blocks, dim3 *threads, int nx, int ny,
                                float *f, float *rho, float *u, int *nodetype,
                                float *ex, float *ey);
void LBM_compute_macro_vars_f64(dim3 *blocks, dim3 *threads, int nx, int ny,
                                double *f, double *rho, double *u,
                                int *nodetype, double *ex, double *ey);
void *LBM_malloc(size_t bytes);
void LBM_memcpy(void *dst, void *src, size_t bytes);
void LBM_free(void *ptr);
void LBM_synchronize();
}
