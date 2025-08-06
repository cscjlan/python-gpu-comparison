#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <hip/hip_runtime.h>
#include <limits>
#include <sstream>
#include <vector>

namespace constants {
static constexpr float ex[9] = {
    0.0f, 1.0f, 0.0f, 1.0f, 1.0f, -1.0f, 0.0f, -1.0f, -1.0f,
};

static constexpr float ey[9] = {
    0.0f, 0.0f, 1.0f, 1.0f, -1.0f, 0.0f, -1.0f, -1.0f, 1.0f,
};

static constexpr float w[9] = {
    4.0f / 9.0f, 1.0f / 9.0f, 1.0f / 9.0f,  1.0f / 36.0f, 1.0f / 36.0f,
    1.0f / 9.0f, 1.0f / 9.0f, 1.0f / 36.0f, 1.0f / 36.0f,
};

// This is the same as 1 / ((1 / 2 * 1 / 3)^2)
static constexpr float inv_es_sq = 36.0f;
} // namespace constants

#define LAUNCH_KERNEL(kernel, ...)                                             \
    gpu::launch_kernel(#kernel, __FILE__, __LINE__, kernel, __VA_ARGS__)
#define HIP_ERRCHK(result) gpu::hip_errchk(result, __FILE__, __LINE__)

namespace gpu {
template <typename F, typename... Args>
__global__ void loop_kernel(F f, size_t num_values, Args... args) {
    const size_t tid = threadIdx.x + blockIdx.x * blockDim.x;
    const size_t stride = blockDim.x * gridDim.x;

    for (size_t index = tid; index < num_values; index += stride) {
        f(index, num_values, args...);
    }
}

static inline void hip_errchk(hipError_t result, const char *file,
                              int32_t line) {
    if (result != hipSuccess) {
        printf("\n\n%s in %s at line %d\n", hipGetErrorString(result), file,
               line);
        exit(EXIT_FAILURE);
    }
}

template <typename... Args>
void launch_kernel(const char *kernel_name, const char *file, int32_t line,
                   void (*kernel)(Args...), dim3 blocks, dim3 threads,
                   size_t num_bytes_shared_mem, hipStream_t stream,
                   const char *lambda_name, Args... args) {
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

    auto error_print_prelude = [&kernel_name, &lambda_name]() {
        std::fprintf(
            stderr,
            "Bad launch parameters for kernel \"%s\" with lambda \"%s\"\n",
            kernel_name, lambda_name);
    };
    // Helper lambda for asserting dim3 launch variable is within allowed limits
    auto assert_within_limits = [&error_print_prelude](
                                    const char *name, int32_t value,
                                    int32_t min, int32_t max) {
        if (not(min <= value && value <= max)) {
            error_print_prelude();
            std::fprintf(stderr, "%s (%d) not within limits [%d, %d]\n", name,
                         value, min, max);
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
    [[maybe_unused]] result = hipDeviceSynchronize();
#endif
    result = hipGetLastError();
    if (result != hipSuccess) {
        printf("Error with kernel \"%s\" executing lambda \"%s\" in %s at line "
               "%d\n%s: %s\n",
               kernel_name, lambda_name, file, line, hipGetErrorName(result),
               hipGetErrorString(result));
        exit(EXIT_FAILURE);
    }
#endif
}

void *allocate(size_t num_bytes) {
    void *ptr = nullptr;
    HIP_ERRCHK(hipMalloc(&ptr, num_bytes));
    if (ptr == nullptr) {
        std::fprintf(stderr, "GPU malloc allocated a nullptr\n");
        std::abort();
    }

    return ptr;
}

void free(void *ptr) { HIP_ERRCHK(hipFree(ptr)); }

void memcpy(void *dst, const void *src, size_t num_bytes) {
    HIP_ERRCHK(hipMemcpy(dst, src, num_bytes, hipMemcpyDefault));
}
} // namespace gpu

template <typename C> void write_to_file(C container, const char *fname) {
    std::stringstream ss;

    if (!container.empty()) {
        ss << container[0];
        for (size_t i = 1; i < container.size(); i++) {
            ss << "," << container[i];
        }
        ss << "\n";
    }

    std::ofstream file(fname);
    if (file.is_open()) {
        file << ss.str();
    } else {
        std::fprintf(stderr, "Failed to open file %s for writing data\n",
                     fname);
        std::abort();
    }
}

int32_t main(int32_t argc, char **argv) {
    if (argc != 3) {
        std::printf("Give two arguments: %s nx ny\n", argv[0]);
        return EXIT_FAILURE;
    }

    const size_t nx = std::atoi(argv[1]);
    const size_t ny = std::atoi(argv[2]);
    static constexpr size_t num_iters = 400ul;

    const size_t num_values = nx * ny;
    const size_t num_bytes = num_values * sizeof(float);

    float *rho = static_cast<float *>(gpu::allocate(num_bytes));
    float *tau = static_cast<float *>(gpu::allocate(num_bytes));
    float *u = static_cast<float *>(gpu::allocate(2 * num_bytes));
    float *fg = static_cast<float *>(gpu::allocate(2 * num_bytes));
    float *f = static_cast<float *>(gpu::allocate(9 * num_bytes));
    int32_t *node_type =
        static_cast<int32_t *>(gpu::allocate(num_values * sizeof(int32_t)));

    static constexpr size_t num_threads = 1024;
    static constexpr size_t num_blocks = 1024;

    LAUNCH_KERNEL(
        gpu::loop_kernel, num_threads, num_blocks, 0, 0, "initialize",
        [](size_t index, size_t num_values, size_t nx, size_t ny, float *rho,
           float *tau, float *u, float *fg, float *f, int32_t *node_type) {
            rho[index] = 1.0f;
            tau[index] = 1.0f;
            u[index + 0 * num_values] = 0.0f;
            u[index + 1 * num_values] = 0.0f;
            fg[index + 0 * num_values] = 1e-7f;
            fg[index + 1 * num_values] = 0.0f;
            node_type[index] =
                static_cast<int32_t>(index < nx || index >= (ny - 1) * nx);
            for (size_t i = 0; i < 9; i++) {
                f[index + i * num_values] = 0.0f;
            }
        },
        num_values, nx, ny, rho, tau, u, fg, f, node_type);

    LAUNCH_KERNEL(
        gpu::loop_kernel, num_threads, num_blocks, 0, 0, "compute edf",
        [](size_t index, size_t num_values, float *rho, float *u, float *f,
           int32_t *node_type) {
            const float s = static_cast<float>(node_type[index] <= 0);

            static constexpr size_t N = 9;
            for (size_t i = 0; i < N; i++) {
                const float u0 = u[index];
                const float u1 = u[index + num_values];
                const float exi = constants::ex[i];
                const float eyi = constants::ey[i];

                const float ux2 = u0 * u0;
                const float uy2 = u1 * u1;
                const float euxy = exi * eyi * u0 * u1;
                const float euxx = exi * exi * ux2;
                const float euyy = eyi * eyi * uy2;
                const float eu2 = 2.0f * euxy + euxx + euyy;
                const float u2 = ux2 + uy2;

                const float term_order1 =
                    constants::inv_es_sq * (exi * u0 + eyi * u1);
                const float term_order2 = 0.5f * constants::inv_es_sq *
                                          (constants::inv_es_sq * eu2 - u2);

                const float f_old = f[index + i * num_values];
                const float f_new = constants::w[i] * rho[index] *
                                    (1.0f + term_order1 + term_order2);

                f[index + i * num_values] = s * f_new + (1.0f - s) * f_old;
            }
        },
        num_values, rho, u, f, node_type);

    for (size_t i = 0; i < num_iters; i++) {
        LAUNCH_KERNEL(
            gpu::loop_kernel, num_threads, num_blocks, 0, 0, "collide",
            [](size_t index, size_t num_values, float *rho, float *tau,
               float *u, float *fg, float *f, int32_t *node_type) {
                const float s = static_cast<float>(node_type[index] <= 0);

                const float rho_i = rho[index];
                const float tau_i = tau[index];
                const float tau_per_rho =
                    std::min(tau_i / rho_i, std::numeric_limits<float>::max());

                const size_t i = index + 0 * num_values;
                const size_t j = index + 1 * num_values;

                const float u0 = u[i] + s * fg[i] * tau_per_rho;
                const float u1 = u[j] + s * fg[j] * tau_per_rho;
                u[i] = u0;
                u[j] = u1;

                const float u0_sq = u0 * u0;
                const float u1_sq = u1 * u1;
                const float u0_sq_p_u1_sq = 0.5f * (u0_sq + u1_sq);
                const float u0_p_u1 = u0 + u1;
                const float u0_m_u1 = u0 - u1;
                const float u0_p_u1_sq = 1.5f * u0_p_u1 * u0_p_u1;
                const float u0_m_u1_sq = 1.5f * u0_m_u1 * u0_m_u1;

                float f_updated[9] = {
                    -u0_sq - u1_sq + 0.33333333f,
                    -0.5f * u1_sq + u0_sq + u0,
                    -0.5f * u0_sq + u1_sq + u1,
                    -u0_sq_p_u1_sq + u0_p_u1 + u0_p_u1_sq,
                    -u0_sq_p_u1_sq + u0_m_u1 + u0_m_u1_sq,
                    -0.5f * u1_sq + u0_sq - u0,
                    -0.5f * u0_sq + u1_sq - u1,
                    -u0_sq_p_u1_sq - u0_p_u1 + u0_p_u1_sq,
                    -u0_sq_p_u1_sq - u0_m_u1 + u0_m_u1_sq,
                };

                const float rho_per_three = 0.333333f * rho_i;
                const float inv_tau =
                    std::min(1.0f / tau_i, std::numeric_limits<float>::max());
                const float tau_m_1 = tau_i - 1.0f;

                static constexpr size_t N = 9;
                static constexpr float multipliers[N] = {
                    2.00f, 1.00f, 1.00f, 0.25f, 0.25f,
                    1.00f, 1.00f, 0.25f, 0.25f,
                };
                for (size_t i = 0; i < N; i++) {
                    const float f_eq = multipliers[i] * rho_per_three *
                                       (f_updated[i] + 0.333333333f);
                    const float f_old = f[index + i * num_values];
                    const float f_new = inv_tau * (tau_m_1 * f_old + f_eq);

                    f_updated[i] = s * f_new + (1.0f - s) * f_old;
                }

                // clang-format off
                // Update 1-8, such that pairs are swapped:
                // 0 <--> 0
                // 1 <--> 5
                // 2 <--> 6
                // 3 <--> 7
                // 4 <--> 8
                f[index + 0 * num_values] = s * f_updated[0] + (1.0f - s) * f_updated[0];
                f[index + 1 * num_values] = s * f_updated[5] + (1.0f - s) * f_updated[1];
                f[index + 2 * num_values] = s * f_updated[6] + (1.0f - s) * f_updated[2];
                f[index + 3 * num_values] = s * f_updated[7] + (1.0f - s) * f_updated[3];
                f[index + 4 * num_values] = s * f_updated[8] + (1.0f - s) * f_updated[4];
                f[index + 5 * num_values] = s * f_updated[1] + (1.0f - s) * f_updated[5];
                f[index + 6 * num_values] = s * f_updated[2] + (1.0f - s) * f_updated[6];
                f[index + 7 * num_values] = s * f_updated[3] + (1.0f - s) * f_updated[7];
                f[index + 8 * num_values] = s * f_updated[4] + (1.0f - s) * f_updated[8];
                // clang-format on
            },
            num_values, rho, tau, u, fg, f, node_type);

        LAUNCH_KERNEL(
            gpu::loop_kernel, num_threads, num_blocks, 0, 0,
            "stream and bounce",
            [](size_t index, size_t num_values, size_t nx, size_t ny, float *f,
               int32_t *node_type) {
                const float s1 = static_cast<float>(node_type[index] <= 0);
                // i over ny, j over nx
                const size_t i = index / nx;
                const size_t j = index % nx;

                for (size_t k = 1; k < 5; k++) {
                    const size_t next_i =
                        (ny + i - static_cast<size_t>(constants::ey[k])) % ny;
                    const size_t next_j =
                        (nx + j + static_cast<size_t>(constants::ex[k])) % nx;
                    const size_t index2 = next_i * nx + next_j;
                    const float s2 = static_cast<float>(node_type[index2] <= 0);

                    const size_t linear_index1 = index + (k + 4) * num_values;
                    const size_t linear_index2 = index2 + k * num_values;
                    const size_t f1 = f[linear_index1];
                    const size_t f2 = f[linear_index2];

                    // s == 0 or s == 1
                    const float s = s1 * s2;
                    f[linear_index1] = s * f2 + (1.0f - s) * f1;
                    f[linear_index2] = s * f1 + (1.0f - s) * f2;
                }
            },
            num_values, nx, ny, f, node_type);

        LAUNCH_KERNEL(
            gpu::loop_kernel, num_threads, num_blocks, 0, 0,
            "compute macro vars",
            [](size_t index, size_t num_values, float *rho, float *u, float *f,
               int32_t *node_type) {
                const float f_i[9] = {
                    f[index + 0 * num_values], f[index + 1 * num_values],
                    f[index + 2 * num_values], f[index + 3 * num_values],
                    f[index + 4 * num_values], f[index + 5 * num_values],
                    f[index + 6 * num_values], f[index + 7 * num_values],
                    f[index + 8 * num_values],
                };

                const float rho_i = f_i[0] + f_i[1] + f_i[2] + f_i[3] + f_i[4] +
                                    f_i[5] + f_i[6] + f_i[7] + f_i[8];
                const float f_dot_ex =
                    f_i[1] + f_i[3] + f_i[4] - f_i[5] - f_i[7] - f_i[8];
                const float f_dot_ey =
                    f_i[2] + f_i[3] - f_i[4] - f_i[6] - f_i[7] + f_i[8];

                const float s = static_cast<float>(node_type[index] <= 0);
                const float inv_rho =
                    std::min(1.0f / rho_i, std::numeric_limits<float>::max());

                rho[index] = s * rho_i;
                u[index + 0 * num_values] = s * f_dot_ex * inv_rho;
                u[index + 1 * num_values] = s * f_dot_ey * inv_rho;
            },
            num_values, rho, u, f, node_type);
    }

    std::vector<float> h_u(2 * num_values, 0.0f);
    std::vector<float> f_u(9 * num_values, 0.0f);

    gpu::memcpy(h_u.data(), u, 2 * num_bytes);
    gpu::memcpy(f_u.data(), f, 9 * num_bytes);

    write_to_file(h_u, "u.dat");
    write_to_file(f_u, "f.dat");

    gpu::free(rho);
    gpu::free(tau);
    gpu::free(u);
    gpu::free(fg);
    gpu::free(f);
    gpu::free(node_type);

    return EXIT_SUCCESS;
}
