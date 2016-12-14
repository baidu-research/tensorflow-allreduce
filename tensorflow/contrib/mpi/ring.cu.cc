
#if GOOGLE_CUDA

#define EIGEN_USE_GPU

#include "tensorflow/contrib/mpi/ring.h"

namespace tensorflow {
namespace contrib {
namespace mpi {

using CPUDevice = Eigen::ThreadPoolDevice;

template<> MPI_Datatype MPIType<float>() { return MPI_FLOAT; };
template<> MPI_Datatype MPIType<int>() { return MPI_INT; };

template<> DataType TensorFlowDataType<float>() { return DT_FLOAT; };
template<> DataType TensorFlowDataType<int>() { return DT_INT32; };

// Generate all necessary specializations for RingAllreduce.
template Status RingAllreduce<GPUDevice, int>(OpKernelContext*, Tensor&, Tensor*);
template Status RingAllreduce<GPUDevice, float>(OpKernelContext*, Tensor&, Tensor*);

// Generate all necessary specializations for RingAllgather.
template Status RingAllgather<GPUDevice, int>(
    OpKernelContext*, Tensor&, Tensor*, std::vector<size_t>&);
template Status RingAllgather<GPUDevice, float>(
    OpKernelContext*, Tensor&, Tensor*, std::vector<size_t>&);

// Synchronously copy data on the GPU, using a different stream than the default
// and than TensorFlow to avoid synchronizing on operations unrelated to the
// allreduce.
template<> void CopyTensorData<GPUDevice>(void* dst, void* src, size_t size) {
    auto stream = CudaStreamForMPI();
    cudaMemcpyAsync(dst, src, size, cudaMemcpyDeviceToDevice, stream);
    cudaStreamSynchronize(stream);
};

// Elementwise accumulation kernel for GPU.
template <typename T>
__global__ void elemwise_accum(T* out, const T* in, const size_t N) {
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x;
       i < N;
       i += blockDim.x * gridDim.x) {
      out[i] += in[i];
    }
}

// Synchronously accumulate tensors on the GPU, using a different stream than
// the default and than TensorFlow to avoid synchronizing on operations
// unrelated to the allreduce.
template<> void AccumulateTensorData<GPUDevice, float>(
        float* dst, float* src, size_t size) {
    auto stream = CudaStreamForMPI();
    elemwise_accum<float><<<32, 256, 0, stream>>>(dst, src, size);
    cudaStreamSynchronize(stream);
};
template<> void AccumulateTensorData<GPUDevice, int>(
        int* dst, int* src, size_t size) {
    auto stream = CudaStreamForMPI();
    elemwise_accum<int><<<32, 256, 0, stream>>>(dst, src, size);
    cudaStreamSynchronize(stream);
};

}
}
}
#endif
