package main

/*
#cgo CFLAGS: -I/opt/homebrew/Cellar/pytorch/2.5.1_4/libexec/lib/python3.13/site-packages/torch/include -I/opt/homebrew/Cellar/pytorch/2.5.1_4/libexec/lib/python3.13/site-packages/torch/include/torch/csrc/api/include
#cgo CXXFLAGS: -I/opt/homebrew/Cellar/pytorch/2.5.1_4/libexec/lib/python3.13/site-packages/torch/include -I/opt/homebrew/Cellar/pytorch/2.5.1_4/libexec/lib/python3.13/site-packages/torch/include/torch/csrc/api/include -std=c++17
#cgo LDFLAGS: -L/opt/homebrew/Cellar/pytorch/2.5.1_4/libexec/lib/python3.13/site-packages/torch/lib -ltorch -ltorch_cpu -lc10 -lstdc++ torch_wrapper.o
#include <stdlib.h>

// Simple C API wrapper for PyTorch C++
typedef void* torch_module_t;
typedef void* torch_tensor_t;

// C wrapper functions - will link with actual libtorch
extern torch_module_t load_torch_module_from_buffer(const char* buffer, long long size);
extern void free_torch_module(torch_module_t module);
extern torch_tensor_t create_tensor_from_data(float* data, long long* dims, int ndims);
extern torch_tensor_t create_int_tensor_from_data(float* data, long long* dims, int ndims);
extern torch_tensor_t forward_module(torch_module_t module, torch_tensor_t numerical_input, torch_tensor_t categorical_input);
extern float* get_tensor_data(torch_tensor_t tensor);
extern long long get_tensor_numel(torch_tensor_t tensor);
extern void free_tensor(torch_tensor_t tensor);
*/
import "C"

import (
	"fmt"
	"unsafe"
)

// TorchTensor wraps the C torch_tensor_t
type TorchTensor struct {
	ptr C.torch_tensor_t
}

// TorchModule wraps the C torch_module_t
type TorchModule struct {
	ptr C.torch_module_t
}

// loadTorchModuleFromBytes loads a PyTorch module from memory
func loadTorchModuleFromBytes(modelBytes []byte) (*TorchModule, error) {
	if len(modelBytes) == 0 {
		return nil, fmt.Errorf("model bytes are empty")
	}

	// Convert Go byte slice to C buffer
	cBuffer := (*C.char)(unsafe.Pointer(&modelBytes[0]))
	size := C.longlong(len(modelBytes))

	ptr := C.load_torch_module_from_buffer(cBuffer, size)
	if ptr == nil {
		return nil, fmt.Errorf("failed to load torch module from buffer (%d bytes)", len(modelBytes))
	}

	return &TorchModule{ptr: ptr}, nil
}

// Forward performs forward inference with the module
func (m *TorchModule) Forward(numericalInput *TorchTensor, categoricalInput *TorchTensor) (*TorchTensor, error) {
	if m.ptr == nil {
		return nil, fmt.Errorf("module is nil")
	}
	if numericalInput.ptr == nil {
		return nil, fmt.Errorf("numerical input tensor is nil")
	}
	if categoricalInput.ptr == nil {
		return nil, fmt.Errorf("categorical input tensor is nil")
	}

	outputPtr := C.forward_module(m.ptr, numericalInput.ptr, categoricalInput.ptr)
	if outputPtr == nil {
		return nil, fmt.Errorf("forward pass failed")
	}

	return &TorchTensor{ptr: outputPtr}, nil
}

// Free releases the module memory
func (m *TorchModule) Free() {
	if m.ptr != nil {
		C.free_torch_module(m.ptr)
		m.ptr = nil
	}
}

// ToFloat64Slice converts tensor to Go float64 slice
func (t *TorchTensor) ToFloat64Slice() ([]float64, error) {
	if t.ptr == nil {
		return nil, fmt.Errorf("tensor is nil")
	}

	// Get tensor size
	numel := int(C.get_tensor_numel(t.ptr))
	if numel <= 0 {
		return nil, fmt.Errorf("tensor has no elements")
	}

	// Get data pointer
	dataPtr := C.get_tensor_data(t.ptr)
	if dataPtr == nil {
		return nil, fmt.Errorf("failed to get tensor data pointer")
	}

	// Convert C float array to Go slice
	floatSlice := (*[1 << 30]C.float)(unsafe.Pointer(dataPtr))[:numel:numel]
	result := make([]float64, numel)
	for i, v := range floatSlice {
		result[i] = float64(v)
	}

	return result, nil
}

// Free releases the tensor memory
func (t *TorchTensor) Free() {
	if t.ptr != nil {
		C.free_tensor(t.ptr)
		t.ptr = nil
	}
}

// createTensorFromData creates a tensor from float32 data
func createTensorFromData(data []float32, dims []int64) (*TorchTensor, error) {
	if len(data) == 0 {
		return nil, fmt.Errorf("data is empty")
	}

	cDims := make([]C.longlong, len(dims))
	for i, d := range dims {
		cDims[i] = C.longlong(d)
	}

	ptr := C.create_tensor_from_data((*C.float)(&data[0]), &cDims[0], C.int(len(dims)))
	if ptr == nil {
		return nil, fmt.Errorf("failed to create tensor")
	}

	return &TorchTensor{ptr: ptr}, nil
}

// createIntTensorFromData creates an integer tensor from float32 data
func createIntTensorFromData(data []float32, dims []int64) (*TorchTensor, error) {
	if len(data) == 0 {
		return nil, fmt.Errorf("data is empty")
	}

	cDims := make([]C.longlong, len(dims))
	for i, d := range dims {
		cDims[i] = C.longlong(d)
	}

	ptr := C.create_int_tensor_from_data((*C.float)(&data[0]), &cDims[0], C.int(len(dims)))
	if ptr == nil {
		return nil, fmt.Errorf("failed to create integer tensor")
	}

	return &TorchTensor{ptr: ptr}, nil
}
