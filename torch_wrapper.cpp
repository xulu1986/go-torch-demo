#include <torch/script.h>
#include <iostream>
#include <sstream>
#include <memory>

extern "C" {

// Load a TorchScript model from memory buffer
void* load_torch_module_from_buffer(const char* buffer, long long size) {
    try {
        // Create a string stream from the buffer
        std::string model_data(buffer, size);
        std::istringstream stream(model_data);
        
        // Load the model from the stream
        torch::jit::script::Module* module = new torch::jit::script::Module(torch::jit::load(stream));
        module->eval(); // Set to evaluation mode
        
        return static_cast<void*>(module);
    } catch (const std::exception& e) {
        std::cerr << "Error loading model from buffer: " << e.what() << std::endl;
        return nullptr;
    }
}

// Free a TorchScript module
void free_torch_module(void* module) {
    if (module) {
        delete static_cast<torch::jit::script::Module*>(module);
    }
}

// Create a tensor from float data
void* create_tensor_from_data(float* data, long long* dims, int ndims) {
    try {
        std::vector<int64_t> sizes(dims, dims + ndims);
        
        // Create tensor options
        auto options = torch::TensorOptions().dtype(torch::kFloat32);
        
        // Create tensor from data (copy the data)
        torch::Tensor tensor = torch::from_blob(data, sizes, options).clone();
        
        return static_cast<void*>(new torch::Tensor(tensor));
    } catch (const std::exception& e) {
        std::cerr << "Error creating tensor: " << e.what() << std::endl;
        return nullptr;
    }
}

// Create an integer tensor from float data (for categorical features)
void* create_int_tensor_from_data(float* data, long long* dims, int ndims) {
    try {
        std::vector<int64_t> sizes(dims, dims + ndims);
        
        // Calculate total number of elements
        int64_t total_elements = 1;
        for (int i = 0; i < ndims; i++) {
            total_elements *= dims[i];
        }
        
        // Convert float data to int64 data
        std::vector<int64_t> int_data(total_elements);
        for (int64_t i = 0; i < total_elements; i++) {
            int_data[i] = static_cast<int64_t>(data[i]);
        }
        
        // Create tensor options for int64
        auto options = torch::TensorOptions().dtype(torch::kLong);
        
        // Create tensor from int data
        torch::Tensor tensor = torch::from_blob(int_data.data(), sizes, options).clone();
        
        return static_cast<void*>(new torch::Tensor(tensor));
    } catch (const std::exception& e) {
        std::cerr << "Error creating int tensor: " << e.what() << std::endl;
        return nullptr;
    }
}

// Forward pass through the module with two inputs
void* forward_module(void* module, void* numerical_input, void* categorical_input) {
    try {
        torch::jit::script::Module* mod = static_cast<torch::jit::script::Module*>(module);
        torch::Tensor* numerical_tensor = static_cast<torch::Tensor*>(numerical_input);
        torch::Tensor* categorical_tensor = static_cast<torch::Tensor*>(categorical_input);
        
        // Prepare inputs for forward pass
        std::vector<torch::jit::IValue> inputs;
        inputs.push_back(*numerical_tensor);
        inputs.push_back(*categorical_tensor);
        
        // Forward pass
        torch::jit::IValue output = mod->forward(inputs);
        torch::Tensor output_tensor = output.toTensor();
        
        return static_cast<void*>(new torch::Tensor(output_tensor));
    } catch (const std::exception& e) {
        std::cerr << "Error in forward pass: " << e.what() << std::endl;
        return nullptr;
    }
}

// Get tensor data pointer
float* get_tensor_data(void* tensor) {
    try {
        torch::Tensor* t = static_cast<torch::Tensor*>(tensor);
        return t->data_ptr<float>();
    } catch (const std::exception& e) {
        std::cerr << "Error getting tensor data: " << e.what() << std::endl;
        return nullptr;
    }
}

// Get number of elements in tensor
long long get_tensor_numel(void* tensor) {
    try {
        torch::Tensor* t = static_cast<torch::Tensor*>(tensor);
        return t->numel();
    } catch (const std::exception& e) {
        std::cerr << "Error getting tensor numel: " << e.what() << std::endl;
        return 0;
    }
}

// Free a tensor
void free_tensor(void* tensor) {
    if (tensor) {
        delete static_cast<torch::Tensor*>(tensor);
    }
}

} // extern "C" 