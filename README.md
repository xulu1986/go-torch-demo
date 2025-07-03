# PyTorch Model Inference Demo

This Go program demonstrates **real PyTorch model inference** using a TorchScript model stored in JSON format. It uses custom CGO bindings to load and run TorchScript models directly from Go.

## 🏗️ **Architecture**

```
JSON Model File → Base64 Decode → TorchScript Load → Feature Encoding → Inference → Output
     ↓                ↓              ↓                    ↓              ↓           ↓
   428KB          Memory Buffer   PyTorch Model     Dynamic Features  Forward Pass  Predictions
```

### **Code Structure**
```
go-torch-demo/
├── main.go              # Main application logic
├── torch_bindings.go    # CGO bindings (update CGO flags here)
├── types.go             # Data structures
├── features.go          # Dynamic feature processing
├── utils.go             # Utility functions
├── torch_wrapper.cpp    # C++ wrapper for PyTorch API
├── data/
│   └── model.json       # TorchScript model + validation data
├── go.mod               # Go module definition
└── README.md            # This file
```

## 📋 **Prerequisites**

### Install PyTorch (libtorch)
```bash
# macOS (using Homebrew)
brew install pytorch
```

### Go Requirements
- Go 1.21 or later
- CGO enabled
- C++ compiler (clang++ or g++)

## ⚙️ **Setup**

### 1. Find Your PyTorch Installation

### 2. Update CGO Flags
Edit `torch_bindings.go` and update the hardcoded CGO flags with your PyTorch path:

```go
/*
#cgo CFLAGS: -I/YOUR/PYTORCH/PATH/include -I/YOUR/PYTORCH/PATH/include/torch/csrc/api/include
#cgo CXXFLAGS: -I/YOUR/PYTORCH/PATH/include -I/YOUR/PYTORCH/PATH/include/torch/csrc/api/include -std=c++17
#cgo LDFLAGS: -L/YOUR/PYTORCH/PATH/lib -ltorch -ltorch_cpu -lc10 -lstdc++ torch_wrapper.o
*/
```

### 3. Compile C++ Wrapper
```bash
# Replace with your actual PyTorch path
PYTORCH_PATH="/opt/homebrew/Cellar/pytorch/2.5.1_4/libexec/lib/python3.13/site-packages/torch"

clang++ -c -fPIC torch_wrapper.cpp -o torch_wrapper.o \
  -I$PYTORCH_PATH/include \
  -I$PYTORCH_PATH/include/torch/csrc/api/include \
  -std=c++17
```

### 4. Run the Demo
```bash
go run *.go
```
