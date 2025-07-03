package main

import (
	"fmt"
)

// prepareValidationInput prepares input tensors from validation data
func prepareValidationInput(validationData []ValidationData, featureInfo FeatureInfo) (*TorchTensor, *TorchTensor, error) {
	if len(validationData) == 0 {
		return nil, nil, fmt.Errorf("no validation data provided")
	}

	batchSize := len(validationData)

	// Get feature names dynamically from model metadata
	numericalFeatures := featureInfo.FeatureNames["numerical"]
	categoricalFeatures := featureInfo.FeatureNames["categorical"]

	numNumericalFeatures := len(numericalFeatures)
	numCategoricalFeatures := len(categoricalFeatures)

	fmt.Printf("Model expects %d numerical features: %v\n", numNumericalFeatures, numericalFeatures)
	fmt.Printf("Model expects %d categorical features: %v\n", numCategoricalFeatures, categoricalFeatures)

	// Prepare feature data arrays
	var numericalData []float32
	var categoricalData []float32

	if numNumericalFeatures > 0 {
		numericalData = make([]float32, batchSize*numNumericalFeatures)
	}
	if numCategoricalFeatures > 0 {
		categoricalData = make([]float32, batchSize*numCategoricalFeatures)
	}

	// Process each sample
	for i, sample := range validationData {
		// Process numerical features
		if numNumericalFeatures > 0 {
			numericalOffset := i * numNumericalFeatures
			for j, featureName := range numericalFeatures {
				value, err := getFeatureValue(sample, featureName)
				if err != nil {
					return nil, nil, fmt.Errorf("failed to get numerical feature %s: %v", featureName, err)
				}
				numericalData[numericalOffset+j] = convertToFloat32(value)
			}
		}

		// Process categorical features
		if numCategoricalFeatures > 0 {
			categoricalOffset := i * numCategoricalFeatures
			var categoricalIndices []int

			for j, featureName := range categoricalFeatures {
				encoder, exists := featureInfo.MissingValueHandling.LabelEncoders[featureName]
				if !exists {
					return nil, nil, fmt.Errorf("no encoder found for categorical feature: %s", featureName)
				}

				value, err := getFeatureValue(sample, featureName)
				if err != nil {
					return nil, nil, fmt.Errorf("failed to get categorical feature %s: %v", featureName, err)
				}

				valueStr := fmt.Sprintf("%v", value) // Convert to string for encoding
				index := encodeCategoricalFromEncoder(valueStr, encoder)
				categoricalData[categoricalOffset+j] = float32(index)
				categoricalIndices = append(categoricalIndices, index)
			}

			// Debug: Show first few samples
			if i < 3 {
				fmt.Printf("Sample %d categorical indices: %v\n", i+1, categoricalIndices)
			}
		}
	}

	// Create numerical tensor
	var numericalTensor *TorchTensor
	var err error

	if numNumericalFeatures > 0 {
		numericalDims := []int64{int64(batchSize), int64(numNumericalFeatures)}
		numericalTensor, err = createTensorFromData(numericalData, numericalDims)
		if err != nil {
			return nil, nil, fmt.Errorf("failed to create numerical tensor: %v", err)
		}
	} else {
		// Create empty tensor for models with no numerical features
		numericalDims := []int64{int64(batchSize), 0}
		dummyNumericalData := make([]float32, 1) // Minimum size for tensor creation
		numericalTensor, err = createTensorFromData(dummyNumericalData, numericalDims)
		if err != nil {
			return nil, nil, fmt.Errorf("failed to create empty numerical tensor: %v", err)
		}
	}

	// Create categorical tensor
	var categoricalTensor *TorchTensor

	if numCategoricalFeatures > 0 {
		categoricalDims := []int64{int64(batchSize), int64(numCategoricalFeatures)}
		categoricalTensor, err = createIntTensorFromData(categoricalData, categoricalDims)
		if err != nil {
			return nil, nil, fmt.Errorf("failed to create categorical tensor: %v", err)
		}
	} else {
		// Create empty tensor for models with no categorical features
		categoricalDims := []int64{int64(batchSize), 0}
		dummyCategoricalData := make([]float32, 1) // Minimum size for tensor creation
		categoricalTensor, err = createIntTensorFromData(dummyCategoricalData, categoricalDims)
		if err != nil {
			return nil, nil, fmt.Errorf("failed to create empty categorical tensor: %v", err)
		}
	}

	return numericalTensor, categoricalTensor, nil
}
