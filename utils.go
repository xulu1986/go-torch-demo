package main

import (
	"encoding/json"
	"fmt"
	"io/ioutil"
)

// loadModelData loads and parses the model data from JSON file
func loadModelData(filePath string) (*ModelData, *TorchModelData, error) {
	// Read the JSON file
	data, err := ioutil.ReadFile(filePath)
	if err != nil {
		return nil, nil, fmt.Errorf("failed to read file: %w", err)
	}

	// Parse the outer JSON
	var modelData ModelData
	if err := json.Unmarshal(data, &modelData); err != nil {
		return nil, nil, fmt.Errorf("failed to parse outer JSON: %w", err)
	}

	// Parse the inner JSON
	var torchData TorchModelData
	if err := json.Unmarshal([]byte(modelData.Data), &torchData); err != nil {
		return nil, nil, fmt.Errorf("failed to parse inner JSON: %w", err)
	}

	return &modelData, &torchData, nil
}

// getFeatureValue extracts a feature value from the dynamic ValidationData map
func getFeatureValue(sample ValidationData, featureName string) (interface{}, error) {
	value, exists := sample[featureName]
	if !exists {
		return nil, fmt.Errorf("feature %s not found in sample", featureName)
	}
	return value, nil
}

// convertToFloat32 converts various numeric types to float32
func convertToFloat32(value interface{}) float32 {
	switch v := value.(type) {
	case float64:
		return float32(v)
	case float32:
		return v
	case int:
		return float32(v)
	case int64:
		return float32(v)
	case string:
		// Try to parse string as float
		if f, err := fmt.Sscanf(v, "%f", new(float64)); err == nil && f == 1 {
			var result float64
			fmt.Sscanf(v, "%f", &result)
			return float32(result)
		}
		return 0.0
	default:
		return 0.0
	}
}

// encodeCategorical encodes categorical values using a simple mapping
func encodeCategorical(value string, mapping map[string]int) int {
	if encoded, exists := mapping[value]; exists {
		return encoded
	}
	// Return 0 for unknown categories
	return 0
}

// encodeCategoricalFromEncoder encodes categorical values using label encoder
func encodeCategoricalFromEncoder(value string, encoder LabelEncoder) int {
	for i, class := range encoder.Classes {
		if class == value {
			return i
		}
	}
	// Return 0 for unknown categories (this should be rare if validation data is clean)
	fmt.Printf("WARNING: Unknown categorical value '%s' not found in encoder classes, using index 0\n", value)
	return 0
}

// truncateString truncates a string to a maximum length
func truncateString(s string, maxLen int) string {
	if len(s) <= maxLen {
		return s
	}
	return s[:maxLen]
}
