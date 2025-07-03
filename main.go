package main

import (
	"encoding/base64"
	"fmt"
	"log"
	"math"
	"strings"
)

func main() {
	fmt.Println("=== PyTorch Model Inference Demo ===")

	// Load and parse the JSON file
	modelData, torchData, err := loadModelData("Data/model.json")
	if err != nil {
		log.Fatalf("Failed to load model data: %v", err)
	}

	fmt.Printf("\nModel Information:\n")
	fmt.Printf("- Interval: %d\n", modelData.Interval)
	fmt.Printf("- Task Type: %s\n", torchData.TaskType)
	fmt.Printf("- Learning Rate: %f\n", torchData.LearningRate)
	fmt.Printf("- Epochs: %d\n", torchData.Epochs)
	fmt.Printf("- Validation Samples: %d\n", len(torchData.ValidationData))
	fmt.Printf("- Expected Predictions: %d\n", len(torchData.ValidationPredictions))

	fmt.Printf("\nFeature Mappings:\n")
	fmt.Printf("- Platforms: %d categories (indices 0-%d)\n", len(torchData.FeatureInfo.MissingValueHandling.LabelEncoders["platform"].Classes), len(torchData.FeatureInfo.MissingValueHandling.LabelEncoders["platform"].Classes)-1)
	fmt.Printf("- Geos: %d categories (indices 0-%d)\n", len(torchData.FeatureInfo.MissingValueHandling.LabelEncoders["geo"].Classes), len(torchData.FeatureInfo.MissingValueHandling.LabelEncoders["geo"].Classes)-1)
	fmt.Printf("- Placement Types: %d categories (indices 0-%d)\n", len(torchData.FeatureInfo.MissingValueHandling.LabelEncoders["placement_type"].Classes), len(torchData.FeatureInfo.MissingValueHandling.LabelEncoders["placement_type"].Classes)-1)
	fmt.Printf("- Pub App Objects: %d categories (indices 0-%d)\n", len(torchData.FeatureInfo.MissingValueHandling.LabelEncoders["pub_app_object_id"].Classes), len(torchData.FeatureInfo.MissingValueHandling.LabelEncoders["pub_app_object_id"].Classes)-1)

	// Decode the PyTorch model
	modelBytes, err := base64.StdEncoding.DecodeString(torchData.TorchModel.Model)
	if err != nil {
		log.Fatalf("Failed to decode model: %v", err)
	}

	fmt.Printf("\nModel decoded: %d bytes\n", len(modelBytes))

	// Load the PyTorch model directly from bytes using our custom bindings
	fmt.Printf("\nLoading PyTorch model from memory...\n")
	model, err := loadTorchModuleFromBytes(modelBytes)
	if err != nil {
		log.Fatalf("Failed to load PyTorch model: %v", err)
	}
	defer model.Free()

	fmt.Printf("Model loaded successfully!\n")

	// Prepare input data from validation samples
	fmt.Printf("\nPreparing validation data...\n")
	numericalTensor, categoricalTensor, err := prepareValidationInput(torchData.ValidationData, torchData.FeatureInfo)
	if err != nil {
		log.Fatalf("Failed to prepare input tensors: %v", err)
	}
	defer numericalTensor.Free()
	defer categoricalTensor.Free()

	fmt.Printf("Numerical tensor prepared with shape: [%d, %d]\n", len(torchData.ValidationData), len(torchData.FeatureInfo.FeatureNames["numerical"]))
	fmt.Printf("Categorical tensor prepared with shape: [%d, %d]\n", len(torchData.ValidationData), len(torchData.FeatureInfo.FeatureNames["categorical"]))

	// Perform forward inference
	fmt.Printf("\nPerforming forward inference...\n")
	outputTensor, err := model.Forward(numericalTensor, categoricalTensor)
	if err != nil {
		log.Fatalf("Failed to perform forward inference: %v", err)
	}
	defer outputTensor.Free()

	// Extract predictions
	predictions, err := outputTensor.ToFloat64Slice()
	if err != nil {
		log.Fatalf("Failed to extract predictions: %v", err)
	}

	// Display results and perform strict validation
	fmt.Printf("\n=== Validation Results ===\n")
	fmt.Printf("%-3s %-12s %-8s %-4s %-8s %-12s %-20s %-12s %-12s %-12s %-8s\n",
		"#", "RTB_ID", "Platform", "Geo", "DNT", "OS_Ver", "Placement", "Predicted", "Expected", "Error", "Match")
	fmt.Printf("%s\n", strings.Repeat("-", 140))

	var totalSquaredError float64
	var totalAbsError float64
	var exactMatches int
	var closeMatches int
	tolerance := 1e-6 // Very small tolerance for floating point comparison

	for i := 0; i < len(predictions) && i < len(torchData.ValidationPredictions); i++ {
		predicted := predictions[i]
		expected := torchData.ValidationPredictions[i]
		error := predicted - expected
		absError := math.Abs(error)
		squaredError := error * error

		totalSquaredError += squaredError
		totalAbsError += absError

		// Check for exact or near-exact matches
		var matchStatus string
		if absError == 0.0 {
			matchStatus = "EXACT"
			exactMatches++
		} else if absError < tolerance {
			matchStatus = "CLOSE"
			closeMatches++
		} else {
			matchStatus = "DIFF"
		}

		sample := torchData.ValidationData[i]
		fmt.Printf("%-3d %-12s %-8s %-4s %-8s %-12s %-20s %-12.6f %-12.6f %-12.6f %-8s\n",
			i+1,
			truncateString(fmt.Sprintf("%v", sample["rtb_id"]), 12),
			fmt.Sprintf("%v", sample["platform"]),
			fmt.Sprintf("%v", sample["geo"]),
			fmt.Sprintf("%v", sample["do_not_track"]),
			fmt.Sprintf("%v", sample["major_os_version"]),
			truncateString(fmt.Sprintf("%v", sample["placement_type"]), 20),
			predicted,
			expected,
			error,
			matchStatus)
	}

	// Calculate and display metrics
	numSamples := float64(len(predictions))
	mse := totalSquaredError / numSamples
	rmse := math.Sqrt(mse)
	mae := totalAbsError / numSamples

	fmt.Printf("\n=== Performance Metrics ===\n")
	fmt.Printf("MSE (Mean Squared Error): %.10f\n", mse)
	fmt.Printf("RMSE (Root Mean Squared Error): %.10f\n", rmse)
	fmt.Printf("MAE (Mean Absolute Error): %.10f\n", mae)
	fmt.Printf("Number of samples: %.0f\n", numSamples)

	fmt.Printf("\n=== Validation Summary ===\n")
	fmt.Printf("Exact matches: %d/%d (%.1f%%)\n", exactMatches, len(predictions), float64(exactMatches)/float64(len(predictions))*100)
	fmt.Printf("Close matches (< %.0e): %d/%d (%.1f%%)\n", tolerance, closeMatches, len(predictions), float64(closeMatches)/float64(len(predictions))*100)
	fmt.Printf("Different values: %d/%d (%.1f%%)\n", len(predictions)-exactMatches-closeMatches, len(predictions), float64(len(predictions)-exactMatches-closeMatches)/float64(len(predictions))*100)

	// Final validation check
	if exactMatches == len(predictions) {
		fmt.Printf("\n✅ SUCCESS: All predictions match exactly with Python validation outputs!\n")
	} else if exactMatches+closeMatches == len(predictions) {
		fmt.Printf("\n⚠️  CLOSE: All predictions are very close to Python validation outputs (within %.0e tolerance)\n", tolerance)
	} else {
		fmt.Printf("\n❌ WARNING: %d predictions differ significantly from Python validation outputs\n", len(predictions)-exactMatches-closeMatches)
		fmt.Printf("This may indicate issues with:\n")
		fmt.Printf("- Model loading or deserialization\n")
		fmt.Printf("- Input preprocessing or feature encoding\n")
		fmt.Printf("- Tensor shape or data type mismatches\n")
		fmt.Printf("- Numerical precision differences between Python and Go\n")
	}

	fmt.Printf("\n=== Inference completed! ===\n")
}
