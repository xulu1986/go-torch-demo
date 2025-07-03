package main

// ModelData represents the top-level JSON structure
type ModelData struct {
	Interval int    `json:"interval"`
	Data     string `json:"data"`
}

// TorchModelData represents the inner JSON structure
type TorchModelData struct {
	TorchModel            TorchModel       `json:"torch_model"`
	FeatureInfo           FeatureInfo      `json:"feature_info"`
	TaskType              string           `json:"task_type"`
	LearningRate          float64          `json:"learning_rate"`
	WeightDecay           float64          `json:"weight_decay"`
	Epochs                int              `json:"epochs"`
	BatchSize             int              `json:"batch_size"`
	NumWorkers            int              `json:"num_workers"`
	WeightColumn          string           `json:"weight_column"`
	ValidationSamples     int              `json:"validation_samples"`
	ValidationTolerance   float64          `json:"validation_tolerance"`
	ValidationData        []ValidationData `json:"validation_data"`
	ValidationPredictions []float64        `json:"validation_predictions"`
	TrainingHistory       TrainingHistory  `json:"training_history"`
}

// TorchModel represents the PyTorch model structure
type TorchModel struct {
	Model  string `json:"model"`
	Config string `json:"config"`
}

// FeatureInfo contains categorical feature mappings
type FeatureInfo struct {
	NumNumericalFeatures   int                  `json:"num_numerical_features"`
	NumCategoricalFeatures int                  `json:"num_categorical_features"`
	CategoricalVocabSizes  map[string]int       `json:"categorical_vocab_sizes"`
	FeatureNames           map[string][]string  `json:"feature_names"`
	TargetColumn           string               `json:"target_column"`
	MissingValueHandling   MissingValueHandling `json:"missing_value_handling"`
}

// MissingValueHandling contains label encoders and missing value info
type MissingValueHandling struct {
	NumericalMissingValue   interface{}             `json:"numerical_missing_value"`
	CategoricalMissingValue string                  `json:"categorical_missing_value"`
	LabelEncoders           map[string]LabelEncoder `json:"label_encoders"`
}

// LabelEncoder contains the classes and dtype for categorical encoding
type LabelEncoder struct {
	Classes []string `json:"classes"`
	Dtype   string   `json:"dtype"`
}

// ValidationData represents a single validation sample
// Uses map[string]interface{} to handle any number of features dynamically
type ValidationData map[string]interface{}

// TrainingHistory represents training metrics
type TrainingHistory struct {
	TrainLosses       []float64              `json:"train_losses"`
	Config            map[string]interface{} `json:"config"`
	TotalTrainingTime float64                `json:"total_training_time"`
}
