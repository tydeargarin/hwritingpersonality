"""
Personality Trait Mapping from Handwriting Features
Based on: "A Framework for Determining the Big Five Personality Traits 
Using Machine Learning Classification through Graphology"

Maps seven handwriting features to Big Five personality traits:
1. Openness to Experience
2. Conscientiousness
3. Extraversion
4. Agreeableness
5. Neuroticism
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
import pickle
from pathlib import Path


class PersonalityTraitMapper:
    """
    Maps handwriting features to Big Five personality traits using graphology rules
    and machine learning classification.
    """
    
    def __init__(self):
        """Initialize the personality trait mapper."""
        self.scaler = StandardScaler()
        self.models = {}
        self.feature_names = [
            'baseline_angle', 'baseline_stability', 'top_margin',
            'avg_line_spacing', 'line_spacing_variance',
            'avg_word_spacing', 'word_spacing_variance',
            'avg_letter_height', 'avg_letter_width', 'letter_size_variance',
            'avg_slant_angle', 'slant_consistency',
            'avg_pressure', 'pressure_variance'
        ]
        self.trait_names = [
            'openness',
            'conscientiousness',
            'extraversion',
            'agreeableness',
            'neuroticism'
        ]
    
    def determine_decision_rules(self, features):
        """
        Step 1: Determine decision rules for each class based on handwriting analysis features.
        Based on research paper methodology - establishes rules for feature interpretation.
        
        Args:
            features: Dictionary containing handwriting features
            
        Returns:
            Dictionary with decision rule scores for each feature category
        """
        if isinstance(features, pd.Series):
            features = features.to_dict()
        elif isinstance(features, pd.DataFrame):
            features = features.iloc[0].to_dict()
        
        # Extract feature values
        baseline_angle = features.get('baseline_angle', 0)
        baseline_stability = features.get('baseline_stability', 0)
        top_margin = features.get('top_margin', 0)
        avg_line_spacing = features.get('avg_line_spacing', 0)
        line_spacing_variance = features.get('line_spacing_variance', 0)
        avg_word_spacing = features.get('avg_word_spacing', 0)
        word_spacing_variance = features.get('word_spacing_variance', 0)
        avg_letter_height = features.get('avg_letter_height', 0)
        avg_letter_width = features.get('avg_letter_width', 0)
        letter_size_variance = features.get('letter_size_variance', 0)
        avg_slant_angle = features.get('avg_slant_angle', 0)
        slant_consistency = features.get('slant_consistency', 0)
        avg_pressure = features.get('avg_pressure', 0)
        pressure_variance = features.get('pressure_variance', 0)
        
        # Decision rules based on graphology principles (as per research paper)
        rules = {
            'baseline_stability': 'stable' if baseline_stability < 10 else 'unstable',
            'baseline_direction': 'ascending' if baseline_angle > 5 else 'descending' if baseline_angle < -5 else 'straight',
            'letter_size': 'large' if avg_letter_height > 0.03 else 'small' if avg_letter_height < 0.015 else 'medium',
            'letter_consistency': 'consistent' if letter_size_variance < 1e-7 else 'variable',
            'line_spacing': 'wide' if avg_line_spacing > 0.08 else 'narrow' if avg_line_spacing < 0.05 else 'moderate',
            'line_consistency': 'consistent' if line_spacing_variance < 0.0005 else 'variable',
            'word_spacing': 'wide' if avg_word_spacing > 0.05 else 'narrow' if avg_word_spacing < 0.02 else 'moderate',
            'word_consistency': 'consistent' if word_spacing_variance < 0.0005 else 'variable',
            'slant_direction': 'right' if avg_slant_angle > 10 else 'left' if avg_slant_angle < -10 else 'vertical',
            'slant_consistency': 'consistent' if slant_consistency < 10 else 'variable',
            'pressure_level': 'high' if avg_pressure > 0.5 else 'low' if avg_pressure < 0.3 else 'moderate',
            'pressure_consistency': 'consistent' if pressure_variance < 0.005 else 'variable',
            'top_margin_size': 'large' if top_margin > 0.1 else 'small' if top_margin < 0.05 else 'moderate'
        }
        
        return rules
    
    def map_features_to_traits_rules(self, features):
        """
        Step 2: Map features for psychological identification by applying the Big Five 
        personality psychology method.
        Based on research paper: maps handwriting features to Big Five traits using 
        graphology principles and decision rules.
        
        Args:
            features: Dictionary or DataFrame row containing handwriting features
            
        Returns:
            Dictionary with trait scores (0-1 scale, higher = more of that trait)
        """
        if isinstance(features, pd.Series):
            features = features.to_dict()
        elif isinstance(features, pd.DataFrame):
            features = features.iloc[0].to_dict()
        
        # Get decision rules
        rules = self.determine_decision_rules(features)
        
        # Extract feature values
        baseline_angle = features.get('baseline_angle', 0)
        baseline_stability = features.get('baseline_stability', 0)
        top_margin = features.get('top_margin', 0)
        avg_line_spacing = features.get('avg_line_spacing', 0)
        line_spacing_variance = features.get('line_spacing_variance', 0)
        avg_word_spacing = features.get('avg_word_spacing', 0)
        word_spacing_variance = features.get('word_spacing_variance', 0)
        avg_letter_height = features.get('avg_letter_height', 0)
        avg_letter_width = features.get('avg_letter_width', 0)
        letter_size_variance = features.get('letter_size_variance', 0)
        avg_slant_angle = features.get('avg_slant_angle', 0)
        slant_consistency = features.get('slant_consistency', 0)
        avg_pressure = features.get('avg_pressure', 0)
        pressure_variance = features.get('pressure_variance', 0)
        
        # Normalize features to 0-1 range for scoring
        def normalize(value, min_val, max_val):
            if max_val == min_val:
                return 0.5
            return max(0, min(1, (value - min_val) / (max_val - min_val)))
        
        # Calculate feature scores based on decision rules and graphology principles
        # Based on research paper and graphology literature (Amend & Ruiz, 1980)
        
        # 1. OPENNESS TO EXPERIENCE
        # Graphology indicators: Large letters, wide spacing, right slant, variable baseline, creative variations
        letter_size_score = normalize(avg_letter_height, 0.01, 0.05)
        letter_width_score = normalize(avg_letter_width, 0.005, 0.02)
        word_spacing_score = normalize(avg_word_spacing, 0, 0.1)
        right_slant_score = normalize(avg_slant_angle, -45, 45) if avg_slant_angle > 0 else 0.1
        baseline_variability = normalize(baseline_stability, 0, 50)  # More variable = more open
        letter_variation = normalize(letter_size_variance, 0, 1e-6)  # More variation = more creative
        
        openness = (
            0.18 * letter_size_score +
            0.18 * letter_width_score +
            0.16 * word_spacing_score +
            0.16 * right_slant_score +
            0.16 * baseline_variability +
            0.16 * letter_variation
        )
        
        # 2. CONSCIENTIOUSNESS
        # Graphology indicators: Stable baseline, consistent spacing, regular letter size, organized layout
        baseline_stability_score = 1.0 - normalize(baseline_stability, 0, 50)  # Stable baseline
        line_spacing_consistency = 1.0 - normalize(line_spacing_variance, 0, 0.01)  # Consistent line spacing
        word_spacing_consistency = 1.0 - normalize(word_spacing_variance, 0, 0.01)  # Consistent word spacing
        letter_size_consistency = 1.0 - normalize(letter_size_variance, 0, 1e-6)  # Consistent letter size
        pressure_score = normalize(avg_pressure, 0, 1)  # Moderate to high pressure
        slant_consistency_score = 1.0 - normalize(slant_consistency, 0, 20)  # Consistent slant
        
        conscientiousness = (
            0.22 * baseline_stability_score +
            0.20 * line_spacing_consistency +
            0.18 * word_spacing_consistency +
            0.18 * letter_size_consistency +
            0.11 * pressure_score +
            0.11 * slant_consistency_score
        )
        
        # 3. EXTRAVERSION
        # Graphology indicators: Right slant, large letters, wide spacing, high pressure, expansive writing
        slant_score = normalize(avg_slant_angle, -45, 45) if avg_slant_angle > 0 else 0.1
        letter_height_score = normalize(avg_letter_height, 0.01, 0.05)
        letter_width_score_ext = normalize(avg_letter_width, 0.005, 0.02)
        word_spacing_score_ext = normalize(avg_word_spacing, 0, 0.1)
        pressure_score_ext = normalize(avg_pressure, 0, 1)
        line_spacing_score = normalize(avg_line_spacing, 0, 0.15)
        
        extraversion = (
            0.22 * slant_score +
            0.20 * letter_height_score +
            0.18 * letter_width_score_ext +
            0.15 * word_spacing_score_ext +
            0.13 * pressure_score_ext +
            0.12 * line_spacing_score
        )
        
        # 4. AGREEABLENESS
        # Graphology indicators: Moderate to small letters, moderate spacing, left/vertical slant, stable baseline
        small_letter_score = 1.0 - normalize(avg_letter_height, 0.01, 0.05)  # Smaller letters
        small_width_score = 1.0 - normalize(avg_letter_width, 0.005, 0.02)
        left_slant_score = normalize(avg_slant_angle, -45, 0) if avg_slant_angle < 0 else (0.5 if abs(avg_slant_angle) < 5 else 0.1)  # Left or vertical
        stable_baseline_score = 1.0 - normalize(baseline_stability, 0, 50)  # Stable baseline
        moderate_line_spacing = normalize(avg_line_spacing, 0, 0.15)  # Moderate spacing
        moderate_word_spacing = normalize(avg_word_spacing, 0, 0.1)
        moderate_pressure = 1.0 - abs(normalize(avg_pressure, 0, 1) - 0.5) * 2  # Moderate pressure
        
        agreeableness = (
            0.18 * small_letter_score +
            0.15 * small_width_score +
            0.15 * left_slant_score +
            0.15 * stable_baseline_score +
            0.13 * moderate_line_spacing +
            0.12 * moderate_word_spacing +
            0.12 * moderate_pressure
        )
        
        # 5. NEUROTICISM
        # Graphology indicators: Unstable baseline, variable spacing, inconsistent slant, variable pressure
        unstable_baseline = normalize(baseline_stability, 0, 50)  # Unstable baseline
        variable_line_spacing = normalize(line_spacing_variance, 0, 0.01)  # Variable spacing
        variable_word_spacing = normalize(word_spacing_variance, 0, 0.01)
        inconsistent_slant = normalize(slant_consistency, 0, 20)  # Inconsistent slant
        variable_pressure = normalize(pressure_variance, 0, 0.01)  # Variable pressure
        variable_letter_size = normalize(letter_size_variance, 0, 1e-6)  # Variable letter size
        
        neuroticism = (
            0.22 * unstable_baseline +
            0.18 * variable_line_spacing +
            0.18 * variable_word_spacing +
            0.15 * inconsistent_slant +
            0.14 * variable_pressure +
            0.13 * variable_letter_size
        )
        
        # Normalize all scores to 0-1 range
        scores = {
            'openness': max(0, min(1, openness)),
            'conscientiousness': max(0, min(1, conscientiousness)),
            'extraversion': max(0, min(1, extraversion)),
            'agreeableness': max(0, min(1, agreeableness)),
            'neuroticism': max(0, min(1, neuroticism))
        }
        
        return scores
    
    def train_classifier(self, features_df, labels_df, trait_name, classifier_type='svm'):
        """
        Train a classifier for a specific personality trait.
        
        Args:
            features_df: DataFrame with handwriting features
            labels_df: DataFrame with personality trait labels
            trait_name: Name of the trait to predict ('openness', 'conscientiousness', etc.)
            classifier_type: Type of classifier ('svm', 'knn', 'decision_tree', 'random_forest')
            
        Returns:
            Trained classifier model
        """
        # Prepare features
        X = features_df[self.feature_names].values
        y = labels_df[trait_name].values
        
        # Handle missing values
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Select classifier (as per research paper: Decision Tree, SVM RBF, KNN)
        # Paper achieved >99% accuracy with these classifiers
        if classifier_type == 'svm':
            # SVM with RBF kernel as used in research paper
            clf = SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42)
        elif classifier_type == 'knn':
            # KNN classifier as used in research paper
            clf = KNeighborsClassifier(n_neighbors=5)
        elif classifier_type == 'decision_tree':
            # Decision Tree classifier as used in research paper
            clf = DecisionTreeClassifier(random_state=42, max_depth=10)
        elif classifier_type == 'random_forest':
            clf = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10)
        else:
            raise ValueError(f"Unknown classifier type: {classifier_type}")
        
        # Train classifier
        clf.fit(X_scaled, y)
        
        return clf
    
    def train_all_traits(self, features_df, labels_df, classifier_type='svm'):
        """
        Train classifiers for all Big Five personality traits.
        
        Args:
            features_df: DataFrame with handwriting features
            labels_df: DataFrame with personality trait labels
            classifier_type: Type of classifier to use
            
        Returns:
            Dictionary of trained models for each trait
        """
        models = {}
        
        for trait in self.trait_names:
            if trait in labels_df.columns:
                print(f"Training {classifier_type} classifier for {trait}...")
                model = self.train_classifier(features_df, labels_df, trait, classifier_type)
                models[trait] = model
                print(f"  {trait} classifier trained successfully")
            else:
                print(f"  Warning: {trait} not found in labels, skipping...")
        
        self.models = models
        return models
    
    def predict_traits(self, features_df, use_classifier=False):
        """
        Step 3: Classify the Big Five personality from handwriting images with a machine 
        learning approach based on the psychological identification mapping.
        Based on research paper methodology - three-step classification process.
        
        Args:
            features_df: DataFrame with handwriting features
            use_classifier: If True, use trained classifiers; if False, use rule-based mapping
            
        Returns:
            DataFrame with predicted trait scores
        """
        if use_classifier and self.models:
            # Step 3: Use trained machine learning classifiers (as per research paper)
            # Paper used Decision Tree, SVM (RBF kernel), and KNN with >99% accuracy
            predictions = {}
            
            # Prepare features
            X = features_df[self.feature_names].values
            X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
            X_scaled = self.scaler.transform(X)
            
            for trait in self.trait_names:
                if trait in self.models:
                    predictions[trait] = self.models[trait].predict(X_scaled)
                else:
                    predictions[trait] = [0.5] * len(features_df)
            
            return pd.DataFrame(predictions, index=features_df.index)
        else:
            # Use rule-based mapping (Steps 1 & 2: Decision rules + Psychological identification)
            predictions = []
            for idx, row in features_df.iterrows():
                # Step 1: Determine decision rules
                rules = self.determine_decision_rules(row)
                # Step 2: Map features for psychological identification
                trait_scores = self.map_features_to_traits_rules(row)
                predictions.append(trait_scores)
            
            return pd.DataFrame(predictions, index=features_df.index)
    
    def evaluate_classifier(self, features_df, labels_df, trait_name, classifier_type='svm', test_size=0.2):
        """
        Evaluate classifier performance for a specific trait.
        
        Args:
            features_df: DataFrame with handwriting features
            labels_df: DataFrame with personality trait labels
            trait_name: Name of the trait to evaluate
            classifier_type: Type of classifier
            test_size: Proportion of data to use for testing
            
        Returns:
            Dictionary with evaluation metrics
        """
        # Prepare data
        X = features_df[self.feature_names].values
        y = labels_df[trait_name].values
        
        # Handle missing values
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train classifier (as per research paper: Decision Tree, SVM RBF, KNN)
        if classifier_type == 'svm':
            clf = SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42)
        elif classifier_type == 'knn':
            clf = KNeighborsClassifier(n_neighbors=5)
        elif classifier_type == 'decision_tree':
            clf = DecisionTreeClassifier(random_state=42, max_depth=10)
        elif classifier_type == 'random_forest':
            clf = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10)
        
        clf.fit(X_train_scaled, y_train)
        
        # Predict
        y_pred = clf.predict(X_test_scaled)
        
        # Evaluate
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True)
        
        # Cross-validation
        cv_scores = cross_val_score(clf, X_train_scaled, y_train, cv=5)
        
        return {
            'trait': trait_name,
            'classifier': classifier_type,
            'accuracy': accuracy,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'classification_report': report
        }
    
    def save_models(self, filepath):
        """Save trained models to file."""
        model_data = {
            'models': self.models,
            'scaler': self.scaler,
            'feature_names': self.feature_names
        }
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        print(f"Models saved to {filepath}")
    
    def load_models(self, filepath):
        """Load trained models from file."""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        self.models = model_data['models']
        self.scaler = model_data['scaler']
        self.feature_names = model_data['feature_names']
        print(f"Models loaded from {filepath}")


def main():
    """
    Main function to demonstrate personality trait mapping.
    """
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Map handwriting features to Big Five personality traits'
    )
    parser.add_argument(
        '--features_csv',
        type=str,
        default='handwriting_features.csv',
        help='CSV file with extracted features (default: handwriting_features.csv)'
    )
    parser.add_argument(
        '--output_csv',
        type=str,
        default='personality_traits.csv',
        help='Output CSV file with predicted traits (default: personality_traits.csv)'
    )
    parser.add_argument(
        '--labels_csv',
        type=str,
        default=None,
        help='Optional CSV file with ground truth labels for training'
    )
    parser.add_argument(
        '--use_classifier',
        action='store_true',
        help='Use trained classifier instead of rule-based mapping'
    )
    parser.add_argument(
        '--classifier_type',
        type=str,
        choices=['svm', 'knn', 'decision_tree', 'random_forest'],
        default='svm',
        help='Type of classifier to use (default: svm)'
    )
    parser.add_argument(
        '--train',
        action='store_true',
        help='Train classifiers (requires --labels_csv)'
    )
    parser.add_argument(
        '--model_file',
        type=str,
        default='personality_models.pkl',
        help='File to save/load trained models (default: personality_models.pkl)'
    )
    
    args = parser.parse_args()
    
    # Load features
    print(f"Loading features from {args.features_csv}...")
    features_df = pd.read_csv(args.features_csv)
    print(f"Loaded {len(features_df)} samples with {len(features_df.columns)} features")
    
    # Initialize mapper
    mapper = PersonalityTraitMapper()
    
    # Train classifiers if requested
    if args.train and args.labels_csv:
        print(f"\nLoading labels from {args.labels_csv}...")
        labels_df = pd.read_csv(args.labels_csv)
        print(f"Training {args.classifier_type} classifiers for all traits...")
        mapper.train_all_traits(features_df, labels_df, args.classifier_type)
        mapper.save_models(args.model_file)
    elif args.use_classifier:
        # Load existing models
        if Path(args.model_file).exists():
            mapper.load_models(args.model_file)
        else:
            print(f"Warning: Model file {args.model_file} not found. Using rule-based mapping instead.")
            args.use_classifier = False
    
    # Predict traits
    print(f"\nPredicting personality traits...")
    predictions_df = mapper.predict_traits(features_df, use_classifier=args.use_classifier)
    
    # Combine with image names
    if 'image_name' in features_df.columns:
        predictions_df.insert(0, 'image_name', features_df['image_name'].values)
    
    # Save predictions
    predictions_df.to_csv(args.output_csv, index=False)
    print(f"\nPredictions saved to {args.output_csv}")
    
    # Display summary statistics
    print("\nTrait Score Summary:")
    print("=" * 60)
    for trait in mapper.trait_names:
        if trait in predictions_df.columns:
            mean_score = predictions_df[trait].mean()
            std_score = predictions_df[trait].std()
            print(f"{trait:20s}: Mean = {mean_score:.3f}, Std = {std_score:.3f}")


if __name__ == '__main__':
    main()

