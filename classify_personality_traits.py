"""
Personality Traits Classification using Machine Learning
Based on: "A Framework for Determining the Big Five Personality Traits 
Using Machine Learning Classification through Graphology"

Implements classification using:
1. Support Vector Machine (SVM) with RBF kernel
2. K-Nearest Neighbors (KNN)
3. Decision Tree

As per research paper methodology achieving >99% accuracy.
"""

import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import (
    classification_report, 
    accuracy_score, 
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score
)
from sklearn.preprocessing import StandardScaler, LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import argparse
import pickle
from tqdm import tqdm


class PersonalityTraitClassifier:
    """
    Classifies Big Five personality traits using SVM, KNN, and Decision Tree
    as per the research paper methodology.
    """
    
    def __init__(self):
        """Initialize the classifier."""
        self.scaler = StandardScaler()
        self.models = {}
        self.label_encoders = {}
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
        self.classifier_types = ['svm', 'knn', 'decision_tree']
    
    def prepare_data(self, features_df, labels_df, trait_name):
        """
        Prepare data for classification.
        
        Args:
            features_df: DataFrame with handwriting features
            labels_df: DataFrame with personality trait labels
            trait_name: Name of the trait to classify
            
        Returns:
            X: Feature matrix
            y: Target labels
        """
        # Extract features
        X = features_df[self.feature_names].values
        
        # Handle missing values
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Extract labels
        if trait_name in labels_df.columns:
            y = labels_df[trait_name].values
            
            # Handle missing labels
            y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)
            
            # If labels are continuous, convert to categories (high/medium/low)
            if y.dtype in [np.float64, np.float32]:
                # Discretize into 3 categories: Low (0-0.33), Medium (0.33-0.67), High (0.67-1.0)
                y_categorical = np.digitize(y, bins=[0.33, 0.67])
                y = y_categorical
        else:
            raise ValueError(f"Trait {trait_name} not found in labels DataFrame")
        
        # Ensure multiple classes exist
        unique_classes = np.unique(y)
        if len(unique_classes) < 2:
            # Return None to indicate insufficient classes
            return None, None
        
        return X, y
    
    def train_classifier(self, X, y, classifier_type='svm'):
        """
        Train a classifier for personality trait prediction.
        As per research paper: SVM (RBF kernel), KNN, Decision Tree.
        
        Args:
            X: Feature matrix
            y: Target labels
            classifier_type: Type of classifier ('svm', 'knn', 'decision_tree')
            
        Returns:
            Trained classifier model
        """
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Select classifier based on research paper
        if classifier_type == 'svm':
            # SVM with RBF kernel as used in research paper
            clf = SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42)
        elif classifier_type == 'knn':
            # KNN classifier as used in research paper
            clf = KNeighborsClassifier(n_neighbors=5)
        elif classifier_type == 'decision_tree':
            # Decision Tree classifier as used in research paper
            clf = DecisionTreeClassifier(random_state=42, max_depth=10)
        else:
            raise ValueError(f"Unknown classifier type: {classifier_type}")
        
        # Train classifier
        clf.fit(X_scaled, y)
        
        return clf
    
    def evaluate_classifier(self, clf, X_train, X_test, y_train, y_test, classifier_type):
        """
        Evaluate classifier performance.
        
        Args:
            clf: Trained classifier
            X_train: Training features
            X_test: Test features
            y_train: Training labels
            y_test: Test labels
            classifier_type: Type of classifier
            
        Returns:
            Dictionary with evaluation metrics
        """
        # Scale features
        X_train_scaled = self.scaler.transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Predictions
        y_train_pred = clf.predict(X_train_scaled)
        y_test_pred = clf.predict(X_test_scaled)
        
        # Calculate metrics
        train_accuracy = accuracy_score(y_train, y_train_pred)
        test_accuracy = accuracy_score(y_test, y_test_pred)
        precision = precision_score(y_test, y_test_pred, average='weighted', zero_division=0)
        recall = recall_score(y_test, y_test_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_test, y_test_pred, average='weighted', zero_division=0)
        
        # Cross-validation
        cv_scores = cross_val_score(clf, X_train_scaled, y_train, cv=5, scoring='accuracy')
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_test_pred)
        
        # Classification report
        report = classification_report(y_test, y_test_pred, output_dict=True, zero_division=0)
        
        return {
            'classifier': classifier_type,
            'train_accuracy': train_accuracy,
            'test_accuracy': test_accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'confusion_matrix': cm,
            'classification_report': report
        }
    
    def train_and_evaluate_all(self, features_df, labels_df, test_size=0.2, random_state=42):
        """
        Train and evaluate all classifiers (SVM, KNN, Decision Tree) for all traits.
        As per research paper methodology.
        
        Args:
            features_df: DataFrame with handwriting features
            labels_df: DataFrame with personality trait labels
            test_size: Proportion of data for testing
            random_state: Random seed for reproducibility
            
        Returns:
            Dictionary with results for all traits and classifiers
        """
        results = {}
        
        for trait in self.trait_names:
            if trait not in labels_df.columns:
                print(f"Warning: {trait} not found in labels, skipping...")
                continue
            
            print(f"\n{'='*70}")
            print(f"Classifying {trait.upper()}")
            print(f"{'='*70}")
            
            # Prepare data
            X, y = self.prepare_data(features_df, labels_df, trait)
            if X is None or y is None:
                print(f"  Skipping {trait} due to insufficient class diversity in labels.")
                continue
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state, stratify=y
            )
            
            trait_results = {}
            
            # Train and evaluate each classifier
            for classifier_type in self.classifier_types:
                print(f"\nTraining {classifier_type.upper()} classifier...")
                
                # Train classifier
                clf = self.train_classifier(X_train, y_train, classifier_type)
                
                # Evaluate
                metrics = self.evaluate_classifier(
                    clf, X_train, X_test, y_train, y_test, classifier_type
                )
                
                # Store model
                if trait not in self.models:
                    self.models[trait] = {}
                self.models[trait][classifier_type] = clf
                
                # Store results
                trait_results[classifier_type] = metrics
                
                # Print results
                print(f"  Test Accuracy: {metrics['test_accuracy']:.4f} ({metrics['test_accuracy']*100:.2f}%)")
                print(f"  Cross-Validation: {metrics['cv_mean']:.4f} ± {metrics['cv_std']:.4f}")
                print(f"  Precision: {metrics['precision']:.4f}")
                print(f"  Recall: {metrics['recall']:.4f}")
                print(f"  F1-Score: {metrics['f1_score']:.4f}")
            
            results[trait] = trait_results
        
        return results
    
    def predict_traits(self, features_df, classifier_type='svm'):
        """
        Predict personality traits using trained classifiers.
        
        Args:
            features_df: DataFrame with handwriting features
            classifier_type: Type of classifier to use ('svm', 'knn', 'decision_tree')
            
        Returns:
            DataFrame with predicted trait classes
        """
        predictions = {}
        
        # Prepare features
        X = features_df[self.feature_names].values
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        X_scaled = self.scaler.transform(X)
        
        for trait in self.trait_names:
            if trait in self.models and classifier_type in self.models[trait]:
                clf = self.models[trait][classifier_type]
                predictions[trait] = clf.predict(X_scaled)
            else:
                # Default prediction if model not available
                predictions[trait] = [1] * len(features_df)  # Medium category
        
        return pd.DataFrame(predictions, index=features_df.index)
    
    def plot_confusion_matrices(self, results, output_dir='classification_results'):
        """
        Plot confusion matrices for all classifiers and traits.
        
        Args:
            results: Dictionary with classification results
            output_dir: Directory to save plots
        """
        Path(output_dir).mkdir(exist_ok=True)
        
        for trait, trait_results in results.items():
            fig, axes = plt.subplots(1, 3, figsize=(18, 5))
            fig.suptitle(f'Confusion Matrices for {trait.upper()}', fontsize=16)
            
            for idx, classifier_type in enumerate(self.classifier_types):
                if classifier_type in trait_results:
                    cm = trait_results[classifier_type]['confusion_matrix']
                    accuracy = trait_results[classifier_type]['test_accuracy']
                    
                    sns.heatmap(
                        cm, 
                        annot=True, 
                        fmt='d', 
                        cmap='Blues',
                        ax=axes[idx],
                        cbar_kws={'label': 'Count'}
                    )
                    axes[idx].set_title(f'{classifier_type.upper()}\nAccuracy: {accuracy*100:.2f}%')
                    axes[idx].set_xlabel('Predicted')
                    axes[idx].set_ylabel('Actual')
            
            plt.tight_layout()
            plt.savefig(f'{output_dir}/confusion_matrix_{trait}.png', dpi=150, bbox_inches='tight')
            plt.close()
        
        print(f"\nConfusion matrices saved to {output_dir}/")
    
    def save_models(self, filepath):
        """Save trained models to file."""
        model_data = {
            'models': self.models,
            'scaler': self.scaler,
            'feature_names': self.feature_names,
            'label_encoders': self.label_encoders
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
        self.label_encoders = model_data.get('label_encoders', {})
        print(f"Models loaded from {filepath}")
    
    def generate_report(self, results, output_file='classification_report.txt'):
        """
        Generate a comprehensive classification report.
        
        Args:
            results: Dictionary with classification results
            output_file: Path to save report
        """
        with open(output_file, 'w') as f:
            f.write("="*80 + "\n")
            f.write("PERSONALITY TRAITS CLASSIFICATION REPORT\n")
            f.write("Based on: A Framework for Determining the Big Five Personality Traits\n")
            f.write("Using Machine Learning Classification through Graphology\n")
            f.write("="*80 + "\n\n")
            
            # Summary table (per trait and classifier)
            f.write("SUMMARY OF RESULTS (Per Trait and Classifier)\n")
            f.write("-"*80 + "\n")
            f.write(f"{'Trait':<20} {'Classifier':<15} {'Accuracy':<12} {'CV Mean':<12} {'F1-Score':<12}\n")
            f.write("-"*80 + "\n")
            
            # Accumulators for average performance per classifier (like tables in the paper)
            classifier_summaries = {clf: {'accuracy': [], 'cv_mean': [], 'f1': []}
                                    for clf in self.classifier_types}
            
            for trait, trait_results in results.items():
                for classifier_type in self.classifier_types:
                    if classifier_type in trait_results:
                        metrics = trait_results[classifier_type]
                        acc = metrics['test_accuracy'] * 100.0
                        cvm = metrics['cv_mean'] * 100.0
                        f1 = metrics['f1_score']
                        
                        # Write per-trait row
                        f.write(
                            f"{trait:<20} {classifier_type:<15} "
                            f"{acc:>10.2f}% "
                            f"{cvm:>10.2f}% "
                            f"{f1:>10.4f}\n"
                        )
                        
                        # Accumulate for classifier-level summary
                        classifier_summaries[classifier_type]['accuracy'].append(acc)
                        classifier_summaries[classifier_type]['cv_mean'].append(cvm)
                        classifier_summaries[classifier_type]['f1'].append(f1)
            
            # Average performance per classifier across all traits (as in the research paper comparison)
            f.write("\n\nAVERAGE PERFORMANCE BY CLASSIFIER (Across All Traits)\n")
            f.write("-"*80 + "\n")
            f.write(f"{'Classifier':<15} {'Avg Acc':<12} {'Std Acc':<12} {'Avg CV':<12} {'Avg F1':<12}\n")
            f.write("-"*80 + "\n")
            
            for classifier_type in self.classifier_types:
                acc_list = classifier_summaries[classifier_type]['accuracy']
                cv_list = classifier_summaries[classifier_type]['cv_mean']
                f1_list = classifier_summaries[classifier_type]['f1']
                
                if not acc_list:
                    continue
                
                avg_acc = np.mean(acc_list)
                std_acc = np.std(acc_list)
                avg_cv = np.mean(cv_list)
                avg_f1 = np.mean(f1_list)
                
                f.write(
                    f"{classifier_type:<15} "
                    f"{avg_acc:>10.2f}% "
                    f"{std_acc:>10.2f}% "
                    f"{avg_cv:>10.2f}% "
                    f"{avg_f1:>10.4f}\n"
                )
            
            f.write("\n" + "="*80 + "\n\n")
            
            # Detailed results for each trait
            for trait, trait_results in results.items():
                f.write(f"\n{'='*80}\n")
                f.write(f"DETAILED RESULTS FOR {trait.upper()}\n")
                f.write(f"{'='*80}\n\n")
                
                for classifier_type in self.classifier_types:
                    if classifier_type in trait_results:
                        metrics = trait_results[classifier_type]
                        f.write(f"{classifier_type.upper()} Classifier:\n")
                        f.write(f"  Test Accuracy: {metrics['test_accuracy']*100:.2f}%\n")
                        f.write(f"  Train Accuracy: {metrics['train_accuracy']*100:.2f}%\n")
                        f.write(f"  Cross-Validation: {metrics['cv_mean']*100:.2f}% ± {metrics['cv_std']*100:.2f}%\n")
                        f.write(f"  Precision: {metrics['precision']:.4f}\n")
                        f.write(f"  Recall: {metrics['recall']:.4f}\n")
                        f.write(f"  F1-Score: {metrics['f1_score']:.4f}\n")
                        f.write("\n")
        
        print(f"\nClassification report saved to {output_file}")


def main():
    """
    Main function to run personality trait classification.
    """
    parser = argparse.ArgumentParser(
        description='Classify Big Five personality traits using SVM, KNN, and Decision Tree'
    )
    parser.add_argument(
        '--features_csv',
        type=str,
        default='handwriting_features.csv',
        help='CSV file with extracted features (default: handwriting_features.csv)'
    )
    parser.add_argument(
        '--labels_csv',
        type=str,
        default=None,
        help='CSV file with personality trait labels (optional - if not provided, uses rule-based mapping)'
    )
    parser.add_argument(
        '--use_rule_based_labels',
        action='store_true',
        help='Use rule-based mapping to generate labels for training (requires map_personality_traits module)'
    )
    parser.add_argument(
        '--test_size',
        type=float,
        default=0.2,
        help='Proportion of data for testing (default: 0.2)'
    )
    parser.add_argument(
        '--random_state',
        type=int,
        default=42,
        help='Random seed for reproducibility (default: 42)'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='classification_results',
        help='Directory to save results (default: classification_results)'
    )
    parser.add_argument(
        '--model_file',
        type=str,
        default='personality_classifiers.pkl',
        help='File to save trained models (default: personality_classifiers.pkl)'
    )
    parser.add_argument(
        '--generate_plots',
        action='store_true',
        help='Generate confusion matrix plots'
    )
    
    args = parser.parse_args()
    
    # Load data
    print(f"Loading features from {args.features_csv}...")
    features_df = pd.read_csv(args.features_csv)
    feature_columns = features_df.columns.tolist()
    print(f"Loaded {len(features_df)} samples with {len(feature_columns)} features")
    if 'image_name' in features_df.columns:
        features_df = features_df.drop_duplicates(subset='image_name', keep='first')
        print(f"After removing duplicate images in features: {len(features_df)} samples")
    
    # Load or generate labels
    label_columns = None
    if args.labels_csv:
        print(f"\nLoading labels from {args.labels_csv}...")
        labels_df = pd.read_csv(args.labels_csv)
        print(f"Loaded {len(labels_df)} samples with {len(labels_df.columns)} columns")
        if 'image_name' in labels_df.columns:
            labels_df = labels_df.drop_duplicates(subset='image_name', keep='first')
            print(f"After removing duplicate images in labels: {len(labels_df)} samples")
        label_columns = labels_df.columns.tolist()
        
        # Ensure same number of samples
        min_samples = min(len(features_df), len(labels_df))
        features_df = features_df.iloc[:min_samples]
        labels_df = labels_df.iloc[:min_samples]
        
        # Merge on image_name if available
        if 'image_name' in features_df.columns and 'image_name' in labels_df.columns:
            merged_df = pd.merge(features_df, labels_df, on='image_name', how='inner')
            features_df = merged_df[feature_columns]
            labels_df = merged_df[label_columns]
            print(f"After merging: {len(features_df)} samples")
    else:
        # Default: Use rule-based mapping to generate labels
        # Check if personality_traits.csv exists first
        if Path('personality_traits.csv').exists():
            print("\nFound existing personality_traits.csv, using it as labels...")
            labels_df = pd.read_csv('personality_traits.csv')
            print(f"Loaded {len(labels_df)} samples from personality_traits.csv")
            if 'image_name' in labels_df.columns:
                labels_df = labels_df.drop_duplicates(subset='image_name', keep='first')
                print(f"After removing duplicate images in labels: {len(labels_df)} samples")
            label_columns = labels_df.columns.tolist()
            
            # Merge on image_name if available
            if 'image_name' in features_df.columns and 'image_name' in labels_df.columns:
                merged_df = pd.merge(features_df, labels_df, on='image_name', how='inner')
                features_df = merged_df[feature_columns]
                labels_df = merged_df[label_columns]
                print(f"After merging: {len(features_df)} samples")
        else:
            # Generate labels using rule-based mapping
            print("\nNo labels file provided. Generating labels using rule-based mapping...")
            from map_personality_traits import PersonalityTraitMapper
            
            mapper = PersonalityTraitMapper()
            labels_df = mapper.predict_traits(features_df, use_classifier=False)
            
            # Add image_name if available
            if 'image_name' in features_df.columns:
                labels_df.insert(0, 'image_name', features_df['image_name'].values)
            label_columns = labels_df.columns.tolist()
            
            print(f"Generated labels for {len(labels_df)} samples")
            print("Note: Using rule-based predictions as labels for training.")
    
    # Initialize classifier
    classifier = PersonalityTraitClassifier()
    
    # Train and evaluate all classifiers
    print("\n" + "="*70)
    print("TRAINING AND EVALUATING CLASSIFIERS")
    print("="*70)
    print("Using: SVM (RBF kernel), KNN, Decision Tree")
    print("As per research paper methodology")
    print("="*70)
    
    results = classifier.train_and_evaluate_all(
        features_df, 
        labels_df, 
        test_size=args.test_size,
        random_state=args.random_state
    )
    
    # Save models
    Path(args.output_dir).mkdir(exist_ok=True)
    classifier.save_models(f"{args.output_dir}/{args.model_file}")
    
    # Generate report
    classifier.generate_report(results, f"{args.output_dir}/classification_report.txt")
    
    # Generate plots if requested
    if args.generate_plots:
        classifier.plot_confusion_matrices(results, args.output_dir)
    
    # Print summary
    print("\n" + "="*70)
    print("CLASSIFICATION SUMMARY")
    print("="*70)
    print(f"{'Trait':<20} {'Classifier':<15} {'Accuracy':<12} {'CV Mean':<12}")
    print("-"*70)
    
    for trait, trait_results in results.items():
        for classifier_type in classifier.classifier_types:
            if classifier_type in trait_results:
                metrics = trait_results[classifier_type]
                print(
                    f"{trait:<20} {classifier_type:<15} "
                    f"{metrics['test_accuracy']*100:>10.2f}% "
                    f"{metrics['cv_mean']*100:>10.2f}%"
                )
    
    print("\n" + "="*70)
    print(f"Results saved to {args.output_dir}/")
    print("="*70)


if __name__ == '__main__':
    main()

