"""
Example usage of the personality trait classification pipeline.
"""

from classify_personality_traits import PersonalityTraitClassifier
import pandas as pd
import numpy as np

# Example 1: Train and evaluate classifiers
def example_train_and_evaluate():
    """Example of training and evaluating all classifiers."""
    # Load features and labels
    features_df = pd.read_csv('handwriting_features.csv')
    
    # Note: You need a labels CSV file with personality trait scores/categories
    # Example labels.csv structure:
    # image_name,openness,conscientiousness,extraversion,agreeableness,neuroticism
    # 0052-1.tif,0.7,0.8,0.6,0.5,0.3
    # ...
    
    try:
        labels_df = pd.read_csv('labels.csv')
        
        # Initialize classifier
        classifier = PersonalityTraitClassifier()
        
        # Train and evaluate all classifiers
        print("Training and evaluating classifiers...")
        results = classifier.train_and_evaluate_all(
            features_df, 
            labels_df, 
            test_size=0.2,
            random_state=42
        )
        
        # Save models
        classifier.save_models('personality_classifiers.pkl')
        
        # Generate report
        classifier.generate_report(results, 'classification_report.txt')
        
        # Generate confusion matrices
        classifier.plot_confusion_matrices(results, 'classification_results')
        
        return results
        
    except FileNotFoundError:
        print("labels.csv not found. Skipping classification example.")
        print("To use this example, create a labels.csv file with personality trait scores.")
        return None


# Example 2: Use trained models for prediction
def example_predict_with_trained_models():
    """Example of using trained models for prediction."""
    # Load features
    features_df = pd.read_csv('handwriting_features.csv')
    
    # Initialize classifier
    classifier = PersonalityTraitClassifier()
    
    # Load trained models
    try:
        classifier.load_models('personality_classifiers.pkl')
        
        # Predict using different classifiers
        print("\nPredicting with SVM...")
        predictions_svm = classifier.predict_traits(features_df, classifier_type='svm')
        
        print("Predicting with KNN...")
        predictions_knn = classifier.predict_traits(features_df, classifier_type='knn')
        
        print("Predicting with Decision Tree...")
        predictions_dt = classifier.predict_traits(features_df, classifier_type='decision_tree')
        
        # Save predictions
        predictions_svm.to_csv('predictions_svm.csv', index=False)
        predictions_knn.to_csv('predictions_knn.csv', index=False)
        predictions_dt.to_csv('predictions_dt.csv', index=False)
        
        print("\nPredictions saved!")
        return predictions_svm, predictions_knn, predictions_dt
        
    except FileNotFoundError:
        print("No trained models found. Train models first.")
        return None


# Example 3: Compare classifier performance
def example_compare_classifiers():
    """Example of comparing different classifier performances."""
    features_df = pd.read_csv('handwriting_features.csv')
    
    try:
        labels_df = pd.read_csv('labels.csv')
        
        classifier = PersonalityTraitClassifier()
        
        # Train and evaluate
        results = classifier.train_and_evaluate_all(features_df, labels_df)
        
        # Compare results
        print("\n" + "="*70)
        print("CLASSIFIER COMPARISON")
        print("="*70)
        
        for trait in classifier.trait_names:
            if trait in results:
                print(f"\n{trait.upper()}:")
                print("-"*70)
                for clf_type in classifier.classifier_types:
                    if clf_type in results[trait]:
                        metrics = results[trait][clf_type]
                        print(f"  {clf_type.upper():15s}: "
                              f"Accuracy = {metrics['test_accuracy']*100:.2f}%, "
                              f"CV = {metrics['cv_mean']*100:.2f}% ± {metrics['cv_std']*100:.2f}%")
        
        return results
        
    except FileNotFoundError:
        print("labels.csv not found.")
        return None


if __name__ == '__main__':
    print("Personality Trait Classification Examples")
    print("=" * 70)
    
    # Uncomment the example you want to run:
    
    # Example 1: Train and evaluate classifiers
    # example_train_and_evaluate()
    
    # Example 2: Use trained models for prediction
    # example_predict_with_trained_models()
    
    # Example 3: Compare classifier performance
    # example_compare_classifiers()
    
    print("\nNote: These examples require a labels.csv file with personality trait scores.")
    print("The labels should be continuous values (0-1) or categorical (Low/Medium/High).")

