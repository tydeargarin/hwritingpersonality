"""
Example usage of the personality trait mapping pipeline.
"""

from map_personality_traits import PersonalityTraitMapper
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Example 1: Map features to traits using rule-based approach
def example_rule_based_mapping():
    """Example of using rule-based mapping (no training data needed)."""
    # Load features
    features_df = pd.read_csv('handwriting_features.csv')
    
    # Initialize mapper
    mapper = PersonalityTraitMapper()
    
    # Predict traits using rule-based mapping
    predictions_df = mapper.predict_traits(features_df, use_classifier=False)
    
    # Add image names
    if 'image_name' in features_df.columns:
        predictions_df.insert(0, 'image_name', features_df['image_name'].values)
    
    # Save results
    predictions_df.to_csv('personality_traits_rule_based.csv', index=False)
    print("Rule-based predictions saved to 'personality_traits_rule_based.csv'")
    
    # Display first few predictions
    print("\nFirst 5 predictions:")
    print(predictions_df.head())
    
    return predictions_df


# Example 2: Train classifiers and predict (requires labels)
def example_classifier_training():
    """Example of training classifiers (requires labeled data)."""
    # Load features and labels
    features_df = pd.read_csv('handwriting_features.csv')
    
    # Note: You need to create a labels CSV file with personality trait scores
    # Example labels.csv structure:
    # image_name,openness,conscientiousness,extraversion,agreeableness,neuroticism
    # 0052-1.tif,0.7,0.8,0.6,0.5,0.3
    # ...
    
    try:
        labels_df = pd.read_csv('labels.csv')
        
        # Initialize mapper
        mapper = PersonalityTraitMapper()
        
        # Train classifiers
        print("Training classifiers...")
        mapper.train_all_traits(features_df, labels_df, classifier_type='svm')
        
        # Save models
        mapper.save_models('personality_models.pkl')
        
        # Predict using trained classifiers
        predictions_df = mapper.predict_traits(features_df, use_classifier=True)
        
        # Add image names
        if 'image_name' in features_df.columns:
            predictions_df.insert(0, 'image_name', features_df['image_name'].values)
        
        # Save results
        predictions_df.to_csv('personality_traits_classifier.csv', index=False)
        print("Classifier predictions saved to 'personality_traits_classifier.csv'")
        
        return predictions_df
        
    except FileNotFoundError:
        print("labels.csv not found. Skipping classifier training example.")
        print("To use this example, create a labels.csv file with personality trait scores.")
        return None


# Example 3: Visualize trait distributions
def example_visualize_traits():
    """Example of visualizing personality trait distributions."""
    # Load predictions
    try:
        predictions_df = pd.read_csv('personality_traits_rule_based.csv')
    except FileNotFoundError:
        print("Running rule-based mapping first...")
        predictions_df = example_rule_based_mapping()
    
    # Get trait columns
    trait_cols = ['openness', 'conscientiousness', 'extraversion', 'agreeableness', 'neuroticism']
    trait_cols = [col for col in trait_cols if col in predictions_df.columns]
    
    # Create subplots
    n_traits = len(trait_cols)
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    for idx, trait in enumerate(trait_cols):
        if idx < len(axes):
            axes[idx].hist(predictions_df[trait], bins=30, edgecolor='black', alpha=0.7)
            axes[idx].set_title(f'{trait.capitalize()} Distribution')
            axes[idx].set_xlabel('Trait Score')
            axes[idx].set_ylabel('Frequency')
            axes[idx].axvline(predictions_df[trait].mean(), color='red', 
                            linestyle='--', label=f'Mean: {predictions_df[trait].mean():.3f}')
            axes[idx].legend()
    
    # Hide unused subplots
    for idx in range(len(trait_cols), len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    plt.savefig('personality_traits_distribution.png', dpi=150, bbox_inches='tight')
    print("Trait distributions saved to 'personality_traits_distribution.png'")


# Example 4: Compare rule-based vs classifier predictions
def example_compare_methods():
    """Example of comparing rule-based and classifier predictions."""
    features_df = pd.read_csv('handwriting_features.csv')
    mapper = PersonalityTraitMapper()
    
    # Rule-based predictions
    rule_predictions = mapper.predict_traits(features_df, use_classifier=False)
    
    # Classifier predictions (if models exist)
    try:
        mapper.load_models('personality_models.pkl')
        classifier_predictions = mapper.predict_traits(features_df, use_classifier=True)
        
        # Compare
        print("\nComparison of Prediction Methods:")
        print("=" * 60)
        for trait in mapper.trait_names:
            if trait in rule_predictions.columns and trait in classifier_predictions.columns:
                rule_mean = rule_predictions[trait].mean()
                classifier_mean = classifier_predictions[trait].mean()
                correlation = np.corrcoef(rule_predictions[trait], classifier_predictions[trait])[0, 1]
                
                print(f"{trait:20s}:")
                print(f"  Rule-based mean:     {rule_mean:.3f}")
                print(f"  Classifier mean:     {classifier_mean:.3f}")
                print(f"  Correlation:         {correlation:.3f}")
                print()
    except FileNotFoundError:
        print("No trained models found. Train models first to compare methods.")


# Example 5: Analyze individual handwriting sample
def example_analyze_sample():
    """Example of analyzing a single handwriting sample."""
    features_df = pd.read_csv('handwriting_features.csv')
    mapper = PersonalityTraitMapper()
    
    # Analyze first sample
    sample = features_df.iloc[0]
    traits = mapper.map_features_to_traits_rules(sample)
    
    print(f"\nPersonality Analysis for: {sample.get('image_name', 'Sample')}")
    print("=" * 60)
    print("\nBig Five Personality Traits:")
    for trait, score in traits.items():
        # Convert score to percentage
        percentage = score * 100
        bar_length = int(percentage / 2)  # Scale for display
        bar = '█' * bar_length + '░' * (50 - bar_length)
        print(f"{trait:20s}: {bar} {percentage:5.1f}%")
    
    print("\nTrait Interpretations:")
    print("-" * 60)
    interpretations = {
        'openness': 'High: Creative, curious, open to new experiences\nLow: Practical, prefers routine',
        'conscientiousness': 'High: Organized, disciplined, reliable\nLow: Spontaneous, flexible',
        'extraversion': 'High: Outgoing, energetic, social\nLow: Reserved, quiet, introverted',
        'agreeableness': 'High: Cooperative, trusting, empathetic\nLow: Competitive, skeptical',
        'neuroticism': 'High: Anxious, emotional, sensitive\nLow: Calm, stable, resilient'
    }
    
    for trait, interpretation in interpretations.items():
        if trait in traits:
            level = "High" if traits[trait] > 0.6 else "Low" if traits[trait] < 0.4 else "Moderate"
            print(f"\n{trait.capitalize()} ({level}):")
            print(f"  {interpretation}")


if __name__ == '__main__':
    print("Personality Trait Mapping Examples")
    print("=" * 60)
    
    # Uncomment the example you want to run:
    
    # Example 1: Rule-based mapping (no training data needed)
    example_rule_based_mapping()
    
    # Example 2: Train classifiers (requires labels.csv)
    # example_classifier_training()
    
    # Example 3: Visualize trait distributions
    # example_visualize_traits()
    
    # Example 4: Compare methods
    # example_compare_methods()
    
    # Example 5: Analyze individual sample
    # example_analyze_sample()

