# Protein Family Classification

A machine learning project for protein family classification using enhanced physicochemical k-mer features and standard k-mer approaches.

## Overview

This project implements a comprehensive pipeline for downloading, processing, and classifying protein sequences from biological families using advanced feature engineering techniques. It compares multiple classification approaches including standard k-mer analysis and enhanced physicochemical grouping schemes.

## Features

### Data Processing
- **Download Protein Families**: Fetches curated protein sequences from UniProt REST API using Pfam identifiers
- **FASTA Processing**: Reads, validates, and manipulates FASTA sequence files
- **Dataset Balancing**: Randomly samples sequences to ensure balanced class representation
- **Format Conversion**: Converts processed sequences from FASTA to CSV format with metadata

### Classification Methods

1. **Standard 3-mer Random Forest**: Traditional k-mer feature extraction with Random Forest classifier
2. **Enhanced Physicochemical Grouping**: Groups amino acids by chemical properties (aliphatic, polar, charged, etc.)
3. **Hydropathy-based Approach**: Classification based on hydrophobic/hydrophilic properties
4. **Structural Approach**: Groups based on structural burial and surface orientation

### Analysis & Evaluation
- Trains classifiers on 80% training, tests on 20% test split
- Calculates accuracy, F1-score, and training/prediction times
- Generates feature importance analysis and biological interpretability mappings
- Produces comprehensive visualizations comparing method performance
- Saves confusion matrices and performance comparisons

## Main Functions

- `download_family(pfam_id, name, limit)` - Download sequences for a protein family
- `read_fasta(file)` - Parse FASTA files into (header, sequence) tuples
- `write_fasta(seqs, out)` - Write sequences to FASTA format
- `balance_fasta_files(target)` - Balance datasets to target sequence count
- `combine_balanced_to_final()` - Merge balanced files into unified dataset
- `fasta_to_csv(input_fasta, output_csv)` - Convert FASTA to CSV with metadata
- `run_enhanced_analysis()` - Execute full classification pipeline with all methods
- `create_enhanced_visualization(results_df, classifiers, X_test, y_test)` - Generate performance visualizations

## Usage

```python
# Download protein families
families = {
    "kinase": "PF00069",
    "gpcr": "PF00001",
    "protease": "PF00082",
    # ... more families
}
for name, pfam_id in families.items():
    download_family(pfam_id, name, limit=250)

# Balance and combine datasets
balance_fasta_files(target=250)
combine_balanced_to_final()
fasta_to_csv()

# Run classification analysis
results, classifiers = run_enhanced_analysis()

# Generate visualizations
create_enhanced_visualization(results, classifiers, X_test, y_test)
```

## Dependencies

- requests - HTTP library for API calls
- pandas - Data manipulation and analysis
- numpy - Numerical computing
- scikit-learn - Machine learning algorithms
- matplotlib - Visualization
- seaborn - Statistical data visualization

## Data Sources

Protein sequences sourced from:
- **UniProt**: https://www.uniprot.org/ (reviewed and unreviewed entries)
- **Pfam**: https://pfam.xfam.org/ (protein family identifiers)

## Output

- `final_dataset.fasta` - Combined balanced sequences with family labels
- `protein_family_dataset.csv` - CSV format with sequence_id, sequence, family, sequence_length
- `enhanced_physiochem_results.png` - Comprehensive performance comparison visualization
- Console output with detailed metrics and biological feature interpretations
