#Rishi
#SAthish
import requests
import glob
import random
import csv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import time
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import warnings
warnings.filterwarnings('ignore')


def download_family(pfam_id, name, limit=250):
    url = "https://rest.uniprot.org/uniprotkb/stream"

    # Correct query format
    query = f"xref:pfam-{pfam_id} AND reviewed:true"

    params = {
        "query": query,
        "format": "fasta",
        "size": limit
    }

    headers = {
        "User-Agent": "Mozilla/5.0"
    }

    r = requests.get(url, params=params, headers=headers, timeout=60)

    print(f"{name} → Status:", r.status_code)

    if r.status_code == 200 and len(r.text) > 0:

        with open(f"{name}.fasta", "w") as f:
            f.write(r.text)

        print(f"Downloaded {name} ({len(r.text)} chars)\n")

    else:
        print(f"Failed {name}\n")


def read_fasta(file):
    seqs = []
    with open(file) as f:
        header = None
        seq = []
        for line in f:
            line = line.strip()
            if line.startswith(">"):
                if header:
                    seqs.append((header, "".join(seq)))
                header = line
                seq = []
            else:
                seq.append(line)
        if header:
            seqs.append((header, "".join(seq)))
    return seqs


def write_fasta(seqs, out):
    with open(out, "w") as f:
        for h, s in seqs:
            f.write(h + "\n")
            f.write(s + "\n")


def balance_fasta_files(target=250):
    TARGET = target
    for file in glob.glob("*.fasta"):
        seqs = read_fasta(file)
        print(file, "original:", len(seqs))
        if len(seqs) > TARGET:
            seqs = random.sample(seqs, TARGET)
        out_file = "balanced_" + file
        write_fasta(seqs, out_file)
        print(" → saved:", out_file, len(seqs))


def combine_balanced_to_final():
    out = open("final_dataset.fasta", "w")
    seq_id = 1
    for file in glob.glob("balanced_*.fasta"):
        label = file.replace("balanced_", "").replace(".fasta", "")
        with open(file) as f:
            seq = ""
            for line in f:
                line = line.strip()
                if line.startswith(">"):
                    if seq:
                        out.write(seq + "\n")
                    out.write(f">SEQ_{seq_id:05d}|{label}\n")
                    seq_id += 1
                    seq = ""
                else:
                    seq += line
            if seq:
                out.write(seq + "\n")
    out.close()
    print("Saved: final_dataset.fasta")


def fasta_to_csv(input_fasta="final_dataset.fasta", output_csv="protein_family_dataset.csv"):
    rows = []
    with open(input_fasta) as f:
        seq_id = ""
        family = ""
        seq = ""
        for line in f:
            line = line.strip()
            if line.startswith(">"):
                if seq:
                    rows.append([
                        seq_id,
                        seq,
                        family,
                        len(seq)
                    ])
                header = line[1:]
                parts = header.split("|")
                seq_id = parts[0]
                family = parts[1]
                seq = ""
            else:
                seq += line
        if seq:
            rows.append([
                seq_id,
                seq,
                family,
                len(seq)
            ])

    with open(output_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "sequence_id",
            "sequence",
            "family",
            "sequence_length"
        ])
        writer.writerows(rows)

    print("Saved:", output_csv)
    print("Total sequences:", len(rows))


# Enhanced physicochemical grouping schemes
PHYSICOCHEMICAL_GROUPS = {
    'scheme_enhanced': {
        'A': 'GAVLIMFWP',
        'C': 'C',
        'S': 'STY',
        'N': 'NQ',
        'D': 'DE',
        'R': 'KRH',
        'P': 'P',
    },
    'scheme_hydropathy': {
        'H': 'AVILMFW',
        'N': 'GTP',
        'P': 'STYNQ',
        'C': 'DE',
        'B': 'KRH',
        'S': 'C',
    },
    'scheme_structural': {
        'S': 'GAVLIMFWP',
        'P': 'STYNQ',
        'C': 'DEKRH',
        'X': 'C',
    }
}


class EnhancedPhysioChemKmerClassifier:
    def __init__(self, scheme='scheme_enhanced', k=3, model_type='rf', use_composition=True, use_entropy=True):
        self.scheme = scheme
        self.k = k
        self.model_type = model_type
        self.use_composition = use_composition
        self.use_entropy = use_entropy
        self.mapping = self._create_mapping_dict(PHYSICOCHEMICAL_GROUPS[scheme])
        self.feature_names = None

        if model_type == 'logistic':
            self.model = LogisticRegression(max_iter=2000, random_state=42, C=0.1, class_weight='balanced')
        else:
            self.model = RandomForestClassifier(n_estimators=200, random_state=42, max_depth=20, class_weight='balanced')

    def _create_mapping_dict(self, groups):
        mapping = {}
        for code, aas in groups.items():
            for aa in aas:
                mapping[aa] = code
        return mapping

    def _sequence_to_property_string(self, sequence):
        return ''.join(self.mapping.get(aa, 'X') for aa in sequence)

    def _calculate_sequence_features(self, sequences):
        all_features = []
        properties = list(set(self.mapping.values()))
        from itertools import product
        all_possible_kmers = [''.join(combo) for combo in product(properties, repeat=self.k)]

        for seq in sequences:
            prop_seq = self._sequence_to_property_string(seq)
            features = []
            if len(prop_seq) >= self.k:
                kmers = [prop_seq[i:i+self.k] for i in range(len(prop_seq)-self.k+1)]
                kmer_counts = {kmer: kmers.count(kmer) for kmer in all_possible_kmers}
                total_kmers = len(kmers)
                kmer_freqs = [kmer_counts[kmer]/total_kmers if total_kmers > 0 else 0 for kmer in all_possible_kmers]
                features.extend(kmer_freqs)
            else:
                features.extend([0] * len(all_possible_kmers))

            if self.use_composition:
                aa_counts = Counter(seq)
                total_aas = len(seq)
                for prop in properties:
                    prop_count = sum(aa_counts.get(aa, 0) for aa in self.mapping.keys() if self.mapping[aa] == prop)
                    features.append(prop_count / total_aas if total_aas > 0 else 0)

            if len(prop_seq) > 1:
                transitions = []
                for i in range(len(prop_seq)-1):
                    transition = prop_seq[i] + prop_seq[i+1]
                    transitions.append(transition)
                for prop1 in properties:
                    for prop2 in properties:
                        transition = prop1 + prop2
                        features.append(transitions.count(transition) / len(transitions) if len(transitions) > 0 else 0)

            if self.use_entropy and len(prop_seq) > 0:
                prop_counts = Counter(prop_seq)
                entropy = 0
                for count in prop_counts.values():
                    p = count / len(prop_seq)
                    if p > 0:
                        entropy -= p * np.log2(p)
                features.append(entropy)

                runs = []
                current_run = 1
                for i in range(1, len(prop_seq)):
                    if prop_seq[i] == prop_seq[i-1]:
                        current_run += 1
                    else:
                        runs.append(current_run)
                        current_run = 1
                runs.append(current_run)
                features.append(np.mean(runs) if runs else 0)
                features.append(max(runs) if runs else 0)

            all_features.append(features)

        self.feature_names = []
        self.feature_names.extend([f"kmer_{kmer}" for kmer in all_possible_kmers])
        if self.use_composition:
            self.feature_names.extend([f"comp_{prop}" for prop in properties])
        if len(properties) > 0:
            self.feature_names.extend([f"trans_{p1}{p2}" for p1 in properties for p2 in properties])
        if self.use_entropy:
            self.feature_names.extend(["entropy", "avg_run", "max_run"])

        return np.array(all_features)

    def fit(self, sequences, labels):
        X = self._calculate_sequence_features(sequences)
        self.model.fit(X, labels)
        return self

    def predict(self, sequences):
        X = self._calculate_sequence_features(sequences)
        return self.model.predict(X)

    def get_feature_importance(self, top_n=10):
        if hasattr(self.model, 'feature_importances_'):
            importances = self.model.feature_importances_
        elif hasattr(self.model, 'coef_'):
            importances = np.mean(np.abs(self.model.coef_), axis=0)
        else:
            return []
        return sorted(zip(self.feature_names, importances), key=lambda x: abs(x[1]), reverse=True)[:top_n]

    def get_num_features(self):
        return len(self.feature_names) if self.feature_names else 0


def run_enhanced_analysis():
    df = pd.read_csv('final_protein_family_dataset.csv')
    sequences = df['sequence'].tolist()
    labels = df['family'].tolist()

    print(f"Dataset loaded: {len(sequences)} sequences, {len(set(labels))} families")

    X_train, X_test, y_train, y_test = train_test_split(
        sequences, labels, test_size=0.2, random_state=42, stratify=labels
    )

    print(f"Training set: {len(X_train)} sequences")
    print(f"Test set: {len(X_test)} sequences")

    methods = {
        'Standard_3mer_RF': {'scheme': None, 'model': 'rf', 'enhanced': False},
        'PhysioChem_Enhanced': {'scheme': 'scheme_enhanced', 'model': 'rf', 'enhanced': True},
        'PhysioChem_Hydropathy': {'scheme': 'scheme_hydropathy', 'model': 'rf', 'enhanced': True},
        'PhysioChem_Structural': {'scheme': 'scheme_structural', 'model': 'rf', 'enhanced': True},
    }

    results = []
    classifiers = {}

    for method_name, params in methods.items():
        print(f"\n{'='*60}")
        print(f"Training {method_name}")
        print(f"{'='*60}")

        start_time = time.time()

        if params['scheme'] is None:
            class StandardKmerClassifier:
                def __init__(self, max_features=1000):
                    self.max_features = max_features
                    self.model = RandomForestClassifier(n_estimators=200, random_state=42, max_depth=20)
                    self.feature_names = None

                def _extract_standard_kmers(self, sequences, k=3):
                    all_kmers = set()
                    for seq in sequences:
                        if len(seq) >= k:
                            kmers = [seq[i:i+k] for i in range(len(seq)-k+1)]
                            all_kmers.update(kmers)

                    if len(all_kmers) > self.max_features:
                        kmer_counts = Counter()
                        for seq in sequences:
                            if len(seq) >= k:
                                kmers = [seq[i:i+k] for i in range(len(seq)-k+1)]
                                kmer_counts.update(kmers)
                        self.feature_names = [kmer for kmer, count in kmer_counts.most_common(self.max_features)]
                    else:
                        self_feature_names = sorted(all_kmers)
                        self.feature_names = self_feature_names

                    feature_vectors = []
                    for seq in sequences:
                        if len(seq) >= k:
                            kmers = [seq[i:i+k] for i in range(len(seq)-k+1)]
                        else:
                            kmers = []
                        kmer_count = {kmer: kmers.count(kmer) for kmer in self.feature_names}
                        total = sum(kmer_count.values())
                        if total > 0:
                            feature_vector = [kmer_count[kmer]/total for kmer in self.feature_names]
                        else:
                            feature_vector = [0] * len(self.feature_names)
                        feature_vectors.append(feature_vector)

                    return np.array(feature_vectors)

                def fit(self, sequences, labels):
                    X = self._extract_standard_kmers(sequences)
                    self.model.fit(X, labels)
                    return self

                def predict(self, sequences):
                    X = self._extract_standard_kmers(sequences)
                    return self.model.predict(X)

                def get_num_features(self):
                    return len(self.feature_names) if self.feature_names else 0

            classifier = StandardKmerClassifier(max_features=1000)
        else:
            classifier = EnhancedPhysioChemKmerClassifier(
                scheme=params['scheme'],
                k=3,
                model_type=params['model'],
                use_composition=True,
                use_entropy=True
            )

        classifier.fit(X_train, y_train)
        training_time = time.time() - start_time

        start_time = time.time()
        y_pred = classifier.predict(X_test)
        prediction_time = time.time() - start_time

        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted')

        results.append({
            'Method': method_name,
            'Accuracy': accuracy,
            'F1_Score': f1,
            'Training_Time': training_time,
            'Prediction_Time': prediction_time,
            'Num_Features': classifier.get_num_features()
        })

        classifiers[method_name] = classifier

        print(f"Accuracy: {accuracy:.4f}")
        print(f"F1-Score: {f1:.4f}")
        print(f"Training Time: {training_time:.2f}s")
        print(f"Prediction Time: {prediction_time:.2f}s")
        print(f"Number of Features: {classifier.get_num_features()}")

        if params['scheme'] is not None and hasattr(classifier, 'get_feature_importance'):
            print(f"\nTop 5 features for {method_name}:")
            top_features = classifier.get_feature_importance(top_n=5)
            for feature, importance in top_features:
                biological_meaning = {
                    'AAA': 'Aliphatic core', 'CCC': 'Cysteine-rich', 'SSS': 'Serine-like cluster',
                    'NNN': 'Amide-rich', 'DDD': 'Acidic region', 'RRR': 'Basic region',
                    'HHH': 'Hydrophobic core', 'PPP': 'Polar surface', 'BBB': 'Basic cluster',
                    'comp_A': 'Aliphatic composition', 'comp_R': 'Basic composition',
                    'comp_D': 'Acidic composition', 'entropy': 'Sequence diversity',
                    'avg_run': 'Property homogeneity'
                }.get(feature, 'Functional pattern')
                print(f"  {feature}: {importance:.4f} ({biological_meaning})")

    return pd.DataFrame(results), classifiers


def create_enhanced_visualization(results_df, classifiers, X_test, y_test):
    plt.figure(figsize=(18, 12))
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    plt.subplot(2, 3, 1)
    x_pos = np.arange(len(results_df))
    width = 0.35
    bars1 = plt.bar(x_pos - width/2, results_df['Accuracy'], width, label='Accuracy', alpha=0.8, color=colors[:len(results_df)])
    bars2 = plt.bar(x_pos + width/2, results_df['F1_Score'], width, label='F1-Score', alpha=0.8, color=colors[:len(results_df)])
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.01, f'{height:.3f}', ha='center', va='bottom', fontsize=8)
    plt.xlabel('Method')
    plt.ylabel('Score')
    plt.title('A) Enhanced Classification Performance', fontweight='bold', fontsize=12)
    plt.xticks(x_pos, results_df['Method'], rotation=45)
    plt.legend()
    plt.ylim(0, 1.0)
    plt.grid(True, alpha=0.3)

    plt.subplot(2, 3, 2)
    methods = results_df['Method'].tolist()
    training_times = results_df['Training_Time'].tolist()
    prediction_times = results_df['Prediction_Time'].tolist()
    x = np.arange(len(methods))
    width = 0.35
    plt.bar(x - width/2, training_times, width, label='Training Time', color=colors[:len(methods)], alpha=0.7)
    plt.bar(x + width/2, prediction_times, width, label='Prediction Time', color=colors[:len(methods)], alpha=0.7)
    plt.xlabel('Method')
    plt.ylabel('Time (seconds)')
    plt.title('B) Computational Efficiency', fontweight='bold', fontsize=12)
    plt.xticks(x, methods, rotation=45)
    plt.legend()
    plt.yscale('log')
    plt.grid(True, alpha=0.3)

    plt.subplot(2, 3, 3)
    feature_counts = results_df['Num_Features'].tolist()
    bars = plt.bar(methods, feature_counts, color=colors[:len(methods)], alpha=0.7)
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 10, f'{height}', ha='center', va='bottom', fontsize=9)
    plt.xlabel('Method')
    plt.ylabel('Number of Features')
    plt.title('C) Feature Space Dimensionality', fontweight='bold', fontsize=12)
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)

    plt.subplot(2, 3, 4)
    for i, (_, row) in enumerate(results_df.iterrows()):
        plt.scatter(row['Num_Features'], row['Accuracy'], s=100, color=colors[i], alpha=0.7, label=row['Method'])
        plt.annotate(row['Method'], (row['Num_Features'], row['Accuracy']), xytext=(5, 5), textcoords='offset points', fontsize=9)
    plt.xlabel('Number of Features')
    plt.ylabel('Accuracy')
    plt.title('D) Accuracy vs Feature Complexity', fontweight='bold', fontsize=12)
    plt.grid(True, alpha=0.3)

    plt.subplot(2, 3, 5)
    best_method = results_df.loc[results_df['Accuracy'].idxmax(), 'Method']
    best_classifier = classifiers[best_method]
    y_pred = best_classifier.predict(X_test)
    cm = confusion_matrix(y_test, y_pred, labels=sorted(set(y_test)))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=sorted(set(y_test)), yticklabels=sorted(set(y_test)))
    plt.title(f'E) Confusion Matrix - {best_method}', fontweight='bold', fontsize=12)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)

    plt.subplot(2, 3, 6)
    plt.axis('off')
    best_row = results_df.loc[results_df['Accuracy'].idxmax()]
    insights = [
        "KEY INSIGHTS:",
        "",
        f"Best Method: {best_row['Method']}",
        f"Best Accuracy: {best_row['Accuracy']:.3f}",
        f"Features Used: {best_row['Num_Features']}",
        f"Training Time: {best_row['Training_Time']:.2f}s",
        "",
        "PhysioChem Advantages:",
        "- Biologically interpretable",
        "- Reduced feature space",
        "- Faster computation",
        "- Domain knowledge integration"
    ]
    plt.text(0.05, 0.95, '\n'.join(insights), transform=plt.gca().transAxes, fontfamily='monospace', fontsize=10, verticalalignment='top', bbox=dict(boxstyle="round,pad=0.3", facecolor='lightblue', alpha=0.7))
    plt.tight_layout()
    plt.savefig('enhanced_physiochem_results.png', dpi=300, bbox_inches='tight')
    plt.show()


def download_external_family(pfam_id, name, limit=200):
    url = "https://rest.uniprot.org/uniprotkb/stream"
    query = f"xref:pfam-{pfam_id} AND reviewed:false"
    params = {
        "query": query,
        "format": "fasta",
        "size": limit
    }
    headers = {"User-Agent": "Mozilla/5.0"}
    r = requests.get(url, params=params, timeout=60, headers=headers)
    print(f"{name}: status {r.status_code}")
    if r.status_code == 200 and len(r.text) > 100:
        filename = f"external_{name}.fasta"
        with open(filename, "w") as f:
            f.write(r.text)
        print(f"Saved → {filename}")
    else:
        print(f"Failed → {name}")


if __name__ == "__main__":
    print("Module loaded. Use the functions to download/process datasets or run analysis.")
