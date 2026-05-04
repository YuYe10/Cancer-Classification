#!/usr/bin/env python3
"""Pathway enrichment analysis for key features in BRCA molecular subtype classification."""

import argparse
import json
from pathlib import Path
from typing import List, Dict, Tuple

import numpy as np
import pandas as pd


LABEL_MAP = {
    'LumA': 0,
    'LumB': 1,
    'HER2': 2,
    'Basal': 3,
}
INV_LABEL_MAP = {v: k for k, v in LABEL_MAP.items()}


def load_shap_results(json_path: Path) -> Dict:
    """Load SHAP values from JSON results."""
    with open(json_path, 'r') as f:
        data = json.load(f)
    return data


def get_top_features_by_class(shap_data: Dict, top_k: int = 50) -> Dict[int, List[str]]:
    """Extract top-k important features for each class based on mean absolute SHAP values.

    Args:
        shap_data: SHAP analysis results
        top_k: Number of top features to extract per class

    Returns:
        dict mapping class_id to list of feature names
    """
    class_features = {}

    feature_values = shap_data.get('feature_values', {})
    class_shap_means = shap_data.get('class_shap_means', {})

    for class_id_str, shap_means in class_shap_means.items():
        class_id = int(class_id_str)
        feature_names = list(shap_means.keys())
        shap_values = [abs(shap_means[f]) for f in feature_names]

        if len(feature_names) != len(shap_values):
            continue

        sorted_indices = np.argsort(shap_values)[::-1][:top_k]
        top_features = [feature_names[i] for i in sorted_indices]
        class_features[class_id] = top_features

    return class_features


def run_enrichr_analysis(features: List[str], organism: str = 'Human') -> Dict:
    """Run Enrichr pathway enrichment analysis.

    Args:
        features: List of gene symbols
        organism: Organism name for Enrichr

    Returns:
        dict with enrichment results
    """
    try:
        import gseapy as gp
    except ImportError:
        print("gseapy not installed. Please install with: pip install gseapy")
        return {}

    enr = gp.Enrichr(
        gene_list=features,
        organism=organism,
        gene_sets=['KEGG_2021_Human', 'GO_Biological_Process_2021'],
        background=None,
        outdir=None,
        no_plot=True
    )

    results = {}
    for gene_set_name, df in enr.results.items():
        if df is not None and len(df) > 0:
            results[gene_set_name] = df.head(20).to_dict('records')

    return results


def format_enrichment_table(enr_results: Dict, gene_set_name: str) -> str:
    """Format enrichment results as markdown table.

    Args:
        enr_results: Enrichment results dictionary
        gene_set_name: Name of the gene set

    Returns:
        markdown formatted string
    """
    if gene_set_name not in enr_results:
        return f"No results for {gene_set_name}"

    records = enr_results[gene_set_name]
    if not records:
        return f"No significant results for {gene_set_name}"

    lines = [
        f"### {gene_set_name}",
        "",
        "| Term | Overlap | Adj. P-value | Genes |",
        "| --- | ---: | ---: | --- |"
    ]

    for record in records:
        term = record.get('Term', 'N/A')
        overlap = record.get('Overlap', 'N/A')
        pval = record.get('Adj. P-value', 1.0)
        genes = record.get('Genes', '')[:100]
        lines.append(f"| {term} | {overlap} | {pval:.2e} | {genes} |")

    return "\n".join(lines)


def generate_enrichment_report(class_features: Dict[int, List[str]], class_names: List[str],
                                output_path: Path) -> str:
    """Generate comprehensive enrichment report for all classes.

    Args:
        class_features: dict mapping class_id to feature list
        class_names: list of class names
        output_path: path to save report

    Returns:
        report content as string
    """
    sections = [
        "# Pathway Enrichment Analysis Report",
        "",
        "## Methods",
        "",
        "Pathway enrichment analysis was performed using Enrichr with KEGG 2021 Human and",
        "Gene Ontology Biological Process 2021 gene sets. Top 50 features by mean absolute",
        "SHAP values were used for each class.",
        "",
    ]

    for class_id, features in class_features.items():
        class_name = class_names[class_id] if class_id < len(class_names) else f"Class {class_id}"
        sections.append(f"## {class_name} Subtype Features")

        sections.append(f"\nTop {len(features)} features analyzed: {', '.join(features[:10])}...")
        sections.append("")

        enr_results = run_enrichr_analysis(features)

        if enr_results:
            for gene_set in ['KEGG_2021_Human', 'GO_Biological_Process_2021']:
                table = format_enrichment_table(enr_results, gene_set)
                sections.append(table)
                sections.append("")
        else:
            sections.append("Enrichment analysis could not be performed (gseapy not available or no results).")
            sections.append("")

    report_content = "\n".join(sections)

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(report_content)

    return report_content


def create_feature_mechanism_mapping(class_features: Dict[int, List[str]],
                                     class_names: List[str]) -> pd.DataFrame:
    """Create mapping table of key features to biological mechanisms.

    Args:
        class_features: dict mapping class_id to feature list
        class_names: list of class names

    Returns:
        DataFrame with feature-mechanism mappings
    """
    mappings = []

    known_markers = {
        'LumA': ['ESR1', 'GATA3', 'FOXA1', 'MAPT', 'KRT8', 'KRT18', 'ANXA4', 'XBP1'],
        'LumB': ['MKI67', 'CDC20', 'TOP2A', 'AURKB', 'CCNB1', 'UBE2C', 'RRM2', 'PLK1'],
        'HER2': ['ERBB2', 'GRB7', 'EGFR', 'HER2', 'CDH3', 'PGR', 'STAT1', 'IL6'],
        'Basal': ['KRT5', 'KRT6A', 'KRT6B', 'KRT14', 'KRT17', 'EGFR', 'FN1', 'S100A8']
    }

    for class_id, features in class_features.items():
        class_name = class_names[class_id] if class_id < len(class_names) else f"Class {class_id}"

        for feature in features[:20]:
            clean_feature = feature.replace('ENSG', '').split('.')[0]

            is_known = 'Yes' if clean_feature in known_markers.get(class_name, []) else 'Unknown'

            mappings.append({
                'Class': class_name,
                'Feature': feature,
                'Known Marker': is_known,
                'Category': 'RNA' if feature.startswith('ENSG') else 'Methylation'
            })

    return pd.DataFrame(mappings)


def main():
    parser = argparse.ArgumentParser(
        description='Pathway enrichment analysis for BRCA subtypes'
    )
    parser.add_argument('--shap-json', type=str, required=True,
                        help='Path to SHAP analysis JSON file')
    parser.add_argument('--top-k', type=int, default=50,
                        help='Number of top features per class')
    parser.add_argument('--output-dir', type=str, default='outputs/logs',
                        help='Output directory')
    parser.add_argument('--class-names', type=str,
                        default='LumA,LumB,HER2,Basal',
                        help='Comma-separated class names')
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    class_names = args.class_names.split(',')

    shap_data = load_shap_results(Path(args.shap_json))

    class_features = get_top_features_by_class(shap_data, top_k=args.top_k)

    print(f"Extracted features for {len(class_features)} classes")

    report_path = output_dir / 'pathway_enrichment_report.md'
    generate_enrichment_report(class_features, class_names, report_path)
    print(f"Saved enrichment report: {report_path}")

    mapping_df = create_feature_mechanism_mapping(class_features, class_names)
    mapping_path = output_dir / 'feature_mechanism_mapping.csv'
    mapping_df.to_csv(mapping_path, index=False)
    print(f"Saved feature-mechanism mapping: {mapping_path}")

    print(f"\nFeature summary:")
    for class_id, features in class_features.items():
        class_name = class_names[class_id] if class_id < len(class_names) else f"Class {class_id}"
        print(f"  {class_name}: {len(features)} features")


if __name__ == '__main__':
    main()