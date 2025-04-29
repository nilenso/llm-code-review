import json
import math
import os
import time
from typing import List

import click
import pandas as pd
from rich.console import Console
from rich.table import Table

from code_reviewer.indexer import CodeIndexer
from code_reviewer.reranker import ContextReranker

console = Console()

def calculate_precision_recall(retrieved: List[str], relevant: List[str]):
    """Calculate precision and recall metrics"""
    if not retrieved:
        return 0.0, 0.0
    if not relevant:
        return 0.0, 1.0

    retrieved_set = set(retrieved)
    relevant_set = set(relevant)

    intersection = retrieved_set.intersection(relevant_set)

    precision = len(intersection) / \
        len(retrieved_set) if retrieved_set else 0.0
    recall = len(intersection) / len(relevant_set) if relevant_set else 0.0

    return precision, recall


def calculate_mrr(retrieved_items: List[str], relevant_items: List[str]) -> float:
    """
    Calculate Mean Reciprocal Rank (MRR).

    For each relevant item, find its rank in the retrieved list,
    then take the reciprocal of that rank. The MRR is the mean
    of these reciprocal ranks.

    Args:
        retrieved_items: List of retrieved items in rank order
        relevant_items: List of relevant items

    Returns:
        Mean Reciprocal Rank score
    """
    if not retrieved_items or not relevant_items:
        return 0.0

    # Find the first relevant item and its rank
    for i, item in enumerate(retrieved_items):
        if item in relevant_items:
            return 1.0 / (i + 1)  # +1 because ranks start at 1, not 0

    return 0.0  # No relevant items found


def calculate_ndcg(retrieved_items: List[str], relevant_items: List[str], k: int = 5) -> float:
    """
    Calculate Normalized Discounted Cumulative Gain (NDCG) at rank k.

    Args:
        retrieved_items: List of retrieved items in rank order
        relevant_items: List of relevant items
        k: Number of items to consider (default: 5)

    Returns:
        NDCG score
    """
    if not retrieved_items or not relevant_items:
        return 0.0

    # Truncate retrieved_items to top k
    retrieved_items = retrieved_items[:k]

    # Calculate DCG
    dcg = 0.0
    for i, item in enumerate(retrieved_items):
        if item in relevant_items:
            # Using binary relevance (1 if relevant, 0 if not)
            # Position i+1 because positions are 1-indexed in the formula
            dcg += 1.0 / math.log2(i + 2)  # +2 because log base 2 of 1 is 0

    # Calculate ideal DCG (IDCG)
    # In the ideal case, all relevant items would be at the top
    idcg = 0.0
    for i in range(min(len(relevant_items), k)):
        idcg += 1.0 / math.log2(i + 2)

    # Calculate NDCG
    ndcg = dcg / idcg if idcg > 0 else 0.0

    return ndcg


def calculate_hit_rate(retrieved_items: List[str], relevant_items: List[str], k: int = 5) -> float:
    """
    Calculate Hit Rate at rank k.

    Hit Rate is 1 if any relevant item is in the top k retrieved items, 0 otherwise.

    Args:
        retrieved_items: List of retrieved items in rank order
        relevant_items: List of relevant items
        k: Number of items to consider (default: 5)

    Returns:
        Hit Rate (1.0 or 0.0)
    """
    if not retrieved_items or not relevant_items:
        return 0.0

    # Truncate retrieved_items to top k
    retrieved_top_k = retrieved_items[:k]

    # Check if any relevant item is in the top k
    for item in retrieved_top_k:
        if item in relevant_items:
            return 1.0

    return 0.0


def retrieve_and_rank(indexer: CodeIndexer, reranker: ContextReranker, test_case: dict):
    """
    Retrieves context using the indexer, reranks it, and extracts files and symbols.

    Args:
        indexer: The CodeIndexer instance.
        reranker: The ContextReranker instance.
        test_case: A dictionary representing the test case, containing the 'query'.
        n_results: The number of initial results to retrieve.

    Returns:
        A tuple containing:
            - retrieval_time (float): Time taken for retrieval and reranking.
            - retrieved_files (List[str]): List of unique file paths retrieved, in rank order.
            - retrieved_symbols (List[str]): List of unique symbols retrieved, in rank order.
    """
    
    retrieved_context = indexer.retrieve_context(test_case['query'], n_results=30)
    if not retrieved_context:
        return [], [], [] # Return empty lists for context, files, and symbols

    initial_retrieved_files = [doc.get('file_path') for doc in retrieved_context if doc.get('file_path')] # Get initial file paths
    console.print(f"  [cyan]Initial retrieval (Top {len(initial_retrieved_files)}):[/cyan] {list(dict.fromkeys(initial_retrieved_files))[:10]}") # Print unique top 10

    reranked_context_docs = reranker.rank(test_case['query'], [doc['content'] for doc in retrieved_context])

    # Map reranked documents back to original context dictionaries, preserving order
    # Handle potential KeyError if reranker returns something not in the original map
    reranked_context_map = {doc['content']: doc for doc in retrieved_context}
    reranked_context = [] # Initialize as empty list
    for doc in reranked_context_docs:
        if doc.text in reranked_context_map:
             reranked_context.append(reranked_context_map[doc.text])
        else:
            # Log or handle cases where reranked text isn't found in the original map
            console.print(f"[yellow]Warning:[/yellow] Reranked document text not found in original context map for query: {test_case['query']}")

    # Extract retrieved files and symbols in rank order (maintaining order from reranked_context)
    retrieved_files = []
    for doc in reranked_context:
        file_path = doc.get('file_path') # Use .get for safety
        if file_path and file_path not in retrieved_files:
            retrieved_files.append(file_path)

    retrieved_symbols = []
    for doc in reranked_context:
        symbols = doc.get('symbols', []) # Use .get with default for safety
        for symbol in symbols:
            if symbol not in retrieved_symbols:
                retrieved_symbols.append(symbol)

    return reranked_context, retrieved_files, retrieved_symbols


@click.command(help="RAG context retrieval evaluator")
@click.option("--test-cases-file", default="./ocra/evals/test_cases.json", help="Path to test cases file")
@click.option("--repo-path", default="./", help="Path to repository root")
@click.option("--embedding-model", default="nomic-embed-text",
              help="The embedding model to evaluate")
@click.option("--reranking-model", default="BAAI/bge-reranker-base",
              help="The reranking model to evaluate")
@click.option("--output-dir", default="./retrieval_results", help="Path to write reports to")
@click.option("--test-report-tag", default="", help="Tag added to filename to help label experiments")
def evaluate_retrieval(test_cases_file, repo_path, embedding_model, reranking_model, output_dir, test_report_tag):
    """Evaluate the context retrieval system using the test cases.

    Args:
        test_cases_file: Path to the JSON file containing test cases
        repo_path: Path to the repository root
        output_dir: Directory to save evaluation results
        test_report_tag: Filename tag to label experiments
    """
    reranker = ContextReranker(model=reranking_model)
    results = []

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Load test cases from file
    with open(test_cases_file, 'r') as f:
        test_cases = json.load(f)

    console.print(f"Initializing code indexer for repository at {repo_path}...")
    indexer = CodeIndexer(repo_path=repo_path, embedding_model=embedding_model, ollama_host="http://localhost:11434")
    console.print("Indexing repository (this may take a while if not already indexed)...")
    indexer.index_repository(force_reindex=True)
    console.print(f"Evaluating {len(test_cases)} test cases using model {embedding_model}")

    for test_case in test_cases:
        console.print(f"\n[bold]Evaluating test case:[/bold] {test_case['name']}")
        console.print(f"Query: {test_case['query']}")

        start_time = time.time()
        reranked_context, retrieved_files, retrieved_symbols = retrieve_and_rank(indexer, reranker, test_case)
        retrieval_time = time.time() - start_time

        # Calculate traditional metrics
        file_precision, file_recall = calculate_precision_recall(retrieved_files, test_case['relevant_files'])
        symbol_precision, symbol_recall = calculate_precision_recall(retrieved_symbols, test_case['relevant_symbols'])

        # Calculate F1 scores
        file_f1 = 2 * (file_precision * file_recall) / (file_precision +
                                                        file_recall) if (file_precision + file_recall) > 0 else 0
        symbol_f1 = 2 * (symbol_precision * symbol_recall) / (symbol_precision +
                                                              symbol_recall) if (symbol_precision + symbol_recall) > 0 else 0

        # Calculate ranking metrics for files
        file_mrr = calculate_mrr(retrieved_files, test_case['relevant_files'])
        file_ndcg = calculate_ndcg(retrieved_files, test_case['relevant_files'], k=10)
        file_hit_rate = calculate_hit_rate(retrieved_files, test_case['relevant_files'], k=10)

        # Calculate ranking metrics for symbols
        symbol_mrr = calculate_mrr(retrieved_symbols, test_case['relevant_symbols'])
        symbol_ndcg = calculate_ndcg(retrieved_symbols, test_case['relevant_symbols'], k=10)
        symbol_hit_rate = calculate_hit_rate(retrieved_symbols, test_case['relevant_symbols'], k=10)

        # Print results
        console.print(f"Retrieved files: {retrieved_files}")
        console.print(f"Relevant files: {test_case['relevant_files']}")
        console.print(f"[green]File metrics:[/green]")
        console.print(f"  Precision: {file_precision:.4f}, Recall: {file_recall:.4f}, F1: {file_f1:.4f}")
        console.print(f"  MRR: {file_mrr:.4f}, NDCG@10: {file_ndcg:.4f}, Hit Rate@10: {file_hit_rate:.4f}")
        console.print(f"[green]Symbol metrics:[/green]")
        console.print(f"  Precision: {symbol_precision:.4f}, Recall: {symbol_recall:.4f}, F1: {symbol_f1:.4f}")
        console.print(f"  MRR: {symbol_mrr:.4f}, NDCG@10: {symbol_ndcg:.4f}, Hit Rate@10: {symbol_hit_rate:.4f}")
        console.print(f"Retrieval time: {retrieval_time:.4f} seconds")

        # Add to results
        results.append({
            'test_case': test_case['name'],
            'query': test_case['query'],
            # Traditional metrics
            'file_precision': file_precision,
            'file_recall': file_recall,
            'file_f1': file_f1,
            'symbol_precision': symbol_precision,
            'symbol_recall': symbol_recall,
            'symbol_f1': symbol_f1,
            # Ranking metrics
            'file_mrr': file_mrr,
            'file_ndcg': file_ndcg,
            'file_hit_rate': file_hit_rate,
            'symbol_mrr': symbol_mrr,
            'symbol_ndcg': symbol_ndcg,
            'symbol_hit_rate': symbol_hit_rate,
            # Performance metrics
            'retrieval_time': retrieval_time,
            'chunk_count': len(reranked_context)
        })

    df = pd.DataFrame(results)
    timestamp = time.strftime("%Y%m%d_%H%M%S")

    report_file = f'retrieval_results_{timestamp}_{test_report_tag}.csv'
    results_file = os.path.join(output_dir, report_file)
    df.to_csv(results_file, index=False)

    console.print(f"\nResults saved to {results_file}")

    # Create primary metrics table
    table = Table(title="RETRIEVAL EVALUATION SUMMARY", show_header=False)
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")
    table.add_row("Total test cases", f"{len(df)}")
    table.add_row("", "")
    table.add_row("[bold]File Retrieval Metrics[/bold]", "")
    table.add_row("Precision", f"{df['file_precision'].mean():.4f}")
    table.add_row("Recall", f"{df['file_recall'].mean():.4f}")
    table.add_row("F1", f"{df['file_f1'].mean():.4f}")
    table.add_row("MRR", f"{df['file_mrr'].mean():.4f}")
    table.add_row("NDCG@10", f"{df['file_ndcg'].mean():.4f}")
    table.add_row("Hit Rate@10", f"{df['file_hit_rate'].mean():.4f}")
    table.add_row("", "")
    table.add_row("[bold]Symbol Retrieval Metrics[/bold]", "")
    table.add_row("Precision", f"{df['symbol_precision'].mean():.4f}")
    table.add_row("Recall", f"{df['symbol_recall'].mean():.4f}")
    table.add_row("F1", f"{df['symbol_f1'].mean():.4f}")
    table.add_row("MRR", f"{df['symbol_mrr'].mean():.4f}")
    table.add_row("NDCG@10", f"{df['symbol_ndcg'].mean():.4f}")
    table.add_row("Hit Rate@10", f"{df['symbol_hit_rate'].mean():.4f}")
    table.add_row("", "")
    table.add_row("Average retrieval time", f"{df['retrieval_time'].mean():.4f} seconds")
    print("\n")
    console.print(table)


if __name__ == "__main__":
    evaluate_retrieval()
