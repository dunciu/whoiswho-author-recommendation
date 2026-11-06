import json
import numpy as np
import math
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import MiniBatchKMeans


def open_file():
    """
    Loads JSON data from 'data/pid_to_info_all.json'.

    Returns:
        dict or list: The loaded JSON data if successful, None otherwise.
    """
    try:
        # Open the file with UTF-8 encoding
        with open("data/pid_to_info_all.json", "r", encoding="utf-8") as file:
            x = json.load(file)
        print("JSON data loaded successfully!")
        return x 
    except FileNotFoundError:
        print("Error: The file 'data/pid_to_info_all.json' was not found. Please check the file path and name.")
        return None  
    except json.JSONDecodeError:
        print("Error: Could not decode JSON. The file might be corrupted or not valid JSON.")
        return None
    except UnicodeDecodeError:
        print("Error: UnicodeDecodeError. The file might not be UTF-8 encoded, or contains invalid characters.")
        return None
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return None

def calculate_ndcg(relevance_scores, k):
    """
    Calculates the Normalized Discounted Cumulative Gain (NDCG) for a given list of relevance scores.
    
    Arguments:
        relevance_scores (list): A list of relevance scores for a ranked list of items.
        k (int): The number of top items to consider.

    Returns:
        float: The NDCG score.
    """
    if not relevance_scores:
        return 0.0
    
    # Calculate Discounted Cumulative Gain (DCG)
    dcg = 0.0
    for i in range(min(k, len(relevance_scores))):
        dcg += relevance_scores[i] / math.log2(i + 2)
        
    # Sort scores to get the ideal DCG (IDCG)
    sorted_scores = sorted(relevance_scores, reverse=True)
    idcg = 0.0
    for i in range(min(k, len(sorted_scores))):
        idcg += sorted_scores[i] / math.log2(i + 2)
        
    # Avoid division by zero
    if idcg == 0.0:
        return 0.0
    
    return dcg / idcg

def run_author_prediction(papers_list, num_test_papers, X_values_to_test):
    """
    Runs the similarity-based author prediction model and evaluates using NDCG.
    """
    print("\n--- Running Similarity-Based Model (NDCG Score) ---")
    
    ndcg_results = {X: [] for X in X_values_to_test}
    alpha = 0.8  # Weighting factor for similarity vs. references

    # Iterate through the papers chronologically
    for i in range(num_test_papers):
        test_paper = papers_list[i]
        
        past_papers = papers_list[:i]
        
        if len(past_papers) < 50:
            continue

        past_corpus = [
            p.get('title', '') + ' ' +
            p.get('abstract', '') + ' ' +
            ' '.join(p.get('keywords', []))
            for p in past_papers
        ]

        vectorizer = TfidfVectorizer(stop_words='english', max_df=0.85, min_df=3, ngram_range=(1, 2))
        past_tfidf_matrix = vectorizer.fit_transform(past_corpus)

        num_clusters = 25 # can change
        kmeans = MiniBatchKMeans(n_clusters=num_clusters, random_state=0, n_init=10)
        kmeans.fit(past_tfidf_matrix)

        test_paper_corpus = [
            test_paper.get('title', '') + ' ' +
            test_paper.get('abstract', '') + ' ' +
            ' '.join(test_paper.get('keywords', []))
        ]
        test_paper_vector = vectorizer.transform(test_paper_corpus)

        test_paper_cluster = kmeans.predict(test_paper_vector)

        same_cluster_indices = [idx for idx, label in enumerate(kmeans.labels_) if label == test_paper_cluster]
        
        if not same_cluster_indices:
            continue

        same_cluster_vectors = past_tfidf_matrix[same_cluster_indices]
        sim_scores = cosine_similarity(test_paper_vector, same_cluster_vectors).flatten()
        
        # Combine original indices and their similarity scores
        ranked_candidates = sorted(zip(same_cluster_indices, sim_scores), key=lambda x: x[1], reverse=True)
        
        # Get actual authors and references
        actual_authors = {author['name'] for author in test_paper.get('authors', [])}
        test_references = {ref['name'] for ref in test_paper.get('references', []) if isinstance(ref, dict) and 'name' in ref}

        for X in X_values_to_test:
            relevance_scores = []
            
            for rank, (original_idx, sim_score) in enumerate(ranked_candidates[:X]):
                candidate_authors = {author['name'] for author in papers_list[original_idx].get('authors', [])}
                
                # Check for correct author prediction
                is_correct = bool(actual_authors.intersection(candidate_authors))
                
                # Check for reference relevance
                is_referenced = bool(test_references.intersection(candidate_authors))
                
                # Calculate relevance score based on our formula
                if is_correct:
                    relevance = (alpha * (1 / (rank + 1))) + ((1 - alpha) * is_referenced)
                else:
                    relevance = 0.0
                
                relevance_scores.append(relevance)
            
            ndcg_score = calculate_ndcg(relevance_scores, X)
            ndcg_results[X].append(ndcg_score)
            
    # Calculate average NDCG scores
    final_results = {}
    for X in X_values_to_test:
        if ndcg_results[X]:
            final_results[X] = (sum(ndcg_results[X]) / len(ndcg_results[X]))
        else:
            final_results[X] = 0.0
            
    return final_results


def run_random_baseline(papers_list, num_test_papers, X_values_to_test):
    """
    Runs a random baseline for author prediction and evaluates using NDCG.
    """
    print("\n--- Running Random Baseline (NDCG Score) ---")
    
    ndcg_results = {X: [] for X in X_values_to_test}
    alpha = 0.8

    for i in range(num_test_papers):
        test_paper = papers_list[i]
        past_papers_indices = list(range(i))

        if len(past_papers_indices) < X_values_to_test[-1]:
            continue
        
        actual_authors = {author['name'] for author in test_paper.get('authors', [])}
        test_references = {ref['name'] for ref in test_paper.get('references', []) if isinstance(ref, dict) and 'name' in ref}
        
        for X in X_values_to_test:
            random_indices = np.random.choice(past_papers_indices, size=X, replace=False)
            relevance_scores = []
            
            for rank, idx in enumerate(random_indices):
                candidate_authors = {author['name'] for author in papers_list[idx].get('authors', [])}
                
                is_correct = bool(actual_authors.intersection(candidate_authors))
                is_referenced = bool(test_references.intersection(candidate_authors))
                
                if is_correct:
                    relevance = (alpha * (1 / (rank + 1))) + ((1 - alpha) * is_referenced)
                else:
                    relevance = 0.0
                    
                relevance_scores.append(relevance)
            
            ndcg_score = calculate_ndcg(relevance_scores, X)
            ndcg_results[X].append(ndcg_score)

    final_results = {}
    for X in X_values_to_test:
        if ndcg_results[X]:
            final_results[X] = (sum(ndcg_results[X]) / len(ndcg_results[X]))
        else:
            final_results[X] = 0.0
            
    return final_results


def main():
    data = open_file()
    if not data:
        return
    
    papers_list = list(data.values())
    
    # Sort papers by year
    papers_list.sort(key=lambda p: int(p.get('year') or 0))

    papers_list = [p for p in papers_list if p.get('year')]

    num_total_papers = len(papers_list)
    
    if num_total_papers == 0:
        print("No papers to test.")
        return
    
    num_test_papers = min(100, num_total_papers)
    X_values = [1, 3, 5, 7, 10, 15, 20]
    
    print(f"Total papers loaded: {num_total_papers}")
    print(f"Running evaluation on the first {num_test_papers} chronologically sorted papers...")
    
    similarity_results = run_author_prediction(papers_list, num_test_papers, X_values)
    random_results = run_random_baseline(papers_list, num_test_papers, X_values)
    
    print("\n--- Final Performance Comparison (NDCG) ---")
    print(f"{'X Value':<10} | {'Similarity-Based Avg. NDCG':<30} | {'Random Baseline Avg. NDCG':<30}")
    print("-" * 75)
    for X in X_values:
        sim_ndcg = similarity_results.get(X, 0)
        rand_ndcg = random_results.get(X, 0)
        print(f"{X:<10} | {sim_ndcg:>25.4f} | {rand_ndcg:>25.4f}")


if __name__ == "__main__":
    main()