import json
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def open_file():
    """
    Loads JSON data from 'data/pid_to_info_all.json' with UTF-8 encoding.

    Returns:
        dict or list: The loaded JSON data if successful, None otherwise.
    """
    try:
        # Open the file with UTF-8 encoding
        with open("data/pid_to_info_all.json", "r", encoding="utf-8") as file:
            x = json.load(file)
        print("JSON data loaded successfully!")
        return x # Return the loaded data
    except FileNotFoundError:
        print("Error: The file 'data/pid_to_info_all.json' was not found. Please check the file path and name.")
        return None # Return None on error
    except json.JSONDecodeError:
        print("Error: Could not decode JSON. The file might be corrupted or not valid JSON.")
        return None
    except UnicodeDecodeError:
        print("Error: UnicodeDecodeError. The file might not be UTF-8 encoded, or contains invalid characters.")
        return None
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return None


def run_author_prediction(papers_list, num_test_papers, X_values_to_test):
    """
    Runs the similarity-based author prediction model.
    """
    print("\n--- Running Similarity-Based Model ---")
    print("Creating text vectors with TF-IDF...")
    corpus = [p.get('title', '') + ' ' + p.get('abstract', '') for p in papers_list]
    vectorizer = TfidfVectorizer(stop_words='english', max_df=0.85, min_df=3, ngram_range=(1,2))
    tfidf_matrix = vectorizer.fit_transform(corpus)
    print("Vectorization complete.")

    results = {}
    for X in X_values_to_test:
        hits = 0
        for i in range(num_test_papers):
            sim_scores = cosine_similarity(tfidf_matrix[i], tfidf_matrix).flatten()
            top_x_indices = np.argsort(sim_scores)[::-1][1:X+1]
            
            candidate_authors = set()
            for idx in top_x_indices:
                authors_of_paper = {author['name'] for author in papers_list[idx].get('authors', [])}
                candidate_authors.update(authors_of_paper)
            
            actual_authors = {author['name'] for author in papers_list[i].get('authors', [])}

            if actual_authors.intersection(candidate_authors):
                hits += 1

        hit_rate = (hits / num_test_papers) * 100
        results[X] = hit_rate
    
    return results


def run_random_baseline(papers_list, num_test_papers, X_values_to_test):
    """
    Runs a random baseline for author prediction.
    """
    print("\n--- Running Random Baseline ---")
    
    results = {}
    for X in X_values_to_test:
        hits = 0
        for i in range(num_test_papers):
            # Select X random indices, making sure not to pick the paper itself (i)
            all_indices = list(range(len(papers_list)))
            all_indices.pop(i) # Remove the current paper's index
            random_indices = np.random.choice(all_indices, size=X, replace=False)
            
            candidate_authors = set()
            for idx in random_indices:
                authors_of_paper = {author['name'] for author in papers_list[idx].get('authors', [])}
                candidate_authors.update(authors_of_paper)
            
            actual_authors = {author['name'] for author in papers_list[i].get('authors', [])}
            
            if actual_authors.intersection(candidate_authors):
                hits += 1
        
        hit_rate = (hits / num_test_papers) * 100
        results[X] = hit_rate
        
    return results


def main():
    data = open_file()
    if not data:
        return
    
    papers_list = list(data.values())
    num_total_papers = len(papers_list)
    
    if num_total_papers == 0:
        print("No papers to test.")
        return
    
    # Set the number of papers to test. Use min() to prevent errors
    # if the dataset is smaller than the desired test size.
    num_test_papers = min(100, num_total_papers)
    X_values = [1, 3, 5, 7, 10]
    
    print(f"Total papers loaded: {num_total_papers}")
    print(f"Running evaluation on {num_test_papers} papers...")
    
    # Run the similarity-based method
    similarity_results = run_author_prediction(papers_list, num_test_papers, X_values)
    
    # Run the random baseline
    random_results = run_random_baseline(papers_list, num_test_papers, X_values)
    
    # Print a final comparison
    print("\n--- Final Performance Comparison ---")
    print(f"{'X Value':<10} | {'Similarity-Based Hit Rate':<30} | {'Random Baseline Hit Rate':<30}")
    print("-" * 75)
    for X in X_values:
        sim_rate = similarity_results.get(X, 0)
        rand_rate = random_results.get(X, 0)
        print(f"{X:<10} | {sim_rate:>25.2f}% | {rand_rate:>25.2f}%")


if __name__ == "__main__":
    main()