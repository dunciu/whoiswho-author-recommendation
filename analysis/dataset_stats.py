import json

def open_file(file_path="data/pid_to_info_all.json"):
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            return json.load(file)
    except Exception as e:
        print(f"Error loading file: {e}")
        return None

def compute_metrics(data, start_year, end_year):
    total_papers = 0
    total_authors_count = 0
    unique_authors = set()
    unique_venues = set()
    unique_institutions = set()

    for paper in data.values():
        if not isinstance(paper, dict):
            continue
            
        raw_year = paper.get('year')
        
        # Safe integer conversion
        year = None
        if raw_year is not None:
            try:
                year = int(raw_year)
            except (ValueError, TypeError):
                continue  # Skip records with unparseable years
        
        # Inclusive filtering: start_year <= year <= end_year
        if year is not None and start_year <= year <= end_year:
            total_papers += 1
            
            # Venue tracking
            venue = paper.get('venue')
            if venue:
                unique_venues.add(venue)
                
            # Authors and Institutions tracking
            authors_list = paper.get('authors', [])
            total_authors_count += len(authors_list)
            
            for author in authors_list:
                if isinstance(author, dict):
                    name = author.get('name')
                    org = author.get('org')
                    
                    if name:
                        unique_authors.add(name)
                    if org:
                        unique_institutions.add(org)

    avg_authors_per_paper = (total_authors_count / total_papers) if total_papers > 0 else 0.0

    return {
        "Period": f"{start_year}-{end_year}",
        "Total Papers": total_papers,
        "Unique Authors": len(unique_authors),
        "Unique Venues": len(unique_venues),
        "Avg Authors/Paper": round(avg_authors_per_paper, 2),
        "Unique Institutions": len(unique_institutions)
    }

def main():
    data = open_file()
    if not data:
        return

    ranges = [
        (2010, 2015),
        (2015, 2020),
        (2010, 2020)
    ]

    print(f"\n{'Period':<12} | {'Total Papers':<12} | {'Unique Authors':<15} | {'Unique Venues':<15} | {'Avg Authors/Paper':<18} | {'Unique Institutions':<18}")
    print("-" * 105)

    for start, end in ranges:
        metrics = compute_metrics(data, start, end)
        print(f"{metrics['Period']:<12} | "
              f"{metrics['Total Papers']:<12} | "
              f"{metrics['Unique Authors']:<15} | "
              f"{metrics['Unique Venues']:<15} | "
              f"{metrics['Avg Authors/Paper']:<18} | "
              f"{metrics['Unique Institutions']:<18}")

if __name__ == "__main__":
    main()