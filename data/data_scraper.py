"""
News article scraper for stock price prediction.
"""
import pandas as pd
from datetime import datetime
from typing import List, Dict, Any, Union
import time
from urllib.parse import urlparse
import sys
import os
import argparse
from googlesearch import search

# Add project root to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.utils import (
    get_sentiment,
    fetch_article_content,
    is_credible_source,
    extract_publication_date,
    deduplicate_articles,
    filter_and_sample,
    save_articles_to_csv
)

# Stock symbols and their corresponding company names for better search results
STOCK_MAPPING = {
    "AAPL": "Apple Inc",
    "AMZN": "Amazon",
    "BAC": "Bank of America",
    "COP": "ConocoPhillips",
    "CVX": "Chevron",
    "GOOG": "Google Alphabet",
    "JPM": "JPMorgan Chase",
    "META": "Meta Facebook",
    "MSFT": "Microsoft",
    "NFLX": "Netflix",
    "NVDA": "NVIDIA",
    "TGT": "Target",
    "TSLA": "Tesla",
    "WMT": "Walmart",
    "XOM": "Exxon Mobil"
}

def fetch_news(company: str, start_date: Union[str, datetime], end_date: Union[str, datetime], ticker: str) -> List[Dict[str, Any]]:
    """
    Fetch news articles using Google search for specific dates.
    
    Args:
        company (str): Company name to search for
        start_date (str or datetime): Start date in YYYY-MM-DD format
        end_date (str or datetime): End date in YYYY-MM-DD format
        ticker (str): Stock ticker symbol
        
    Returns:
        List[Dict]: List of article dictionaries
    """
    # Convert string dates to datetime if needed
    if isinstance(start_date, str):
        start_date = datetime.strptime(start_date, "%Y-%m-%d")
    if isinstance(end_date, str):
        end_date = datetime.strptime(end_date, "%Y-%m-%d")
    
    # Format dates for search
    start_date_str = start_date.strftime("%B %d, %Y")
    end_date_str = end_date.strftime("%B %d, %Y")
    
    # Create search queries
    search_queries = [
        f"{company} stock news {start_date_str}",
        f"{ticker} stock news {start_date_str}",
        f"{company} stock news {end_date_str}",
        f"{ticker} stock news {end_date_str}",
        f"{company} financial news {start_date_str}",
        f"{ticker} financial news {start_date_str}"
    ]
    
    data = []
    seen_urls = set()
    
    for query in search_queries:
        print(f"\nSearching for: {query}")
        try:
            # Search Google
            search_results = search(query, num_results=20, lang="en")
            
            for url in search_results:
                if url in seen_urls or not is_credible_source(url):
                    continue
                    
                seen_urls.add(url)
                print(f"Found article: {url}")
                
                # Fetch article content
                content = fetch_article_content(url)
                if not content or len(content.split()) < 50:
                    continue
                    
                # Extract publication date
                pub_date = extract_publication_date(content)
                if not pub_date or not (start_date <= pub_date <= end_date):
                    continue
                
                # Get sentiment and confidence
                sentiment, confidence = get_sentiment(content)
                
                data.append({
                    "title": url.split('/')[-1].replace('-', ' ').title(),
                    "body": content,
                    "publication_time": pub_date.isoformat(),
                    "source": urlparse(url).netloc,
                    "url": url,
                    "ticker": ticker,
                    "sentiment": sentiment,
                    "sentiment_confidence": confidence
                })
                print(f"Successfully added article from {url}")
                
            # Be nice to the servers
            time.sleep(2)
            
        except Exception as e:
            print(f"Error searching for {query}: {str(e)}")
            continue
            
    return data

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Fetch news articles for specific dates')
    parser.add_argument('--start_date', type=str, required=True, help='Start date in YYYY-MM-DD format')
    parser.add_argument('--end_date', type=str, required=True, help='End date in YYYY-MM-DD format')
    parser.add_argument('--output_dir', type=str, default='data/scraped_data', help='Output directory for the CSV file')
    args = parser.parse_args()
    
    # Validate dates
    try:
        start_date = datetime.strptime(args.start_date, "%Y-%m-%d")
        end_date = datetime.strptime(args.end_date, "%Y-%m-%d")
        if end_date < start_date:
            raise ValueError("End date must be after start date")
    except ValueError as e:
        print(f"Error: Invalid date format. Please use YYYY-MM-DD format. {str(e)}")
        sys.exit(1)
    
    all_articles = []
    for ticker, company_name in STOCK_MAPPING.items():
        print(f"Fetching news for {company_name} ({ticker}) from {args.start_date} to {args.end_date}...")
        articles = fetch_news(company_name, start_date, end_date, ticker)
        if articles:
            print(f"Found {len(articles)} articles for {company_name}")
            articles = deduplicate_articles(articles)
            print(f"After deduplication: {len(articles)} articles")
            all_articles.extend(articles)
        else:
            print(f"No articles found for {company_name}")
    
    if not all_articles:
        print("No articles found for any company in the specified date range.")
        sys.exit(1)
        
    # For each day, sample 10 non-neutral articles per company
    final_data = []
    for ticker in STOCK_MAPPING.keys():
        company_df = pd.DataFrame([a for a in all_articles if a["ticker"] == ticker])
        if not company_df.empty:
            print(f"\nProcessing {ticker}:")
            for date in company_df["publication_time"].str[:10].unique():
                sampled = filter_and_sample(company_df.to_dict("records"), date, n=10)
                print(f"Date {date}: {len(sampled)} articles after sentiment filtering")
                final_data.extend(sampled)
        else:
            print(f"No articles found for {ticker}")
    
    if not final_data:
        print("No articles remained after sentiment filtering.")
        sys.exit(1)
            
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Save with date range in filename
    output_file = os.path.join(args.output_dir, f"filtered_news_{args.start_date}_to_{args.end_date}.csv")
    save_articles_to_csv(final_data, output_file)
    print(f"\nSaved {len(final_data)} articles to {output_file}")
