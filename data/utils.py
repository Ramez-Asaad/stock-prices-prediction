"""
Utility functions for data scraping and processing.
"""
import requests
from bs4 import BeautifulSoup
import re
from urllib.parse import urlparse
from datetime import datetime
from typing import Optional, List, Dict, Any, Tuple
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import random
import nltk
from nltk.tokenize import sent_tokenize
import pandas as pd
import time
from fake_useragent import UserAgent

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

# Load finBERT for sentiment filtering
tokenizer = AutoTokenizer.from_pretrained("yiyanghkust/finbert-tone")
model = AutoModelForSequenceClassification.from_pretrained("yiyanghkust/finbert-tone")
sentiment_labels = ["negative", "neutral", "positive"]

# List of credible financial news sources
CREDIBLE_SOURCES = [
    "reuters.com",
    "bloomberg.com",
    "cnbc.com",
    "marketwatch.com",
    "investing.com",
    "seekingalpha.com",
    "fool.com",
    "wsj.com",
    "ft.com",
    "economist.com",
    "yahoo.com/finance",
    "forbes.com",
    "businessinsider.com",
    "barrons.com",
    "morningstar.com"
]

# Sites that require special handling
RESTRICTED_SITES = {
    "wsj.com": {
        "headers": {
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.5",
            "Connection": "keep-alive",
            "Upgrade-Insecure-Requests": "1",
            "Cache-Control": "max-age=0"
        },
        "cookies": {
            "wsjregion": "na,us",
            "wsjcountry": "us"
        }
    },
    "ft.com": {
        "headers": {
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.5",
            "Connection": "keep-alive",
            "Upgrade-Insecure-Requests": "1"
        }
    },
    "bloomberg.com": {
        "headers": {
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.5",
            "Connection": "keep-alive",
            "Upgrade-Insecure-Requests": "1"
        }
    }
}

def get_random_headers() -> Dict[str, str]:
    """
    Generate random headers for requests.
    
    Returns:
        Dict[str, str]: Headers dictionary
    """
    ua = UserAgent()
    return {
        'User-Agent': ua.random,
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
        'Accept-Language': 'en-US,en;q=0.5',
        'Accept-Encoding': 'gzip, deflate, br',
        'Connection': 'keep-alive',
        'Upgrade-Insecure-Requests': '1',
        'Cache-Control': 'max-age=0',
        'TE': 'Trailers',
    }

def get_site_specific_config(url: str) -> Tuple[Dict[str, str], Dict[str, str]]:
    """
    Get site-specific headers and cookies for a URL.
    
    Args:
        url (str): URL to get configuration for
        
    Returns:
        Tuple[Dict[str, str], Dict[str, str]]: Headers and cookies
    """
    domain = urlparse(url).netloc.lower()
    for site, config in RESTRICTED_SITES.items():
        if site in domain:
            return config.get('headers', {}), config.get('cookies', {})
    return get_random_headers(), {}

def clean_text(text: str) -> str:
    """
    Clean and normalize text content.
    
    Args:
        text (str): Input text
        
    Returns:
        str: Cleaned text
    """
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    # Remove special characters but keep basic punctuation
    text = re.sub(r'[^\w\s.,!?-]', '', text)
    # Fix multiple punctuation
    text = re.sub(r'([.,!?])\1+', r'\1', text)
    return text.strip()

def process_long_text(text: str, max_length: int = 512) -> List[str]:
    """
    Process long text by splitting into sentences and creating chunks.
    
    Args:
        text (str): Input text
        max_length (int): Maximum length for each chunk
        
    Returns:
        List[str]: List of text chunks
    """
    # Split into sentences
    sentences = sent_tokenize(text)
    chunks = []
    current_chunk = []
    current_length = 0
    
    for sentence in sentences:
        sentence = clean_text(sentence)
        sentence_length = len(tokenizer.encode(sentence))
        
        if current_length + sentence_length > max_length:
            if current_chunk:
                chunks.append(' '.join(current_chunk))
            current_chunk = [sentence]
            current_length = sentence_length
        else:
            current_chunk.append(sentence)
            current_length += sentence_length
    
    if current_chunk:
        chunks.append(' '.join(current_chunk))
    
    return chunks

def get_sentiment(text: str) -> Tuple[str, float]:
    """
    Get sentiment of text using FinBERT.
    
    Args:
        text (str): Input text to analyze
        
    Returns:
        Tuple[str, float]: Sentiment label and confidence score
    """
    # Process long text in chunks
    chunks = process_long_text(text)
    sentiments = []
    confidences = []
    
    for chunk in chunks:
        inputs = tokenizer(chunk, return_tensors="pt", truncation=True, max_length=512)
        with torch.no_grad():
            outputs = model(**inputs)
            probs = torch.softmax(outputs.logits, dim=1)
            label_idx = torch.argmax(probs)
            sentiments.append(sentiment_labels[label_idx])
            confidences.append(probs[0][label_idx].item())
    
    # Return the most confident sentiment
    if confidences:
        max_conf_idx = confidences.index(max(confidences))
        return sentiments[max_conf_idx], confidences[max_conf_idx]
    return "neutral", 0.0

def fetch_article_content(url: str, max_retries: int = 3) -> str:
    """
    Fetch and extract the main content from an article URL.
    
    Args:
        url (str): Article URL
        max_retries (int): Maximum number of retry attempts
        
    Returns:
        str: Article content
    """
    headers, cookies = get_site_specific_config(url)
    session = requests.Session()
    
    for attempt in range(max_retries):
        try:
            # Add a small delay between retries
            if attempt > 0:
                time.sleep(2 * attempt)
            
            response = session.get(
                url,
                headers=headers,
                cookies=cookies,
                timeout=15,
                allow_redirects=True
            )
            
            # Handle different response codes
            if response.status_code == 403:
                print(f"Access forbidden for {url}, trying with different headers...")
                headers = get_random_headers()
                continue
            elif response.status_code == 429:
                print(f"Rate limited for {url}, waiting before retry...")
                time.sleep(5 * (attempt + 1))
                continue
            elif response.status_code != 200:
                print(f"Error {response.status_code} for {url}")
                continue
                
            response.raise_for_status()
            
            # Try to detect if we're being redirected to a login page
            if any(x in response.url.lower() for x in ['login', 'signin', 'subscribe']):
                print(f"Redirected to login page for {url}")
                continue
            
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Remove unwanted elements
            for element in soup.find_all(['script', 'style', 'nav', 'header', 'footer', 'aside', 'form']):
                element.decompose()
                
            # Try different content selectors
            content = None
            selectors = [
                'article',
                'main',
                'div[class*="content"]',
                'div[class*="article"]',
                'div[class*="story"]',
                'div[class*="post"]',
                'div[class*="entry"]',
                'div[class*="body"]',
                'div[class*="text"]',
                'div[class*="article-body"]',
                'div[class*="story-body"]'
            ]
            
            for selector in selectors:
                content = soup.select_one(selector)
                if content:
                    break
                    
            if content:
                # Get all text paragraphs
                paragraphs = content.find_all('p')
                text = ' '.join([p.get_text().strip() for p in paragraphs])
                return clean_text(text)
                
            print(f"Could not find content for URL: {url}")
            return ""
            
        except requests.exceptions.RequestException as e:
            print(f"Error fetching content from {url} (attempt {attempt + 1}/{max_retries}): {str(e)}")
            if attempt == max_retries - 1:
                return ""
            continue
        except Exception as e:
            print(f"Unexpected error for {url}: {str(e)}")
            return ""
    
    return ""

def is_credible_source(url: str) -> bool:
    """
    Check if the URL is from a credible source.
    
    Args:
        url (str): URL to check
        
    Returns:
        bool: True if URL is from a credible source
    """
    domain = urlparse(url).netloc.lower()
    return any(source in domain for source in CREDIBLE_SOURCES)

def extract_publication_date(content: str) -> Optional[datetime]:
    """
    Extract publication date from article content.
    
    Args:
        content (str): Article content
        
    Returns:
        Optional[datetime]: Publication date if found, None otherwise
    """
    date_patterns = [
        r'\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]* \d{1,2},? \d{4}\b',
        r'\b\d{1,2} (?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]* \d{4}\b',
        r'\b\d{4}-\d{2}-\d{2}\b',  # YYYY-MM-DD
        r'\b\d{2}/\d{2}/\d{4}\b'   # MM/DD/YYYY
    ]
    
    for pattern in date_patterns:
        matches = re.findall(pattern, content)
        if matches:
            try:
                if '-' in matches[0]:
                    return datetime.strptime(matches[0], "%Y-%m-%d")
                elif '/' in matches[0]:
                    return datetime.strptime(matches[0], "%m/%d/%Y")
                else:
                    return datetime.strptime(matches[0], "%B %d, %Y")
            except ValueError:
                continue
    return None

def deduplicate_articles(articles: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Remove duplicate articles based on title and body.
    
    Args:
        articles (List[Dict]): List of article dictionaries
        
    Returns:
        List[Dict]: List of unique articles
    """
    seen = set()
    unique = []
    for art in articles:
        # Use a more robust deduplication key
        key = (art["title"].lower(), art["url"])
        if key not in seen:
            seen.add(key)
            unique.append(art)
    return unique

def filter_and_sample(articles: List[Dict[str, Any]], date: str, n: int = 10) -> List[Dict[str, Any]]:
    """
    Filter articles by date and sentiment, then sample n articles.
    
    Args:
        articles (List[Dict]): List of article dictionaries
        date (str): Date to filter by (YYYY-MM-DD)
        n (int): Number of articles to sample
        
    Returns:
        List[Dict]: Filtered and sampled articles
    """
    # Filter by date (same day)
    day_articles = [a for a in articles if a["publication_time"][:10] == date]
    
    # Get sentiment and confidence for each article
    for article in day_articles:
        sentiment, confidence = get_sentiment(article["body"])
        article["sentiment"] = sentiment
        article["sentiment_confidence"] = confidence
    
    # Remove neutral sentiment and low confidence articles
    filtered = [
        a for a in day_articles 
        if a["sentiment"] != "neutral" and a["sentiment_confidence"] > 0.6
    ]
    
    # Sort by confidence and take top n
    filtered.sort(key=lambda x: x["sentiment_confidence"], reverse=True)
    return filtered[:n]

def save_articles_to_csv(articles: List[Dict[str, Any]], output_file: str) -> None:
    """
    Save articles to a well-organized CSV file.
    
    Args:
        articles (List[Dict]): List of article dictionaries
        output_file (str): Path to output CSV file
    """
    df = pd.DataFrame(articles)
    
    # Reorder columns
    columns = [
        'ticker',
        'publication_time',
        'title',
        'body',
        'sentiment',
        'sentiment_confidence',
        'source',
        'url'
    ]
    df = df[columns]
    
    # Sort by ticker and publication time
    df = df.sort_values(['ticker', 'publication_time'])
    
    # Save to CSV with proper encoding
    df.to_csv(output_file, index=False, encoding='utf-8') 