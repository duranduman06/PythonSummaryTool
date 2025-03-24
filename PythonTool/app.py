from django.utils.lorem_ipsum import sentence
from flask import Flask, render_template, request
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem import PorterStemmer
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD
import networkx as nx
import wikipediaapi
import requests
import re
import time

app = Flask(__name__)

# Check and download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('punkt')
    nltk.download('stopwords')


def get_wikipedia_content(url):
    try:
        # Validate URL format
        if not re.match(r'https?://en\.wikipedia\.org/wiki/[^/]+', url):
            print("Invalid Wikipedia URL")
            return None, None

        # Extract title from URL
        title = url.split('/wiki/')[-1]
        title = requests.utils.unquote(title)

        # Fetch plain text content using Wikipedia API
        params = {
            'action': 'query',
            'format': 'json',
            'titles': title,
            'prop': 'extracts',
            'explaintext': True,
        }
        response = requests.get(
            'https://en.wikipedia.org/w/api.php',
            params=params,
            headers={'User-Agent': 'TextSummarizer/1.0 (duranduman06@gmail.com)'}
        )
        data = response.json()
        page_data = next(iter(data['query']['pages'].values()))

        if 'extract' not in page_data:
            print("Page not found")
            return None, None

        full_text = page_data['extract']
        page_title = page_data['title']

        # Split text into sections (headers are wrapped with '==')
        sections = re.split(r'\n(=+)\s*(.*?)\s*\1\n', full_text)

        # Structure: [lead, '==', 'Header1', content1, '===', 'Header2', content2, ...]
        cleaned_sections = []
        excluded_titles = {'References', 'External links', 'Further reading',
                           'Notes', 'Citations', 'Bibliography', 'Sources', 'See also', 'In literature'}

        # Add the lead (text before first header)
        if len(sections) > 0:
            cleaned_sections.append(sections[0].strip())

        skip = False
        for i in range(1, len(sections)):
            if i % 3 == 1:  # Header level part (e.g. '==')
                continue
            elif i % 3 == 2:  # Header title
                header_title = sections[i].strip()
                if header_title in excluded_titles:
                    skip = True
                else:
                    skip = False
            else:  # Content part
                if not skip:
                    cleaned_sections.append(sections[i].strip())

        # Combine sections into cleaned text
        cleaned_text = '\n\n'.join([sec for sec in cleaned_sections if sec])

        return page_title, cleaned_text

    except Exception as e:
        print(f"Error fetching Wikipedia content: {str(e)}")
        return None, None



def preprocess_text(text):
    # Remove ISBN/ISSN patterns
    text = re.sub(r'\b(?:ISBN|ISSN)[\s-]*(\d[\s-]*){9,13}\d\b', ' ', text)

    # Remove bracketed numbers like [5]
    text = re.sub(r'\[\d+\]', ' ', text)

    # Remove URLs, email addresses and special characters
    text = re.sub(r'http\S+|www\S+|[^\w\s.,!?]', ' ', text)

    # Remove extra whitespace and normalize text
    text = ' '.join(text.split())

    # Split text into sentences
    sentences = sent_tokenize(text)

    # Remove very short sentences (less than 3 words)
    sentences = [s for s in sentences if len(s.split()) > 2]

    # Get English stopwords
    stop_words = set(stopwords.words('english'))

    # Add custom stopwords
    custom_stops = {'etc', 'eg', 'ie', 'vs'}
    stop_words.update(custom_stops)

    # Initialize Porter Stemmer for word stemming
    ps = PorterStemmer()

    # Process each sentence
    processed_sentences = []
    for sentence in sentences:
        # Normalize numbers and dates
        sentence = re.sub(r'\d+', 'NUM', sentence)
        sentence = re.sub(r'\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\b', 'MONTH', sentence)

        # Handle common words
        sentence = re.sub(r"n't", " not", sentence)
        sentence = re.sub(r"'m", " am", sentence)
        sentence = re.sub(r"'re", " are", sentence)

        # Tokenize, convert to lowercase, and remove stopwords and non-alphanumeric (anything other than letters and nums) words
        words = word_tokenize(sentence.lower())
        words = [ps.stem(word) for word in words if word.isalnum() and word not in stop_words]

        if words:  # Only add non-empty sentences
            processed_sentences.append(' '.join(words))

    return sentences, processed_sentences




def get_tfidf_summary(text, num_sentences=3):
    # Get original and preprocessed sentences
    original_sentences, processed_sentences = preprocess_text(text)

    if len(original_sentences) <= num_sentences:
        return text

    # Create TF-IDF vectorizer and transform sentences
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(processed_sentences)

    # Calculate sentence scores based on TF-IDF values
    sentence_scores = []
    for i in range(len(processed_sentences)):
        score = np.mean(tfidf_matrix[i].toarray())
        sentence_scores.append((score, i))

    # Sort sentences by score and select top ones
    sentence_scores.sort(reverse=True)
    top_sentences = sorted([pair[1] for pair in sentence_scores[:num_sentences]])

    # Combine top sentences to create summary
    summary = ' '.join([original_sentences[i] for i in top_sentences])
    return summary




def get_lsa_summary(text, num_sentences=3):
    # Get original and preprocessed sentences
    original_sentences, processed_sentences = preprocess_text(text)

    if len(original_sentences) <= num_sentences:
        return text

    # Create TF-IDF vectorizer and transform sentences
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(processed_sentences)

    # Apply LSA using TruncatedSVD
    svd = TruncatedSVD(n_components=1)
    lsa_matrix = svd.fit_transform(tfidf_matrix)


    # Calculate sentence scores based on LSA values
    sentence_scores = []
    for i in range(len(processed_sentences)):
        score = lsa_matrix[i].sum()
        sentence_scores.append((score, i))


    # Sort sentences by score and select top ones
    sentence_scores.sort(reverse=True)
    top_sentences = sorted([pair[1] for pair in sentence_scores[:num_sentences]])

    # Combine top sentences to create summary
    summary = ' '.join([original_sentences[i] for i in top_sentences])
    return summary




def get_textrank_summary(text, num_sentences=3):
    # Get original and preprocessed sentences
    original_sentences, processed_sentences = preprocess_text(text)

    if len(original_sentences) <= num_sentences:
        return text

     # Create TF-IDF vectorizer and transform sentences
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(processed_sentences)

    # Create similarity matrix
    similarity_matrix = cosine_similarity(tfidf_matrix, tfidf_matrix)

    # Build graph
    similarity_graph = nx.from_numpy_array(similarity_matrix)

    # Apply TextRank algorithm
    scores = nx.pagerank(similarity_graph)

    # Rank sentences based on TextRank scores
    ranked_sentences = sorted(((score, idx) for idx, score in scores.items()), reverse=True)

    # Select top sentences
    top_sentences = sorted([ranked_sentences[i][1] for i in range(num_sentences)])

    # Combine top sentences to create summary
    summary = ' '.join([original_sentences[i] for i in top_sentences])
    return summary



def get_summary(text, num_sentences=3, method='tfidf'):
    start_time = time.time()

    if method == 'lsa':
        summary = get_lsa_summary(text, num_sentences)
    elif method == 'textrank':
        summary = get_textrank_summary(text, num_sentences)
    else:
        summary = get_tfidf_summary(text, num_sentences)

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Summary generation took {elapsed_time:.2f} seconds.")

    return summary




@app.route('/', methods=['GET', 'POST'])
def home():
    summary = None
    original_text = None
    url = None
    error = None
    title = None

    if request.method == 'POST':
        # Get form parameters
        num_sentences = int(request.form.get('num_sentences', 3))
        method = request.form.get('method', 'tfidf')

        # Handle URL input
        url = request.form.get('url', '').strip()
        if url:
            title, content = get_wikipedia_content(url)  # Get both title and content
            if content:
                original_text = content
                summary = get_summary(content, num_sentences, method)
            else:
                error = "Could not fetch content from the provided URL."
        # Handle direct text input
        else:
            text = request.form.get('text', '').strip()
            if text:
                original_text = text
                summary = get_summary(text, num_sentences, method)

    # Render template with results
    return render_template('index.html',
                         summary=summary,
                         original_text=original_text,
                         url=url,
                         error=error,
                         title=title,
                         active_tab=request.form.get('active_tab', 'text'))

# Run the Flask application in debug mode
if __name__ == '__main__':
    app.run(debug=True)