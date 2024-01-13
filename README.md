# Information Retrieval Systems - Final Project

### Arab International University - Semester 2023-1

#### By

* Mustafa Al Hassny - 202010474
* Moyasser Al Nassif - 202010376
* Nour Habra - 202010515

## Overview

This project implements an Information Retrieval System with a web crawler and a search engine. The system is designed to crawl web pages, index their content, and provide search functionalities using ranking algorithms such as TF-IDF, PageRank, and BM25.

## Components

### 1. Web Crawler

#### TitleCrawler Class

- **Initialization:** The `TitleCrawler` class is initialized with a starting URL and a maximum number of links to crawl.
- **Crawling:** The `crawl` method retrieves content from a web page, extracts the title and text, and updates the positional index.
- **Indexing:** The `update_positional_index` method extracts terms and synonyms from the title/content and updates the positional index.
- **Details Retrieval:** The `retrieve_crawler_details` method retrieves details about the last crawled page.
- **Start Crawl:** The `start` method initiates the crawling process and returns details about the crawled pages.

### 2. Search Engine

#### SearchEngine Class

- **Initialization:** The `SearchEngine` class is initialized with lists of documents and an empty positional index.
- **Indexing Documents:** The `index_documents` method processes documents and constructs the positional index.
- **TF-IDF Search:** The `tf_idf_search` method performs a TF-IDF search based on a given query.
- **BM25 Search:** The `bm25_search` method performs a BM25 search based on a given query.
- **PageRank Calculation:** The `calculate_page_rank` method calculates PageRank scores for documents.
- **PageRank Search:** The `page_rank_search` method performs a search using PageRank.

### 3. PageRankCalculator Class

- The `PageRankCalculator` class is used internally by the `SearchEngine` class to calculate PageRank scores.

### 4. GUI Interface

- The code includes a simple graphical user interface (GUI) implemented using the `customtkinter` library for interacting with the web crawler and search engine.

## Usage

1. **Web Crawling:**

   - Initialize the `TitleCrawler` with a starting URL and the maximum number of links.
   - Click the "Crawl" button to initiate the crawling process.
2. **Search Engine:**

   - Enter a search query in the designated text area.
   - Select the ranking algorithm (TF-IDF, PageRank, BM25) using the radio buttons.
   - Click the "Search" button to retrieve search results.

## Dependencies

- The code relies on external libraries, including `nltk`, `requests`, `bs4` (Beautiful Soup), `numpy`, `networkx`, and `customtkinter`.

## File Output

- The positional index is written to a file named `positional_index.txt`.

## Example

1. **Crawling:**

   - Provide a starting URL and the maximum number of links.
   - Click "Crawl" to retrieve details about the crawled pages.
2. **Searching:**

   - Enter a search query in the designated area.
   - Choose a ranking algorithm (TF-IDF, PageRank, BM25).
   - Click "Search" to see the results.
3. **Positional Index:**

   - The positional index is saved to the `positional_index.txt` file.

## Notes

- Ensure that the necessary libraries are installed before running the code.

