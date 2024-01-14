import nltk
# nltk.download('all')

from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import stopwords
import re

stop_words = set(stopwords.words('english'))
lemma=WordNetLemmatizer()
from nltk.tokenize import sent_tokenize, word_tokenize
def preprocsess(terms):
    document=list(terms)
    document = " ".join(terms)
    words = word_tokenize(document)
    words = [word.lower() for word in words]
    words= [word for word in words if not word in stop_words ]
    words = [word for word in words if re.match(r'^[a-zA-Z]{2,}$', word)]
    terms = [word.replace(",.;:\"'!?__()[]{}<>\n-"," ") for word in words]
    terms=[lemma.lemmatize(word) for word in terms]

    # Remove any empty strings from the terms list
    terms = [term for term in terms if term]

    unique_terms = []

    for term in terms:
        if term not in unique_terms:
            unique_terms.append(term)


    return set(unique_terms)
from nltk.corpus import wordnet
from urllib.parse import urlparse
import requests
from bs4 import BeautifulSoup

class TitleCrawler:
    def __init__(self, start_url=None, max_links=20):
        self.urls_to_be_visited = [start_url] if start_url is not None else []
        self.visited = []
        self.domain = None if start_url is None else "https://" + urlparse(start_url).netloc
        self.max_links = max_links
        self.link_count = 0
        self.positional_index = {}
        self.docs = []
        self.term = set()
        self.synonyms = set()

    def crawl(self, link):
        page_content = requests.get(link).text
        soup = BeautifulSoup(page_content, "html.parser")
        title = soup.find("title")
        content = soup.get_text()

        print("PAGE BEING CRAWLED:", title.text,'|','Page Number',self.link_count, "|", link)

        self.visited.append(link)
        self.update_positional_index(title.text, content, link)

        self.docs.append(content)

        urls = soup.find_all("a")
        for url in urls:
            url = url.get("href")
            if url is not None:
                if url.startswith(self.domain):
                    if url not in self.visited and url not in self.urls_to_be_visited:
                        self.urls_to_be_visited.append(url)

    def update_positional_index(self, title, content, link):
        # Extract terms and synonyms from the title/content
        self.terms = set(title.split()) | set(content.split())
        self.terms = preprocsess(self.terms)
        self.synonyms = set()
        for term in self.terms:
            for syn in wordnet.synsets(term):
                for lemma in syn.lemmas():
                    self.synonyms.add(lemma.name())

        # Update the positional index
        for term in self.terms | self.synonyms:
            if term not in self.positional_index:
                self.positional_index[term] = {}
            for position, word in enumerate(content.split()):
                if word == term:
                    if self.link_count in self.positional_index[term]:
                        self.positional_index[term][self.link_count].append(position)
                    else:
                        self.positional_index[term][self.link_count] = [position]

        self.link_count += 1
        if self.link_count >= self.max_links:
            return

        # Remove entries with empty dictionaries
        empty_keys = [key for key, value in self.positional_index.items() if not value]
        for key in empty_keys:
            del self.positional_index[key]

 


    def retrieve_crawler_details(self):
        # Retrieve the details of the last crawled page
        page_title = []
        page_link = []
        page_number = []

        if self.visited:
            last_crawled_link = self.visited[-1]
            last_crawled_index = self.link_count

            title = BeautifulSoup(requests.get(last_crawled_link).text, "html.parser").find("title")
            page_title = title.text
            page_link = last_crawled_link
            page_number = str(last_crawled_index)

        return page_title, page_link, page_number

    def get_page_title(self, url):
        page_content = requests.get(url).text
        soup = BeautifulSoup(page_content, "html.parser")
        title = soup.find("title")
        return title.text if title else ""

    def start(self):
        details=""
        while self.urls_to_be_visited and self.link_count < self.max_links:
            url = self.urls_to_be_visited.pop(0)
            self.crawl(url)
            page_title = self.get_page_title(url)
            page_number = self.link_count
            details += f"Title: {page_title}\n Link: {url}\n Number: {page_number}\n \n"

        # Print the positional index after all pages have been crawled
        # print("POSITIONAL INDEX:", self.positional_index)
        self.write_positional_index_to_file()
        return details

    def write_positional_index_to_file(self):
        with open('positional_index.txt', 'w') as f:
            for term, links in self.positional_index.items():
                f.write(f"{term}: {links}\n")

crawler = TitleCrawler(None,0)

import math
import networkx as nx
import numpy as np
class SearchEngine:
    def init(self):
        self.docs = []
        self.positional_index = {}
        self.page_rank_scores = []
        self.ad_maxtrix=[]

    def index_documents(self, documents):
        self.docs = documents

        # Clear the positional index
        self.positional_index = {}

        # Construct the positional index
        for doc_id, doc in enumerate(self.docs):
            doc_tokens = doc.lower().split()

            # Count term frequencies in the document
            term_freq = {}
            for term in doc_tokens:
                term_freq[term] = term_freq.get(term, 0) + 1

            # Update the positional index
            for term, freq in term_freq.items():
                if term in self.positional_index:
                    self.positional_index[term].append((doc_id, freq))
                else:
                    self.positional_index[term] = [(doc_id, freq)]

    def tf_idf_search(self, query):
        # Tokenize the query
        query_tokens = query.lower().split()

        # Calculate TF-IDF scores for documents containing the query terms
        doc_scores = {}
        for term in query_tokens:
            if term in self.positional_index:
                docs_with_term = self.positional_index[term]
                idf = math.log(len(self.docs) / len(docs_with_term))

                for doc_id, tf in docs_with_term:
                    tf_idf = tf * idf

                    if doc_id in doc_scores:
                        doc_scores[doc_id] += tf_idf
                    else:
                        doc_scores[doc_id] = tf_idf

        # Sort the documents based on TF-IDF scores
        sorted_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)

        # Generate the result string
        result_string = ""
        for doc_id, score in sorted_docs:
            result_string += f"Document {doc_id} link {crawler.visited[doc_id]} Score={score}\n"

        return result_string

    def bm25_search(self, query, k1=1.2, b=0.75):
        # Tokenize the query
        query_tokens = query.lower().split()

        # Calculate IDF for query terms
        query_idf = {}
        for term in query_tokens:
            if term in self.positional_index:
                query_idf[term] = math.log((len(self.docs) - len(self.positional_index[term]) + 0.5) / (len(self.positional_index[term]) + 0.5))

        # Calculate BM25 scores for documents containing the query terms
        doc_scores = {}
        avg_doc_length = sum(len(doc.split()) for doc in self.docs) / len(self.docs)
        for term in query_tokens:
            if term in self.positional_index:
                docs_with_term = self.positional_index[term]

                for doc_id, tf in docs_with_term:
                    doc_length = len(self.docs[doc_id].split())
                    K = k1 * ((1 - b) + b * (doc_length / avg_doc_length))
                    score = query_idf[term] * ((tf * (k1 + 1)) / (tf + K))

                    if doc_id in doc_scores:
                        doc_scores[doc_id] += score
                    else:
                        doc_scores[doc_id] = score

        # Sort the documents based on the BM25 scores
        sorted_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)

        # Generate the result string
        result_string = ""
        for doc_id, score in sorted_docs:
            result_string += f"Document {doc_id} link {crawler.visited[doc_id]}Score={score}\n"

        return result_string
    def calculate_page_rank(self, iteration_count):
        # Build the adjacency matrix
        self.adjacency_matrix = np.zeros((len(self.docs), len(self.docs)))

        for term, postings in self.positional_index.items():
            for (doc_id1, _) in postings:
                for (doc_id2, _) in postings:
                    if doc_id1 != doc_id2:
                        self.adjacency_matrix[doc_id1][doc_id2] = 1

        # Normalize the adjacency matrix
        out_degree = np.sum(self.adjacency_matrix, axis=1)
        self.adjacency_matrix = np.divide(self.adjacency_matrix, out_degree[:, np.newaxis], where=out_degree[:, np.newaxis] != 0)

        # Calculate the PageRank scores
        self.page_rank_scores = np.ones(len(self.docs)) / len(self.docs)

        for _ in range(iteration_count):
            self.page_rank_scores = np.dot(self.adjacency_matrix, self.page_rank_scores)

    def page_rank_search(self, query):
        # Tokenize the query
        query_tokens = query.lower().split()

        # Calculate scores for documents containing the query terms
        doc_scores = {}
        for term in query_tokens:
            if term in self.positional_index:
                for doc_id, _ in self.positional_index[term]:
                    if doc_id in doc_scores:
                        doc_scores[doc_id] += self.page_rank_scores[doc_id]
                    else:
                        doc_scores[doc_id] = self.page_rank_scores[doc_id]

        # Sort the documents based on the PageRank scores
        sorted_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)

        # Generate the result string
        result_string = ""
        for doc_id, score in sorted_docs:
            result_string += f"Document {doc_id} link {crawler.visited[doc_id]}\n Score={score}\n"

        return result_string

class PageRankCalculator:
    def __init__(self):
        self.graph = None
        self.node_count = 0
        self.page_rank = []

    def initialize(self, graph, node_count):
        self.graph = graph
        self.node_count = node_count
        self.page_rank = [1 / node_count] * node_count

    def get_page_rank_for_one_iteration(self):
        temp_page_rank = self.page_rank.copy()

        for in_link in range(self.node_count):
            page_rank = 0
            for out_link in range(self.node_count):
                if self.graph[out_link][in_link] == 1:
                    out_links_count = sum(1 for n in self.graph[out_link] if n == 1)
                    page_rank += 1 * temp_page_rank[out_link] / out_links_count
            self.page_rank[in_link] = page_rank

    def print_current_page_ranks(self):
        for i in range(self.node_count):
            print(f"node {i}: => {self.page_rank[i]}")

    def get_final_page_rank(self, iteration_count):
        print("initial page ranks:")
        self.print_current_page_ranks()
        for i in range(iteration_count):
            self.get_page_rank_for_one_iteration()
            print(f"Page rank after {i+1}/{iteration_count} iteration:")
            self.print_current_page_ranks()
            print()

# Create an instance of SearchEngine
search_engine = SearchEngine()
# Index the documents
documents =crawler.docs
search_engine.index_documents(documents)
##################################################################
# Create an instance of SearchEngine
search_engine = SearchEngine()
# Index the documents
documents =crawler.docs
search_engine.index_documents(documents)
# Calculate PageRank
search_engine.calculate_page_rank(10)
# Calculate the PageRank scores
search_engine.calculate_page_rank(iteration_count=50)
##################################################################
# GUI

import tkinter
import tkinter.messagebox
import customtkinter 

# url = tkinter.StringVar(value='')
# term = tkinter.StringVar(value='')

customtkinter.set_appearance_mode("System")  # Modes: "System" (standard), "Dark", "Light"
customtkinter.set_default_color_theme("blue")  # Themes: "blue" (standard), "green", "dark-blue"


app = customtkinter.CTk()

# configure window
app.title("IRS Final Project")
app.geometry(f"{1100}x{580}")

# configure grid layout 
app.grid_columnconfigure(0, weight=1)
app.grid_columnconfigure(1, weight=1)
        
        
def crawl_button_click():
    global crawler
    url = crawler_textbox.get()
    crawler = TitleCrawler(url, 10)
    details = crawler.start()
    crawlOutput.delete("0.0", "end")
    crawlOutput.insert("0.0", details)  # insert at line 0 character 0

def search_button_click():
    global crawler
    search_query = search_textbox.get()
    selected_ranking = selected_ranking_radio.get()
    
    if crawler is not None:
        documents = crawler.docs
        search_engine.index_documents(documents)
        if selected_ranking == 'TF-IDF':
            result = search_engine.tf_idf_search(search_query)
        elif selected_ranking == "PageRank":
            search_engine.calculate_page_rank(iteration_count=50)
            result = search_engine.page_rank_search(search_query)
        elif selected_ranking == "BM25":
            result = search_engine.bm25_search(search_query)
        else:
            result = 'Invalid ranking selection'
        searchOutput.delete("0.0", "end")
        searchOutput.insert("0.0", result) 
    else:
        searchOutput.delete("0.0", "end")
        searchOutput.insert("0.0", "Please crawl a website first") 

logo_label = customtkinter.CTkLabel(app, text="Crawler", font=customtkinter.CTkFont(size=20, weight="bold"))
logo_label.grid(row=0, column=0, padx=20, pady=(20, 10))
        
crawler_textbox = customtkinter.CTkEntry(app, placeholder_text="URL to crawl")
crawler_textbox.grid(row=1, column=0, columnspan=1, padx=(20, 20), pady=(0, 0), sticky="nsew")

button = customtkinter.CTkButton(app, text="Crawl", command=crawl_button_click)
button.grid(row=2, column=0, padx=20, pady=10, sticky="ew")

crawlOutput = customtkinter.CTkTextbox(app, width=250, height=300)
crawlOutput.grid(row=3, column=0, padx=(20, 0), pady=(20, 0), sticky="nsew")

# Right Section
logo_label = customtkinter.CTkLabel(app, text="Search", font=customtkinter.CTkFont(size=20, weight="bold"))
logo_label.grid(row=0, column=1, padx=20, pady=(20, 10))
        
search_textbox = customtkinter.CTkEntry(app, placeholder_text="Search term")
search_textbox.grid(row=1, column=1, columnspan=1, padx=(20, 20), pady=(0, 0), sticky="nsew")

button = customtkinter.CTkButton(app, text="Search", command=search_button_click)
button.grid(row=2, column=1, padx=20, pady=10, sticky="ew")

radiobutton_frame = customtkinter.CTkFrame(app)
radiobutton_frame.grid(row=4, column=1, padx=(20, 20), pady=(20, 0), sticky="n")
selected_ranking_radio = tkinter.StringVar(value='')

label_radio_group = customtkinter.CTkLabel(master=radiobutton_frame, text="Ranking Method:")
label_radio_group.grid(row=0, column=2, columnspan=1, padx=0, pady=10, sticky="n")
radio_button_1 = customtkinter.CTkRadioButton(master=radiobutton_frame, variable=selected_ranking_radio, value="TF-IDF", text="TF-IDF")
radio_button_1.grid(row=1, column=2, pady=5, padx=5, sticky="n")
radio_button_2 = customtkinter.CTkRadioButton(master=radiobutton_frame, variable=selected_ranking_radio, value="PageRank", text="PageRank")
radio_button_2.grid(row=1, column=3, pady=5, padx=5, sticky="n")
radio_button_3 = customtkinter.CTkRadioButton(master=radiobutton_frame, variable=selected_ranking_radio, value="BM25", text="BM25")
radio_button_3.grid(row=1, column=4, pady=5, padx=5, sticky="n")

searchOutput = customtkinter.CTkTextbox(app, width=250)
searchOutput.grid(row=3, column=1, padx=(20, 0), pady=(20, 0), sticky="nsew")

app.mainloop()