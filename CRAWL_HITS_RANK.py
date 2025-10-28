
import scrapy
from scrapy.crawler import CrawlerProcess
from scrapy.linkextractors import LinkExtractor
from urllib.parse import urlparse
import numpy as np


class WebCrawlerSpider(scrapy.Spider):
    name = 'web_crawler'
    
    custom_settings = {
        'ROBOTSTXT_OBEY': True,
        'CONCURRENT_REQUESTS': 1,
        'DOWNLOAD_DELAY': 1,
        'DEPTH_LIMIT': 3,
        'LOG_LEVEL': 'INFO',
    }
    
    adjacency_list = {}  
    
    def __init__(self, start_url, max_pages=10, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.start_urls = [start_url]
        self.allowed_domains = [urlparse(start_url).netloc]
        self.max_pages = max_pages
        self.pages_crawled = 0
        self.link_extractor = LinkExtractor(allow_domains=self.allowed_domains, unique=True)
    
    def parse(self, response):
        if self.pages_crawled >= self.max_pages:
            return
        
        self.pages_crawled += 1
        current_url = response.url
        
        # title = response.css('title::text').get()
        self.logger.info(f"  [{self.pages_crawled}/{self.max_pages}]: {current_url}")
        
        links = self.link_extractor.extract_links(response)
        outlinks = [link.url for link in links]
        
        WebCrawlerSpider.adjacency_list[current_url] = outlinks
        
        if self.pages_crawled < self.max_pages:
            for link in links:
                yield response.follow(link.url, callback=self.parse)
    
    def closed(self, reason):
        self.logger.info(f"\n Crawled {self.pages_crawled} pages")

class GraphRanking:
    def __init__(self, adjacency_list):
        self.adjacency_list = adjacency_list
        self.nodes = list(adjacency_list.keys())
        self.node_index = {node: i for i, node in enumerate(self.nodes)}
        self.n = len(self.nodes)
    
    def pagerank(self, damping=0.85, max_iter=100, tol=1e-6):
        pr = np.ones(self.n) / self.n
        
        for _ in range(max_iter):
            pr_new = np.ones(self.n) * (1 - damping) / self.n
            # PR(p) = (1 - d)/N + d*sum(PR(q)/L(q)) -> L(q) -> is the number of outgoing links for a page q,q is the page incoming to p
            for i, node in enumerate(self.nodes):
                for source in self.nodes:
                    if node in self.adjacency_list[source]:
                        j = self.node_index[source]
                        outlinks = len(self.adjacency_list[source])
                        if outlinks > 0:
                            pr_new[i] += damping * pr[j] / outlinks
            
            if np.linalg.norm(pr_new - pr) < tol:
                break
            pr = pr_new
        
        results = [(self.nodes[i], pr[i]) for i in range(self.n)]
        return sorted(results, key=lambda x: x[1], reverse=True)
    
    # HITS –> Hyperlink-Induced Topic Search
    def hits(self, max_iter=100, tol=1e-6):
        hub = np.ones(self.n)
        auth = np.ones(self.n)
        
        for _ in range(max_iter):
            auth_new = np.zeros(self.n)
            for i, node in enumerate(self.nodes):
                for source in self.nodes:
                    if node in self.adjacency_list[source]:
                        j = self.node_index[source]
                        auth_new[i] += hub[j]
            
            hub_new = np.zeros(self.n)
            for i, node in enumerate(self.nodes):
                for target in self.adjacency_list[node]:
                    if target in self.node_index:
                        j = self.node_index[target]
                        hub_new[i] += auth_new[j]
            
            auth_new = auth_new / (np.linalg.norm(auth_new) + 1e-10)
            hub_new = hub_new / (np.linalg.norm(hub_new) + 1e-10)
            
            if np.linalg.norm(auth_new - auth) < tol and np.linalg.norm(hub_new - hub) < tol:
                break
            
            auth = auth_new
            hub = hub_new
        
        auth_results = [(self.nodes[i], auth[i]) for i in range(self.n)]
        hub_results = [(self.nodes[i], hub[i]) for i in range(self.n)]
        
        return {
            'authority': sorted(auth_results, key=lambda x: x[1], reverse=True),
            'hub': sorted(hub_results, key=lambda x: x[1], reverse=True)
        }
    
    def display_results(self):
        print("\n" + "="*70)
        print(" PAGERANK RESULTS")
        print("="*70)
        pr_results = self.pagerank()
        for i, (url, score) in enumerate(pr_results, 1):
            print(f"{i}. {url}")
            print(f"   Score: {score:.6f}\n")
        
        print("="*70)
        print(" HITS RESULTS")
        print("="*70)
        hits_results = self.hits()
        
        print("\n TOP AUTHORITIES (Best content sources):")
        for i, (url, score) in enumerate(hits_results['authority'][:5], 1):
            print(f"{i}. {url}")
            print(f"   Authority: {score:.6f}\n")
        
        print(" TOP HUBS (Best link aggregators):")
        for i, (url, score) in enumerate(hits_results['hub'][:5], 1):
            print(f"{i}. {url}")
            print(f"   Hub: {score:.6f}\n")


# ==================== MAIN EXECUTION ====================

if __name__ == "__main__":
    # Step 1: Crawl the website
    print("Starting web crawler...")
    
    process = CrawlerProcess({'USER_AGENT': 'Mozilla/5.0 (Educational Crawler)'})
    process.crawl(WebCrawlerSpider, start_url='https://www.python.org', max_pages=5)
    process.start()
    
    # Step 2: Analyze with PageRank and HITS
    if WebCrawlerSpider.adjacency_list:
        print("\n\nAnalyzing graph structure...")
        ranker = GraphRanking(WebCrawlerSpider.adjacency_list)
        ranker.display_results()
    else:
        print("No pages crawled. Check the URL or increase max_pages.")




