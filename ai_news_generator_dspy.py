import os
import dspy
import streamlit as st
from datetime import datetime
from typing import List, Dict, Optional, Any
import json
from dataclasses import dataclass
from enum import Enum
import asyncio
from concurrent.futures import ThreadPoolExecutor

# Import additional utilities
from dspy.utils import download
import requests
from urllib.parse import quote_plus

# Configuration
class Config:
    SERPER_API_KEY = os.environ.get("SERPER_API_KEY", "")
    OPENAI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
    MODEL = "gemini/gemini-2.5-flash"
    TEMPERATURE = 0.3
    MAX_TOKENS = 8192
    SEARCH_RESULTS_LIMIT = 10
    CACHE_DIR = ".dspy_cache"

# Data Models
@dataclass
class NewsArticle:
    title: str
    snippet: str
    link: str
    date: Optional[str] = None
    source: Optional[str] = None

class OutputFormat(Enum):
    BLOG_POST = "blog_post"
    SUMMARY = "summary"
    BULLET_POINTS = "bullet_points"

# Custom News Search Tool
class NewsSearchTool:
    def __init__(self, api_key: str, num_results: int = 10):
        self.api_key = api_key
        self.num_results = num_results
        self.base_url = "https://serpapi.com/search"
        
    def search(self, query: str, time_range: str = "qdr:w") -> List[NewsArticle]:
        """Search for news articles using Serper API."""
        if not self.api_key:
            # Fallback to mock data if no API key
            return self._mock_search(query)
        
        params = {
            "q": query,
            "tbm": "nws",  # News search
            "api_key": self.api_key,
            "num": self.num_results,
            "tbs": time_range  # Time range (qdr:d = day, qdr:w = week, qdr:m = month)
        }
        
        try:
            response = requests.get(self.base_url, params=params)
            response.raise_for_status()
            data = response.json()
            
            articles = []
            for item in data.get("news_results", [])[:self.num_results]:
                articles.append(NewsArticle(
                    title=item.get("title", ""),
                    snippet=item.get("snippet", ""),
                    link=item.get("link", ""),
                    date=item.get("date", ""),
                    source=item.get("source", "")
                ))
            return articles
        except Exception as e:
            st.error(f"Search API error: {e}")
            return self._mock_search(query)
    
    def _mock_search(self, query: str) -> List[NewsArticle]:
        """Return mock news data for demonstration."""
        mock_articles = [
            NewsArticle(
                title=f"Breaking: Major advancement in {query}",
                snippet=f"Researchers announce breakthrough in {query} technology that could revolutionize the industry...",
                link="https://example.com/article1",
                date="2 hours ago",
                source="Tech News Daily"
            ),
            NewsArticle(
                title=f"{query} startup raises $50M in Series B funding",
                snippet=f"Leading {query} company secures major investment to expand operations globally...",
                link="https://example.com/article2",
                date="1 day ago",
                source="Venture Beat"
            ),
            NewsArticle(
                title=f"How {query} is transforming business operations",
                snippet=f"Industry experts discuss the impact of {query} on modern enterprises...",
                link="https://example.com/article3",
                date="3 days ago",
                source="Business Insider"
            )
        ]
        return mock_articles[:self.num_results]

# DSPy Signatures
class GenerateSearchQueries(dspy.Signature):
    """Generate multiple search queries to comprehensively cover a news topic."""
    topic: str = dspy.InputField(desc="The main topic to search news about")
    time_range: str = dspy.InputField(desc="Time range for news (e.g., 'past week', 'past month')")
    search_queries: List[str] = dspy.OutputField(desc="List of 3-5 optimized search queries")

class AnalyzeNewsArticles(dspy.Signature):
    """Analyze and extract key information from news articles."""
    articles: str = dspy.InputField(desc="JSON string of news articles")
    topic: str = dspy.InputField(desc="The main topic being researched")
    
    key_developments: List[str] = dspy.OutputField(desc="List of key developments and findings")
    trends: List[str] = dspy.OutputField(desc="Identified trends and patterns")
    notable_quotes: List[str] = dspy.OutputField(desc="Important quotes or statements")
    summary: str = dspy.OutputField(desc="Executive summary of the news")

class GenerateBlogPost(dspy.Signature):
    """Generate an engaging blog post from analyzed news content."""
    topic: str = dspy.InputField(desc="The main topic")
    analysis: str = dspy.InputField(desc="Analyzed news content")
    tone: str = dspy.InputField(desc="Writing tone (professional, casual, technical)")
    
    title: str = dspy.OutputField(desc="Catchy blog post title")
    introduction: str = dspy.OutputField(desc="Engaging introduction paragraph")
    body: str = dspy.OutputField(desc="Main body content with sections")
    conclusion: str = dspy.OutputField(desc="Thought-provoking conclusion")

class GenerateSummary(dspy.Signature):
    """Generate a concise summary of news content."""
    topic: str = dspy.InputField(desc="The main topic")
    analysis: str = dspy.InputField(desc="Analyzed news content")
    
    summary: str = dspy.OutputField(desc="Concise summary (200-300 words)")
    key_takeaways: List[str] = dspy.OutputField(desc="3-5 key takeaways")

# DSPy Modules
class NewsSearcher(dspy.Module):
    """Module for intelligent news searching with query optimization."""
    
    def __init__(self, search_tool: NewsSearchTool):
        super().__init__()
        self.search_tool = search_tool
        self.query_generator = dspy.ChainOfThought(GenerateSearchQueries)
        
    def forward(self, topic: str, time_range: str = "past week") -> List[NewsArticle]:
        # Generate optimized search queries
        query_result = self.query_generator(topic=topic, time_range=time_range)
        
        # Execute searches in parallel
        all_articles = []
        seen_links = set()
        
        for query in query_result.search_queries:
            articles = self.search_tool.search(query)
            for article in articles:
                if article.link not in seen_links:
                    seen_links.add(article.link)
                    all_articles.append(article)
        
        return all_articles

class ContentAnalyzer(dspy.Module):
    """Module for analyzing and structuring news content."""
    
    def __init__(self):
        super().__init__()
        self.analyzer = dspy.ChainOfThought(AnalyzeNewsArticles)
        
    def forward(self, articles: List[NewsArticle], topic: str) -> dspy.Prediction:
        # Convert articles to JSON string for analysis
        articles_data = [
            {
                "title": a.title,
                "snippet": a.snippet,
                "source": a.source,
                "date": a.date
            }
            for a in articles
        ]
        articles_json = json.dumps(articles_data, indent=2)
        
        return self.analyzer(articles=articles_json, topic=topic)

class BlogWriter(dspy.Module):
    """Module for generating blog posts from analyzed content."""
    
    def __init__(self):
        super().__init__()
        self.writer = dspy.ChainOfThought(GenerateBlogPost)
        
    def forward(self, topic: str, analysis: dspy.Prediction, tone: str = "professional") -> dspy.Prediction:
        # Prepare analysis data
        analysis_text = f"""
Key Developments:
{chr(10).join(f'• {dev}' for dev in analysis.key_developments)}

Trends:
{chr(10).join(f'• {trend}' for trend in analysis.trends)}

Notable Quotes:
{chr(10).join(f'• {quote}' for quote in analysis.notable_quotes)}

Summary:
{analysis.summary}
"""
        
        return self.writer(topic=topic, analysis=analysis_text, tone=tone)

class SummaryGenerator(dspy.Module):
    """Module for generating concise summaries."""
    
    def __init__(self):
        super().__init__()
        self.summarizer = dspy.ChainOfThought(GenerateSummary)
        
    def forward(self, topic: str, analysis: dspy.Prediction) -> dspy.Prediction:
        analysis_text = f"""
Key Developments: {', '.join(analysis.key_developments)}
Trends: {', '.join(analysis.trends)}
Summary: {analysis.summary}
"""
        return self.summarizer(topic=topic, analysis=analysis_text)

# Main AI News Generator
class AINewsGenerator(dspy.Module):
    """Main module orchestrating the news generation pipeline."""
    
    def __init__(self, search_tool: NewsSearchTool):
        super().__init__()
        self.searcher = NewsSearcher(search_tool)
        self.analyzer = ContentAnalyzer()
        self.blog_writer = BlogWriter()
        self.summary_generator = SummaryGenerator()
        
    def forward(self, topic: str, output_format: OutputFormat = OutputFormat.BLOG_POST, 
                time_range: str = "past week", tone: str = "professional") -> Dict[str, Any]:
        # Search for news
        articles = self.searcher(topic=topic, time_range=time_range)
        
        if not articles:
            return {
                "error": "No articles found for the given topic.",
                "articles": [],
                "content": None
            }
        
        # Analyze articles
        analysis = self.analyzer(articles=articles, topic=topic)
        
        # Generate output based on format
        content = None
        if output_format.value == OutputFormat.BLOG_POST.value:
            blog = self.blog_writer(topic=topic, analysis=analysis, tone=tone)
            content = f"""# {blog.title}

{blog.introduction}

{blog.body}

## Conclusion

{blog.conclusion}

---
*Sources: {len(articles)} articles analyzed*"""
        elif output_format.value == OutputFormat.SUMMARY.value:
            summary = self.summary_generator(topic=topic, analysis=analysis)
            content = f"""## Summary: {topic}

{summary.summary}

### Key Takeaways:
{chr(10).join(f'{i+1}. {takeaway}' for i, takeaway in enumerate(summary.key_takeaways))}

---
*Based on {len(articles)} news articles*"""
        elif output_format.value == OutputFormat.BULLET_POINTS.value:
            content = f"""## {topic} - News Highlights

### Key Developments:
{chr(10).join(f'• {dev}' for dev in analysis.key_developments)}

### Trends:
{chr(10).join(f'• {trend}' for trend in analysis.trends)}

### Notable Quotes:
{chr(10).join(f'• {quote}' for quote in analysis.notable_quotes)}

---
*Analyzed {len(articles)} recent articles*"""
        
        return {
            "content": content,
            "articles": articles,
            "analysis": analysis,
            "metadata": {
                "topic": topic,
                "articles_count": len(articles),
                "time_range": time_range,
                "generated_at": datetime.now().isoformat()
            }
        }

# Optimization and Persistence
class OptimizedNewsGenerator:
    """Wrapper for optimized news generation with save/load capabilities."""
    
    def __init__(self, generator: AINewsGenerator):
        self.generator = generator
        self.is_optimized = False
        
    def optimize(self, training_examples: List[dspy.Example], metric=None):
        """Optimize the generator using DSPy's MIPROv2."""
        if not training_examples:
            st.warning("No training examples provided for optimization.")
            return
            
        # Default metric: length and quality of generated content
        if metric is None:
            def default_metric(example, pred, trace=None):
                if pred.get("error"):
                    return 0.0
                content = pred.get("content", "")
                # Simple metric based on content length and structure
                score = min(len(content) / 1000, 1.0)  # Normalize by expected length
                if "##" in content:  # Has sections
                    score += 0.2
                if "Key" in content:  # Has key points
                    score += 0.2
                return min(score, 1.0)
            metric = default_metric
        
        # Use MIPROv2 optimizer
        optimizer = dspy.MIPROv2(
            metric=metric,
            num_candidates=5,
            init_temperature=0.7
        )
        
        self.generator = optimizer.compile(
            self.generator,
            trainset=training_examples
        )
        self.is_optimized = True
        
    def save(self, filepath: str):
        """Save the optimized generator."""
        self.generator.save(filepath)
        
    def load(self, filepath: str):
        """Load an optimized generator."""
        self.generator.load(filepath)
        self.is_optimized = True

# Streamlit UI
def main():
    st.set_page_config(
        page_title="AI News Generator (DSPy)",
        page_icon="📰",
        layout="wide"
    )
    
    # Configure DSPy only once at module level
    if 'dspy_configured' not in st.session_state:
        if Config.OPENAI_API_KEY:
            lm = dspy.LM(
                model=Config.MODEL,
                temperature=Config.TEMPERATURE,
                max_tokens=Config.MAX_TOKENS,
                cache=True
            )
            dspy.configure(lm=lm)
            st.session_state.dspy_configured = True
        else:
            st.error("Please set OPENAI_API_KEY environment variable")
            return
    
    # Initialize components
    if 'generator' not in st.session_state:
        search_tool = NewsSearchTool(
            api_key=Config.SERPER_API_KEY,
            num_results=Config.SEARCH_RESULTS_LIMIT
        )
        generator = AINewsGenerator(search_tool)
        st.session_state.generator = OptimizedNewsGenerator(generator)
    
    # Header
    st.title("🤖 AI News Generator powered by DSPy")
    st.markdown("Generate comprehensive news content using advanced language model programming")
    
    # Sidebar
    with st.sidebar:
        st.header("Configuration")
        
        # Model info
        st.info(f"Model: {Config.MODEL}")
        if not Config.SERPER_API_KEY:
            st.warning("Using mock data (set SERPER_API_KEY for real news)")
        
        # Output format
        output_format = st.selectbox(
            "Output Format",
            options=[OutputFormat.BLOG_POST, OutputFormat.SUMMARY, OutputFormat.BULLET_POINTS],
            format_func=lambda x: x.value.replace("_", " ").title()
        )
        
        # Time range
        time_range = st.selectbox(
            "Time Range",
            options=["past day", "past week", "past month"],
            index=1
        )
        
        # Tone (for blog posts)
        tone = "professional"
        if output_format == OutputFormat.BLOG_POST:
            tone = st.selectbox(
                "Writing Tone",
                options=["professional", "casual", "technical"],
                index=0
            )
        
        # Advanced settings
        with st.expander("Advanced Settings"):
            temperature = st.slider("Temperature", 0.0, 1.0, Config.TEMPERATURE)
            max_results = st.slider("Max Search Results", 5, 20, Config.SEARCH_RESULTS_LIMIT)
            
            if temperature != Config.TEMPERATURE:
                st.info("Temperature changes require app restart to take effect")
        
        # Model management
        st.markdown("### Model Management")
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("Save Model", use_container_width=True):
                try:
                    st.session_state.generator.save("news_generator_model.json")
                    st.success("Model saved!")
                except Exception as e:
                    st.error(f"Error saving model: {e}")
        
        with col2:
            if st.button("Load Model", use_container_width=True):
                try:
                    st.session_state.generator.load("news_generator_model.json")
                    st.success("Model loaded!")
                except Exception as e:
                    st.error(f"Error loading model: {e}")
        
        # Optimization status
        if st.session_state.generator.is_optimized:
            st.success("✅ Using optimized model")
        else:
            st.info("ℹ️ Using base model")
    
    # Main content area
    topic = st.text_area(
        "Enter your topic",
        height=100,
        placeholder="Enter the topic you want to generate news content about...",
        help="Be specific for better results (e.g., 'artificial intelligence in healthcare' rather than just 'AI')"
    )
    
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        generate_button = st.button(
            "Generate Content",
            type="primary",
            use_container_width=True,
            disabled=not topic
        )
    
    with col2:
        show_analysis = st.checkbox("Show Analysis", value=False)
    
    with col3:
        show_sources = st.checkbox("Show Sources", value=True)
    
    # Generate content
    if generate_button and topic:
        with st.spinner(f"Generating {output_format.value.replace('_', ' ')}..."):
            try:
                # Generate content
                result = st.session_state.generator.generator(
                    topic=topic,
                    output_format=output_format,
                    time_range=time_range,
                    tone=tone
                )
                
                # Display content
                if result.get("error"):
                    st.error(result["error"])
                else:
                    # Main content
                    st.markdown("### Generated Content")
                    st.markdown(result["content"])
                    
                    # Analysis details
                    if show_analysis and result.get("analysis"):
                        with st.expander("📊 Detailed Analysis", expanded=True):
                            analysis = result["analysis"]
                            
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                st.markdown("#### Key Developments")
                                for dev in analysis.key_developments:
                                    st.markdown(f"• {dev}")
                            
                            with col2:
                                st.markdown("#### Trends")
                                for trend in analysis.trends:
                                    st.markdown(f"• {trend}")
                            
                            if analysis.notable_quotes:
                                st.markdown("#### Notable Quotes")
                                for quote in analysis.notable_quotes:
                                    st.markdown(f"> {quote}")
                    
                    # Sources
                    if show_sources and result.get("articles"):
                        with st.expander(f"📰 Sources ({len(result['articles'])} articles)", expanded=False):
                            for i, article in enumerate(result["articles"], 1):
                                st.markdown(f"**{i}. {article.title}**")
                                st.markdown(f"*{article.source or 'Unknown source'}* - {article.date or 'Date unknown'}")
                                st.markdown(f"{article.snippet}")
                                st.markdown(f"[Read more]({article.link})")
                                st.markdown("---")
                    
                    # Metadata
                    with st.expander("ℹ️ Generation Metadata", expanded=False):
                        st.json(result.get("metadata", {}))
                    
                    # Download option
                    if result.get("content"):
                        st.download_button(
                            label="Download Content",
                            data=result["content"],
                            file_name=f"{topic.lower().replace(' ', '_')}_news.md",
                            mime="text/markdown"
                        )
                    
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
                st.exception(e)
    
    # Footer
    st.markdown("---")
    st.markdown("Built with DSPy - The framework for programming language models")

# Enable async support for Streamlit
if __name__ == "__main__":
    main()