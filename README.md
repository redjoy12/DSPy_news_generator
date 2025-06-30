# AI News Generator with DSPy

A powerful news content generation system built using DSPy - the framework for programming (not prompting) language models. This implementation improves upon traditional prompt-based systems by using DSPy's modular, optimizable approach.

## 🚀 Key Improvements Over Original Implementation

### 1. **Modular Architecture with DSPy**
- Replaced LangGraph's StateGraph with DSPy's composable modules
- Each component (search, analysis, writing) is a self-contained, reusable module
- Clear separation of concerns with typed signatures

### 2. **Automatic Prompt Optimization**
- Uses DSPy's MIPROv2 optimizer to automatically improve prompts
- No manual prompt engineering required
- System learns from examples to optimize performance

### 3. **Better Error Handling**
- Robust error handling with fallback mechanisms
- Mock data support for development/testing
- Graceful degradation when APIs are unavailable

### 4. **Enhanced Features**
- Multiple output formats (blog post, summary, bullet points)
- Configurable writing tones
- Source tracking and citations
- Parallel search execution for better performance
- Model save/load functionality

### 5. **Production-Ready**
- Built-in caching for API efficiency
- Streamlit UI with advanced controls
- Environment-based configuration
- Metadata tracking for all generations

## 📋 Prerequisites

- Python 3.9+
- OpenAI API key
- Serper API key (optional, for real news - fallback to mock data available)

## 🛠️ Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd ai-news-generator-dspy
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up environment variables:
```bash
export OPENAI_API_KEY="your-openai-api-key"
export SERPER_API_KEY="your-serper-api-key"  # Optional
```

## 🚀 Usage

### Basic Usage

Run the Streamlit app:
```bash
streamlit run ai_news_generator_dspy.py
```

### Programmatic Usage

```python
import dspy
from ai_news_generator_dspy import AINewsGenerator, NewsSearchTool, OutputFormat

# Configure DSPy
lm = dspy.LM("openai/gpt-4o-mini", temperature=0.7)
dspy.configure(lm=lm)

# Initialize generator
search_tool = NewsSearchTool(api_key="your-serper-key", num_results=10)
generator = AINewsGenerator(search_tool)

# Generate content
result = generator(
    topic="artificial intelligence in healthcare",
    output_format=OutputFormat.BLOG_POST,
    time_range="past week",
    tone="professional"
)

print(result["content"])
```

## 📊 Architecture

```
AINewsGenerator (Main Orchestrator)
├── NewsSearcher (Query Optimization + Search)
│   ├── GenerateSearchQueries (DSPy Signature)
│   └── NewsSearchTool (API Integration)
├── ContentAnalyzer (Information Extraction)
│   └── AnalyzeNewsArticles (DSPy Signature)
├── BlogWriter (Content Generation)
│   └── GenerateBlogPost (DSPy Signature)
└── SummaryGenerator (Alternative Output)
    └── GenerateSummary (DSPy Signature)
```

## 🔧 Configuration Options

### Output Formats
- **Blog Post**: Full-length article with introduction, body, and conclusion
- **Summary**: Concise overview with key takeaways
- **Bullet Points**: Structured highlights of key developments

### Time Ranges
- Past day
- Past week
- Past month

### Writing Tones (for blog posts)
- Professional
- Casual
- Technical

## 🎯 Optimization

The system supports automatic optimization using DSPy's MIPROv2:

```python
from ai_news_generator_dspy import OptimizedNewsGenerator
import dspy

# Create training examples
training_examples = [
    dspy.Example(
        topic="quantum computing breakthroughs",
        output_format=OutputFormat.BLOG_POST,
        expected_quality=0.9
    ).with_inputs("topic", "output_format")
    # Add more examples...
]

# Optimize
opt_generator = OptimizedNewsGenerator(generator)
opt_generator.optimize(training_examples)

# Save optimized model
opt_generator.save("optimized_news_model.json")
```

## 📈 Advanced Features

### 1. **Parallel Search Execution**
The system generates multiple optimized search queries and executes them in parallel for comprehensive coverage.

### 2. **Deduplication**
Automatically removes duplicate articles based on URL to ensure unique content.

### 3. **Structured Analysis**
Extracts:
- Key developments
- Trends and patterns
- Notable quotes
- Executive summaries

### 4. **Metadata Tracking**
Every generation includes:
- Topic
- Article count
- Time range
- Generation timestamp

### 5. **Caching**
Built-in caching reduces API calls and improves response times for repeated queries.

## 🔍 Monitoring and Debugging

### Enable DSPy History Inspection
```python
# After generation
dspy.inspect_history(n=5)  # View last 5 LM calls
```

### MLflow Integration (Optional)
```python
import mlflow
mlflow.dspy.autolog()  # Enable automatic tracking
```

## 🚧 Troubleshooting

### No Serper API Key
The system will automatically fall back to mock data for demonstration purposes.

### Rate Limiting
Adjust the temperature and max_tokens in the Config class to manage API usage.

### Memory Issues
For large-scale operations, consider:
- Reducing SEARCH_RESULTS_LIMIT
- Implementing batch processing
- Using async operations

## 🔮 Future Enhancements

1. **Multi-language support**
2. **Real-time streaming updates**
3. **Integration with more news sources**
4. **Advanced filtering and categorization**
5. **Collaborative editing features**
6. **Export to multiple formats (PDF, DOCX)**

## 📄 License

MIT License

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📞 Support

For issues or questions, please create an issue in the repository.

---

Built with ❤️ using [DSPy](https://dspy.ai) - The framework for programming language models
