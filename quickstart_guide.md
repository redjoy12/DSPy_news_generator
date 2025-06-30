# AI News Generator - Quick Start Guide

## 🚀 Get Started in 5 Minutes

### 1. Clone and Setup

```bash
# Clone the repository
git clone <repository-url>
cd ai-news-generator-dspy

# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure API Keys

```bash
# Copy environment example
cp .env.example .env

# Edit .env and add your OpenAI API key (required)
# OPENAI_API_KEY=sk-...
```

### 3. Run the Application

```bash
# Start the Streamlit app
streamlit run ai_news_generator_dspy.py
```

The app will open in your browser at `http://localhost:8501`

## 🎯 First Run

1. **Enter a topic**: Try "artificial intelligence in healthcare"
2. **Select output format**: Start with "Blog Post"
3. **Click "Generate Content"**
4. **Review the results**: Check the generated content and sources

## 💡 Pro Tips

### Use Mock Data (No Serper API)
The app works without a Serper API key using mock data:
```python
# The system automatically falls back to mock data
# Perfect for testing and development
```

### Quick Test with Python
```python
import dspy
from ai_news_generator_dspy import AINewsGenerator, NewsSearchTool, OutputFormat

# Quick setup
dspy.configure(lm=dspy.LM("openai/gpt-4o-mini"))
search_tool = NewsSearchTool(api_key="", num_results=5)  # Empty key = mock data
generator = AINewsGenerator(search_tool)

# Generate content
result = generator(
    topic="quantum computing",
    output_format=OutputFormat.SUMMARY
)
print(result["content"])
```

### Optimize for Better Results
```bash
# Run the optimization example
python optimize_example.py
```

## 🔧 Common Issues

### "No module named 'dspy'"
```bash
pip install dspy-ai
```

### "OpenAI API key not found"
```bash
export OPENAI_API_KEY="your-key-here"
# Or add to .env file
```

### Rate Limiting
Reduce temperature in the sidebar: Advanced Settings → Temperature → 0.3

## 📚 Next Steps

1. **Try different output formats**: Summary, Bullet Points
2. **Experiment with time ranges**: Past day, week, month
3. **Save an optimized model**: Use the optimization script
4. **Customize the code**: Add your own modules and signatures

## 🆘 Need Help?

- Check the [full README](README.md)
- Review the [comparison guide](comparison.md)
- Explore [DSPy documentation](https://dspy.ai)

Happy news generating! 🎉
