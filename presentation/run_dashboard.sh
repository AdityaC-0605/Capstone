#!/bin/bash

# Sustainable Credit Risk AI Dashboard Runner
# Handles virtual environment setup automatically

echo "🌱 Sustainable Credit Risk AI - Dashboard Runner"
echo "============================================================"

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "🔧 Creating virtual environment..."
    python3 -m venv venv
    echo "✅ Virtual environment created"
fi

# Activate virtual environment
echo "🔄 Activating virtual environment..."
source venv/bin/activate

# Install required packages
echo "📦 Installing/updating packages..."
pip install streamlit plotly pandas numpy

echo "✅ Setup complete!"
echo ""
echo "🌟 Starting Sustainable Credit Risk AI Dashboard..."
echo "📱 Dashboard will open in your browser automatically"
echo "🔗 URL: http://localhost:8501"
echo ""
echo "💡 Tips:"
echo "   - Use the sidebar to navigate between sections"
echo "   - Try the Live Prediction Demo with different inputs"
echo "   - Explore the Sustainability Dashboard"
echo ""
echo "🛑 Press Ctrl+C to stop the dashboard"
echo ""

# Run the dashboard
streamlit run presentation/dashboard.py --server.port=8501 --server.address=localhost