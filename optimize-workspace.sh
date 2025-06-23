#!/bin/bash

# Workspace Performance Optimization Script
echo "🚀 Optimizing workspace for better VS Code performance..."

# Create a clean logs directory if it doesn't exist
mkdir -p logs

# Move scattered log files to logs directory
echo "📁 Organizing log files..."
find . -maxdepth 1 -name "*.log" -exec mv {} logs/ \; 2>/dev/null || true

# Clean up Python cache files
echo "🧹 Cleaning Python cache files..."
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
find . -name "*.pyc" -delete 2>/dev/null || true
find . -name "*.pyo" -delete 2>/dev/null || true

# Clean up pytest cache
echo "🧪 Cleaning test cache..."
find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true

# Show workspace size before and after
echo "📊 Workspace optimization complete!"
echo ""
echo "💡 Performance Tips:"
echo "   1. Open only the 'projectp/' folder as your workspace for better performance"
echo "   2. Use the projectp-workspace.code-workspace file for focused development"
echo "   3. The main workspace now excludes cache and log files from indexing"
echo ""
echo "🔧 To use the optimized workspace:"
echo "   - Close current workspace"
echo "   - Open 'projectp-workspace.code-workspace' or"
echo "   - Open just the 'projectp/' folder directly"

# Create a quick start script for the projectp workspace
cat > start-projectp-workspace.sh << 'EOF'
#!/bin/bash
echo "🚀 Starting ProjectP focused workspace..."
code projectp-workspace.code-workspace
EOF

chmod +x start-projectp-workspace.sh

echo ""
echo "✅ Created 'start-projectp-workspace.sh' for easy workspace startup"
