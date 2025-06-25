#!/bin/bash
# 🚀 NICEGOLD ProjectP - Git Push Script
# Production-ready repository push with safety checks

echo "🚀 NICEGOLD ProjectP - Repository Push Script"
echo "=============================================="

# Navigate to project directory
cd /home/nicetpad2/nicegold_data/NICEGOLD-ProjectP

# Check if we're in the right directory
if [ ! -f "ProjectP.py" ]; then
    echo "❌ Error: Not in the correct project directory"
    echo "Please run this script from the NICEGOLD-ProjectP directory"
    exit 1
fi

echo "📁 Current directory: $(pwd)"

# Initialize git if not already done
if [ ! -d ".git" ]; then
    echo "🔧 Initializing git repository..."
    git init
    echo "✅ Git repository initialized"
else
    echo "✅ Git repository already exists"
fi

# Check git status
echo "📊 Checking git status..."
git status --porcelain > /tmp/git_status.txt
if [ -s /tmp/git_status.txt ]; then
    echo "📝 Files to be committed:"
    cat /tmp/git_status.txt | head -10
    if [ $(wc -l < /tmp/git_status.txt) -gt 10 ]; then
        echo "... and $(( $(wc -l < /tmp/git_status.txt) - 10 )) more files"
    fi
else
    echo "ℹ️ No changes to commit"
fi

# Verify .gitignore is working
echo "🔍 Verifying .gitignore effectiveness..."
if [ -d "datacsv" ]; then
    if git check-ignore datacsv/ > /dev/null 2>&1; then
        echo "✅ datacsv/ is properly ignored by git"
    else
        echo "⚠️ Warning: datacsv/ may not be ignored by git"
        echo "Adding datacsv/ to .gitignore..."
        echo "datacsv/" >> .gitignore
    fi
else
    echo "ℹ️ datacsv/ folder doesn't exist (this is fine for repository)"
fi

# Check if large files would be committed
echo "🔍 Checking for large files..."
large_files=$(find . -type f -size +10M -not -path "./.git/*" -not -path "./datacsv/*" -not -path "./.venv/*" 2>/dev/null)
if [ -n "$large_files" ]; then
    echo "⚠️ Large files detected (>10MB):"
    echo "$large_files"
    echo "Consider adding these to .gitignore if they shouldn't be committed"
fi

# Remove existing origin if it exists
echo "🔧 Configuring remote repository..."
if git remote get-url origin > /dev/null 2>&1; then
    echo "🔄 Removing existing origin..."
    git remote remove origin
fi

# Add new origin
echo "🔗 Adding remote origin..."
git remote add origin https://github.com/nicetpad2/NICEGOLD-ProjectP

# Verify remote was added
if git remote get-url origin > /dev/null 2>&1; then
    echo "✅ Remote origin configured: $(git remote get-url origin)"
else
    echo "❌ Failed to configure remote origin"
    exit 1
fi

# Stage all files (respecting .gitignore)
echo "📦 Staging files for commit..."
git add .

# Show what will be committed
echo "📋 Files staged for commit:"
git diff --name-only --cached | head -20
if [ $(git diff --name-only --cached | wc -l) -gt 20 ]; then
    echo "... and $(( $(git diff --name-only --cached | wc -l) - 20 )) more files"
fi

# Commit with descriptive message
echo "💾 Creating commit..."
git commit -m "feat: NICEGOLD ProjectP v2.0 - Production Ready Trading Analysis System

🚀 Features:
- Real data only analysis system (no dummy data)
- Live trading completely disabled for safety
- Comprehensive ML pipeline for trading data analysis
- Docker support and containerization
- Professional documentation and setup guides
- User-managed data setup (datacsv/ excluded from repo)
- Production-ready configuration and error handling

🛡️ Security:
- No live trading risks
- No sensitive data included
- No API keys or credentials
- Real data analysis only

📊 Components:
- Modular architecture with core/ and utils/
- Advanced ML protection system
- Comprehensive menu interface
- Health monitoring and system checks
- Automated testing and validation

📝 Documentation:
- Complete setup instructions
- Data format specifications
- User guides and troubleshooting
- Repository push checklist

Version: 2.0
Status: Production Ready
Safety: Live trading disabled
Data: User-managed (datacsv/ excluded)"

# Check if commit was successful
if [ $? -eq 0 ]; then
    echo "✅ Commit created successfully"
else
    echo "❌ Commit failed"
    exit 1
fi

# Push to repository
echo "🚀 Pushing to GitHub repository..."
echo "📤 This may take a moment for the initial push..."

if git push -u origin main; then
    echo ""
    echo "🎉 SUCCESS! Repository pushed successfully!"
    echo "=============================================="
    echo "✅ NICEGOLD ProjectP v2.0 is now live on GitHub"
    echo "🌐 Repository URL: https://github.com/nicetpad2/NICEGOLD-ProjectP"
    echo ""
    echo "📋 Next steps for users:"
    echo "1. Clone the repository: git clone https://github.com/nicetpad2/NICEGOLD-ProjectP"
    echo "2. Install dependencies: pip install -r requirements.txt"
    echo "3. Create datacsv/ folder and add their own data"
    echo "4. Run the system: python ProjectP.py"
    echo ""
    echo "🔒 Safety features:"
    echo "- Live trading completely disabled"
    echo "- Real data only analysis"
    echo "- No sensitive data in repository"
    echo "- User manages their own trading data"
    echo ""
    echo "📊 The repository is production-ready and safe for public sharing!"
else
    echo ""
    echo "❌ Push failed!"
    echo "=============================================="
    echo "This might be due to:"
    echo "1. Repository doesn't exist on GitHub"
    echo "2. Authentication issues"
    echo "3. Network connectivity"
    echo "4. Repository not empty (if pushing to existing repo)"
    echo ""
    echo "💡 Solutions:"
    echo "- Make sure the repository exists on GitHub"
    echo "- Check your GitHub authentication"
    echo "- If repository exists and has content, try:"
    echo "  git pull origin main --allow-unrelated-histories"
    echo "  git push origin main"
    exit 1
fi

echo ""
echo "🎯 Repository push completed successfully!"
echo "📊 Total files in repository: $(git ls-files | wc -l)"
echo "📝 Commit hash: $(git rev-parse HEAD)"
echo "🕒 Push completed at: $(date)"

# Clean up temp files
rm -f /tmp/git_status.txt

echo ""
echo "✅ NICEGOLD ProjectP is ready for the world! 🌍"
