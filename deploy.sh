#!/bin/bash
# Render deployment script

echo "🚀 Deploying AgriPrice Prophet to Render..."

# Check if git is initialized
if [ ! -d .git ]; then
    git init
    git add .
    git commit -m "Initial commit for Render deployment"
fi

# Add remote if not exists
if ! git remote | grep -q render; then
    git remote add render https://github.com/YOUR_USERNAME/agriprice-prophet.git
fi

# Push to GitHub
git push -u origin main

echo "✅ Code pushed to GitHub"
echo ""
echo "📋 Next steps:"
echo "1. Go to https://render.com"
echo "2. Click 'New +' → 'Web Service'"
echo "3. Connect your GitHub repository"
echo "4. Use these settings:"
echo "   - Name: agriprice-prophet-api"
echo "   - Environment: Python"
echo "   - Build Command: pip install -r requirements.txt"
echo "   - Start Command: uvicorn app.api:app --host 0.0.0.0 --port 10000"
echo "5. Click 'Create Web Service'"
