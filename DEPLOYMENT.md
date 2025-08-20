# Deploy ATS Resume Matcher to Render

This guide will help you deploy your ATS Resume Matcher web application to Render.

## 🚀 Quick Deployment Steps

### Method 1: Using Render Dashboard (Recommended)

1. **Push your code to GitHub** (if not already done):
   ```bash
   git add .
   git commit -m "Prepare for Render deployment"
   git push origin main
   ```

2. **Go to [Render Dashboard](https://dashboard.render.com/)**
   - Sign up or log in to your Render account
   - Connect your GitHub account if not already connected

3. **Create New Web Service**:
   - Click "New +" → "Web Service"
   - Connect your GitHub repository: `Shine-5705/Job-Matcher`
   - Select the repository and branch (`main`)

4. **Configure the Service**:
   - **Name**: `ats-resume-matcher`
   - **Environment**: `Python 3`
   - **Build Command**: `pip install -r requirements.txt && python build.py`
   - **Start Command**: `gunicorn --bind 0.0.0.0:$PORT app:app`
   - **Plan**: Free (for testing)

5. **Environment Variables**:
   - `FLASK_ENV`: `production`
   - `PYTHON_VERSION`: `3.10.12`
   - `SECRET_KEY`: (auto-generated)

6. **Deploy**:
   - Click "Create Web Service"
   - Wait for deployment to complete (5-10 minutes)

### Method 2: Using render.yaml (Infrastructure as Code)

If you prefer automated deployment, Render will automatically detect the `render.yaml` file in your repository and deploy based on that configuration.

## 📁 Deployment Files Created

- **`render.yaml`**: Render service configuration
- **`Procfile`**: Alternative process file
- **`build.py`**: Build script for NLTK data setup
- **`requirements.txt`**: Updated with production dependencies

## 🔧 Production Optimizations Applied

### Security
- Environment-based secret key management
- Production-safe Flask configuration
- Secure file upload handling

### Performance
- Gunicorn WSGI server for production
- Optimized NLTK data downloads
- Efficient error handling

### Reliability
- Health check endpoint
- Graceful error handling
- Robust file processing

## 🌐 Post-Deployment

After successful deployment, your ATS Resume Matcher will be available at:
`https://ats-resume-matcher.onrender.com`

### Features Available:
- ✅ Professional landing page
- ✅ Drag-and-drop file upload
- ✅ Real-time resume analysis
- ✅ Interactive demo with sample data
- ✅ Comprehensive results dashboard
- ✅ Export functionality

## 🐛 Troubleshooting

### Common Issues:

1. **Build Timeout**:
   - NLTK downloads might take time on first build
   - This is normal, subsequent builds will be faster

2. **Memory Issues**:
   - Free tier has 512MB RAM limit
   - Large PDF files might need processing optimization

3. **NLTK Data Issues**:
   - Build script handles automatic NLTK data download
   - SSL certificate issues are handled automatically

### Logs Access:
- Go to your Render dashboard
- Select your service
- Click "Logs" tab to see real-time deployment logs

## 🔄 Updates and Maintenance

To update your deployed application:
1. Push changes to your GitHub repository
2. Render will automatically rebuild and redeploy
3. No manual intervention needed

## 💡 Upgrade Options

### Paid Plans Benefits:
- **Starter ($7/month)**: No sleep, faster builds, 1GB RAM
- **Standard ($25/month)**: 2GB RAM, better performance
- **Pro ($85/month)**: 4GB RAM, priority support

## 🎯 Next Steps

1. **Test the deployment** with sample files
2. **Monitor performance** using Render dashboard
3. **Set up custom domain** (optional)
4. **Configure environment variables** for production
5. **Set up monitoring and alerts**

Your ATS Resume Matcher is now ready for production use! 🎉
