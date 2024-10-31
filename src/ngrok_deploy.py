import streamlit as st
from allatonce import MedicAI
import os
from dotenv import load_dotenv
from pyngrok import ngrok
import subprocess
import sys
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def setup_ngrok():
    """Setup ngrok tunnel for Streamlit and save public URL to a text file"""
    try:
        ngrok_token = os.getenv('NGROK_AUTH_TOKEN')
        if not ngrok_token:
            raise ValueError("NGROK_AUTH_TOKEN not found in environment variables")
        
        ngrok.set_auth_token(ngrok_token)
        
        public_url = ngrok.connect(8501)
        logger.info(f"Ngrok tunnel established at: {public_url}")
        
        with open("public_url.txt", "w") as file:
            file.write(str(public_url))
        
        return public_url
    except Exception as e:
        logger.error(f"Error setting up ngrok: {str(e)}")
        raise

def setup_environment():
    """Setup all required environment variables and dependencies"""
    try:
        load_dotenv()
        
        required_vars = ["PINECONE_API_KEY", "NGROK_AUTH_TOKEN"]
        missing_vars = [var for var in required_vars if not os.getenv(var)]
        if missing_vars:
            raise ValueError(f"Missing required environment variables: {', '.join(missing_vars)}")
            
        required_packages = ["streamlit", "pyngrok", "python-dotenv"]
        for package in required_packages:
            try:
                __import__(package)
            except ImportError:
                subprocess.check_call([sys.executable, "-m", "pip", "install", package])
                
    except Exception as e:
        logger.error(f"Error setting up environment: {str(e)}")
        raise

def start_deployment():
    """Main function to start the deployment"""
    try:
        setup_environment()
        
        public_url = setup_ngrok()
        
        subprocess.run(["streamlit", "run", "chatuimedic.py"])
        
    except Exception as e:
        logger.error(f"Deployment failed: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    start_deployment()