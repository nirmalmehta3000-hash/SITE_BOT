# Gerryson Mehta AI Assistant

An intelligent AI-powered assistant built with Streamlit and OpenRouter, featuring multiple AI models and MySQL chat storage.

## Features

- 🤖 **Multiple AI Models**: Access to Claude, GPT-4, Llama, and Gemini via OpenRouter
- 💬 **Interactive Chat**: Beautiful user interface with onboarding flow
- 🗄️ **Chat Storage**: All conversations saved to MySQL database
- 🚀 **Railway Ready**: Optimized for Railway deployment
- 🌍 **UTF-8 Support**: Full international character support

## Tech Stack

- **Frontend**: Streamlit
- **AI Models**: OpenRouter (Claude, GPT-4, Llama, Gemini)
- **Database**: MySQL
- **Deployment**: Railway
- **Language**: Python 3.11+

## Quick Start

1. Clone the repository
2. Install dependencies: `pip install -r requirements.txt`
3. Set environment variables (see Railway deployment)
4. Run locally: `streamlit run app.py`

## Railway Deployment

1. Connect your GitHub repository to Railway
2. Add MySQL database service
3. Set environment variables:
   ```
   OPENROUTER_API_KEY=your-key-here
   PYTHONIOENCODING=utf-8
   LANG=C.UTF-8
   LC_ALL=C.UTF-8
   ```
4. Deploy!

## Environment Variables

- `OPENROUTER_API_KEY`: Your OpenRouter API key
- `MYSQLHOST`, `MYSQLPORT`, `MYSQLDATABASE`, `MYSQLUSER`, `MYSQLPASSWORD`: MySQL credentials (auto-provided by Railway)

## License

Private project - Gerryson Mehta
