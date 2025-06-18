# LaunchDarkly Flask API Project

This project is a Flask-based API that integrates with LaunchDarkly, OpenAI, and supports static content serving. It is ready for local development and deployment to AWS Elastic Beanstalk.

## Features
- RESTful API endpoints (e.g., `/context` for multi-context payloads)
- LaunchDarkly feature flag evaluation
- OpenAI integration
- CORS configuration for local and production use
- Static file serving (e.g., `index.html` and assets)
- Production-ready with Gunicorn and Elastic Beanstalk support

## Setup

1. **Clone the repository and navigate to the project directory.**
2. **Create and activate a virtual environment:**
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```
3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
4. **Copy the example environment file and fill in your secrets:**
   ```bash
   cp .env.example .env
   # Edit .env with your LaunchDarkly and OpenAI keys
   ```
5. **Run the app locally:**
   ```bash
   gunicorn app:app
   # Or for development:
   python app.py
   ```

## Deploying to AWS Elastic Beanstalk
1. Ensure `requirements.txt` and `Procfile` are present.
2. Use the provided `package.sh` to create a deployment zip:
   ```bash
   ./package.sh
   ```
3. Upload the zip to Elastic Beanstalk via the AWS Console.
4. Set environment variables in the AWS Console (do not upload `.env`).

## Static Content
- Place static files in the `static/` directory.
- The base route `/` serves `static/index.html`.
- Assets in `static/assets/` are served at `/assets/<filename>`.

## Environment Variables
See `.env.example` for required keys:
- `LD_SDK_KEY` (LaunchDarkly SDK key)
- `OPENAI_API_KEY` (OpenAI API key)

## Security
- Do **not** commit your `.env` file.
- Use environment variables for all secrets in production.

---

For more details, see comments in the code or ask for specific endpoint documentation.
