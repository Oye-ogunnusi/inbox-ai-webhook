# Email Triage API

A FastAPI-based service that uses OpenAI's GPT-4o-mini model to automatically generate professional email replies. This application provides a simple REST API endpoint that accepts email details and returns an AI-generated response.

## Overview

This application serves as an intelligent email assistant that can help automate or streamline email responses. It accepts incoming email content (subject and body) and leverages OpenAI's language model to craft appropriate, professional replies.

## Features

- **AI-Powered Responses**: Utilizes OpenAI's GPT-4o-mini model for generating contextually appropriate email replies
- **RESTful API**: Simple POST endpoint for easy integration with existing systems
- **Fast & Lightweight**: Built on FastAPI for high performance and minimal overhead
- **Professional Tone**: Configured to generate professional, business-appropriate responses

## Prerequisites

Before running this application, ensure you have the following installed:

- Python 3.8 or higher
- pip (Python package installer)
- An OpenAI API key (obtain from [OpenAI Platform](https://platform.openai.com/))

## Installation

1. **Clone or download the project files**

   Ensure you have the following files in your project directory:
   - `app.py`
   - `requirements.txt`

2. **Create a virtual environment** (recommended)

   ```bash
   python -m venv venv
   
   # On Windows
   venv\Scripts\activate
   
   # On macOS/Linux
   source venv/bin/activate
   ```

3. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

## Configuration

### Setting up the OpenAI API Key

The application requires an OpenAI API key to function. Set this as an environment variable:

**On Windows (Command Prompt):**
```cmd
set OPENAI_API_KEY=your_api_key_here
```

**On Windows (PowerShell):**
```powershell
$env:OPENAI_API_KEY="your_api_key_here"
```

**On macOS/Linux:**
```bash
export OPENAI_API_KEY=your_api_key_here
```

**For persistent configuration**, create a `.env` file in your project root:
```
OPENAI_API_KEY=your_api_key_here
```

> **Security Note**: Never commit your API key to version control. Add `.env` to your `.gitignore` file.

## Running the Application

Start the FastAPI server using uvicorn:

```bash
uvicorn app:app --reload
```

By default, the server will run on `http://127.0.0.1:8000`

The `--reload` flag enables auto-reload during development (remove in production).

### Alternative Running Options

**Specify host and port:**
```bash
uvicorn app:app --host 0.0.0.0 --port 8080
```

**Production mode (without reload):**
```bash
uvicorn app:app --host 0.0.0.0 --port 8000 --workers 4
```

## API Documentation

Once the server is running, FastAPI automatically generates interactive API documentation:

- **Swagger UI**: http://127.0.0.1:8000/docs
- **ReDoc**: http://127.0.0.1:8000/redoc

## Usage

### Endpoint: `/triage`

**Method:** `POST`

**Request Body:**
```json
{
  "subject": "Meeting Request",
  "body_text": "Hi, I would like to schedule a meeting to discuss the project timeline. Are you available next Tuesday?"
}
```

**Response:**
```json
{
  "reply_text": "Hi,\n\nThank you for reaching out. I would be happy to discuss the project timeline with you. Tuesday works well for me. Please let me know what time would be convenient for you, and I'll send over a calendar invite.\n\nLooking forward to our conversation.\n\nBest regards"
}
```

### Example Using cURL

```bash
curl -X POST "http://127.0.0.1:8000/triage" \
     -H "Content-Type: application/json" \
     -d '{
       "subject": "Question about pricing",
       "body_text": "Hello, I am interested in your enterprise plan. Could you provide more details about the pricing structure?"
     }'
```

### Example Using Python Requests

```python
import requests

url = "http://127.0.0.1:8000/triage"
payload = {
    "subject": "Thank you",
    "body_text": "Thank you for your help with the project yesterday. I really appreciate your support!"
}

response = requests.post(url, json=payload)
print(response.json())
```

### Example Using JavaScript (Fetch)

```javascript
fetch('http://127.0.0.1:8000/triage', {
  method: 'POST',
  headers: {
    'Content-Type': 'application/json',
  },
  body: JSON.stringify({
    subject: 'Follow-up',
    body_text: 'Just following up on my previous email. Have you had a chance to review the proposal?'
  })
})
.then(response => response.json())
.then(data => console.log(data));
```

## Architecture

### Application Structure

```
project/
│
├── app.py              # Main FastAPI application
├── requirements.txt    # Python dependencies
└── README.md          # This file
```

### Components

1. **FastAPI Framework**: Provides the web framework and automatic API documentation
2. **OpenAI Client**: Handles communication with OpenAI's API
3. **Async Endpoint**: Processes email triage requests asynchronously

### Request Flow

1. Client sends POST request to `/triage` with email subject and body
2. Application constructs a prompt for the AI model
3. Request is sent to OpenAI's GPT-4o-mini model
4. AI-generated reply is extracted from the response
5. Reply text is returned to the client in JSON format

## Dependencies

- **fastapi**: Modern web framework for building APIs
- **uvicorn**: ASGI server for running FastAPI applications
- **openai**: Official OpenAI Python client library

## Customization

### Modifying the AI Behavior

To change how the AI responds, you can modify the system message in `app.py`:

```python
{"role": "system", "content": "You are a helpful email assistant."}
```

Example customizations:
- `"You are a formal business email assistant."`
- `"You are a friendly customer support representative."`
- `"You are a technical support specialist."`

### Changing the Model

To use a different OpenAI model, modify the `model` parameter:

```python
response = client.chat.completions.create(
    model="gpt-4",  # or "gpt-3.5-turbo", etc.
    # ... rest of the code
)
```

### Adding Additional Features

Consider extending the application with:
- Email classification/categorization
- Priority detection
- Sentiment analysis
- Multiple language support
- Response templates
- Email history context

## Error Handling

The application currently provides basic error handling. For production use, consider adding:

- API key validation
- OpenAI API error handling
- Request validation
- Rate limiting
- Logging
- Response timeouts

## Troubleshooting

**Issue: "OpenAI API key not found"**
- Solution: Ensure the `OPENAI_API_KEY` environment variable is set correctly

**Issue: "Module not found" errors**
- Solution: Ensure all dependencies are installed: `pip install -r requirements.txt`

**Issue: Port already in use**
- Solution: Specify a different port: `uvicorn app:app --port 8001`

**Issue: Slow response times**
- Solution: OpenAI API calls can take a few seconds. This is normal. Consider implementing caching for common queries.

## Security Considerations

⚠️ **Important Security Notes:**

1. **Never expose your API key** in code or version control
2. **Use environment variables** for sensitive configuration
3. **Implement authentication** before deploying to production
4. **Add rate limiting** to prevent abuse
5. **Validate input** to prevent injection attacks
6. **Use HTTPS** in production environments

## License

[Specify your license here]

## Contributing

[Add contribution guidelines if applicable]

## Contact

[Add contact information if applicable]

## Acknowledgments

- Built with [FastAPI](https://fastapi.tiangolo.com/)
- Powered by [OpenAI](https://openai.com/)
