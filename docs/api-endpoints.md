# API Endpoints Documentation

## Overview

The AI Assistant Backend provides RESTful API endpoints for interacting with the knowledge agent. The API is built with FastAPI and supports both regular and streaming responses for real-time interaction.

## Base Configuration

- **Framework**: FastAPI
- **Host**: localhost
- **Port**: 8081
- **Base URL**: `http://localhost:8081`

## Authentication

Currently, the API does not require authentication. CORS is configured to allow all origins for development purposes.

## Endpoints

### POST /ask/stream

Streams responses from the knowledge agent in real-time.

**URL**: `POST /ask/stream`

**Description**: Processes user messages through the knowledge agent and streams the response back as it's generated. This endpoint provides a real-time experience for users.

#### Request Format

```json
{
  "messages": [
    {
      "role": "human",
      "content": "What are the graduation requirements at Obuda University?"
    }
  ]
}
```

#### Request Schema

```python
class Message:
    role: str           # "human" or "ai"
    content: str        # Message content

class RequestModel:
    messages: List[Message]  # List of conversation messages
```

#### Response Format

**Content-Type**: `text/plain; charset=utf-8`  
**Transfer-Encoding**: `chunked`

The response is streamed as plain text chunks containing the AI's response as it's generated.

#### Example Usage

##### cURL
```bash
curl -X POST "http://localhost:8081/ask/stream" \
     -H "Content-Type: application/json" \
     -d '{
       "messages": [
         {
           "role": "human",
           "content": "What are the graduation requirements?"
         }
       ]
     }'
```

##### Python with requests
```python
import requests
import json

url = "http://localhost:8081/ask/stream"
payload = {
    "messages": [
        {
            "role": "human",
            "content": "What are the graduation requirements?"
        }
    ]
}

response = requests.post(url, json=payload, stream=True)
for chunk in response.iter_content(chunk_size=None):
    if chunk:
        print(chunk.decode('utf-8'), end='')
```

##### JavaScript (Fetch API)
```javascript
const response = await fetch('http://localhost:8081/ask/stream', {
    method: 'POST',
    headers: {
        'Content-Type': 'application/json',
    },
    body: JSON.stringify({
        messages: [
            {
                role: 'human',
                content: 'What are the graduation requirements?'
            }
        ]
    })
});

const reader = response.body.getReader();
while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    const chunk = new TextDecoder().decode(value);
    console.log(chunk);
}
```

#### Response Characteristics

- **Streaming**: Response is delivered in real-time chunks
- **Markdown**: Content is formatted with markdown for rich text display
- **Source Attribution**: Includes references to source documents
- **Multi-language**: Responds in the user's language

#### Error Handling

**400 Bad Request**
```json
{
  "detail": "Invalid request format"
}
```

**500 Internal Server Error**
```json
{
  "detail": "Internal processing error"
}
```

## Message Flow

### Conversation Context

The API supports multi-turn conversations by including previous messages in the request:

```json
{
  "messages": [
    {
      "role": "human", 
      "content": "What is Obuda University?"
    },
    {
      "role": "ai",
      "content": "Óbuda University is a public university in Budapest, Hungary..."
    },
    {
      "role": "human",
      "content": "What are the admission requirements?"
    }
  ]
}
```

### Processing Pipeline

1. **Message Extraction**: Latest message becomes the user query
2. **Context Building**: Previous messages form conversation history  
3. **Contextualization**: Query is contextualized using conversation history
4. **Knowledge Retrieval**: Internal database and web sources are searched
5. **Response Generation**: LLM generates response using retrieved context
6. **Streaming**: Response is streamed back in real-time

## Application Lifecycle

### Startup
```python
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Initialize database connection
    app.state.db_client = MongoDB()
    
    # Initialize knowledge agent
    app.state.knowledge_agent = Knowledge()
    
    yield
    
    # Cleanup on shutdown
    app.state.db_client.close()
```

### Components Initialized
- **MongoDB Client**: Database connection for document storage
- **Knowledge Agent**: Pre-configured agent for University of Obuda domain

## CORS Configuration

```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],          # All origins (development only)
    allow_credentials=True,       # Include credentials
    allow_methods=["*"],          # All HTTP methods
    allow_headers=["*"],          # All headers
)
```

> **⚠️ Production Note**: In production, replace `allow_origins=["*"]` with specific allowed origins for security.

## Development Server

### Running the Server

```bash
# Using uvicorn directly
uvicorn server.app:app --host localhost --port 8081 --log-level debug

# Using the provided script
uv run server/app.py
```

### Server Configuration

```python
uvicorn.run(
    app, 
    host="localhost", 
    port=8081, 
    log_level="debug"
)
```

## Integration Examples

### Frontend Integration (React)

```javascript
// Hook for streaming API calls
function useStreamingChat() {
    const [response, setResponse] = useState('');
    const [isLoading, setIsLoading] = useState(false);

    const sendMessage = async (messages) => {
        setIsLoading(true);
        setResponse('');

        try {
            const response = await fetch('/ask/stream', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ messages })
            });

            const reader = response.body.getReader();
            while (true) {
                const { done, value } = await reader.read();
                if (done) break;
                
                const chunk = new TextDecoder().decode(value);
                setResponse(prev => prev + chunk);
            }
        } catch (error) {
            console.error('Streaming error:', error);
        } finally {
            setIsLoading(false);
        }
    };

    return { response, isLoading, sendMessage };
}
```

### CLI Client

```python
import requests
import sys

def stream_chat():
    messages = []
    
    while True:
        user_input = input("You: ")
        if user_input.lower() in ['quit', 'exit']:
            break
            
        messages.append({"role": "human", "content": user_input})
        
        print("AI: ", end="")
        response = requests.post(
            "http://localhost:8081/ask/stream",
            json={"messages": messages},
            stream=True
        )
        
        ai_response = ""
        for chunk in response.iter_content(chunk_size=None):
            if chunk:
                content = chunk.decode('utf-8')
                print(content, end="")
                ai_response += content
        
        messages.append({"role": "ai", "content": ai_response})
        print("\n")

if __name__ == "__main__":
    stream_chat()
```

## Performance Considerations

### Streaming Benefits
- **Real-time Feedback**: Users see responses as they're generated
- **Perceived Performance**: Better user experience even with longer processing times
- **Connection Efficiency**: Single connection for entire response

### Optimization Tips
- **Chunk Size**: Response is streamed in optimal chunks
- **Connection Pooling**: Reuse connections for multiple requests
- **Caching**: Consider caching frequent queries (future enhancement)

## Monitoring and Logging

### Request Logging
```python
# Automatic logging through FastAPI and uvicorn
# Log level: debug for development, info for production
```

### Health Checks

While not explicitly defined, you can add health check endpoints:

```python
@app.get("/health")
async def health_check():
    return {"status": "healthy", "timestamp": datetime.utcnow()}

@app.get("/ready")
async def readiness_check():
    # Check database connection
    # Check knowledge agent status
    return {"status": "ready"}
```

## Security Considerations

### Current State (Development)
- No authentication required
- CORS allows all origins
- No rate limiting

### Production Recommendations
1. **Authentication**: Implement API key or OAuth
2. **CORS**: Restrict to specific domains
3. **Rate Limiting**: Prevent abuse
4. **HTTPS**: Encrypt all communications
5. **Input Validation**: Sanitize user inputs
6. **Monitoring**: Log and monitor API usage

## Error Handling

### Client Errors (4xx)
- **400**: Malformed request body
- **422**: Validation errors in request schema

### Server Errors (5xx)
- **500**: Internal processing errors
- **503**: Service unavailable (database connection issues)

### Error Response Format
```json
{
    "detail": "Error description",
    "type": "error_type",
    "timestamp": "2024-01-01T00:00:00Z"
}
```

## Future API Enhancements

### Planned Endpoints
- `GET /health` - Health check endpoint
- `POST /ask` - Non-streaming response endpoint
- `GET /sources` - List available knowledge sources
- `POST /feedback` - User feedback on responses

### Advanced Features
- **Response Caching**: Cache frequent queries
- **Request Queuing**: Handle high load scenarios
- **Webhook Support**: Notify external systems
- **Multi-format Response**: JSON, XML, plain text options
