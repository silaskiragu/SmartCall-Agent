# **AI Agent System with RAG and Outbound Calling**

A comprehensive AI agent management system that combines Retrieval-Augmented Generation (RAG) capabilities with real-time voice calling using LiveKit and OpenAI's Realtime API.

## **🚀 Features**

* **AI Agent Creation & Management**: Create custom AI agents with unique personas and behaviors  
* **RAG Integration**: Upload documents (PDF, TXT, DOCX, URLs) to create knowledge bases for agents  
* **Outbound Voice Calling**: Make real-time voice calls using LiveKit SIP integration  
* **Real-time Conversations**: Powered by OpenAI's Realtime API for natural voice interactions  
* **Vector Database**: Pinecone integration for efficient knowledge retrieval  
* **User Authentication**: JWT-based authentication system  
* **Analytics Dashboard**: Call metrics and agent performance tracking  
* **Document Management**: Upload, process, and manage agent knowledge bases  
* **Multiple Voice Options**: Customizable voice actors and tones for calls

## **🏗️ Architecture**

The system consists of two main components:

1. **FastAPI Backend** (`main.py`): RESTful API for agent management, RAG processing, and call orchestration  
2. **LiveKit Agent Worker** (`agent_worker.py`): Real-time voice conversation handler with knowledge base integration

## **📋 Prerequisites**

* Python 3.8+  
* MongoDB database  
* Pinecone account and API key  
* OpenAI API key  
* LiveKit Cloud account  
* SIP trunk for outbound calling

## **🛠️ Installation**

**Clone the repository**

 git clone [https://github.com/shubhamprasad318/ai\_wao\_agent](https://github.com/shubhamprasad318/ai_rag_agent_sip)  
cd ai\_rag\_agent\_sip

**Install dependencies**

 pip install \-r requirements.txt

**Environment Setup**

 Create a `.env` file in the root directory:

 \# OpenAI Configuration  
OPENAI\_API\_KEY=your\_openai\_api\_key\_here

\# MongoDB Configuration  
MONGODB\_URI=mongodb://localhost:27017  
MONGO\_DB\_NAME=ai\_agent\_demo

\# Pinecone Configuration  
PINECONE\_API\_KEY=your\_pinecone\_api\_key\_here  
PINECONE\_INDEX\_NAME=luminous-pine

\# LiveKit Configuration  
LIVEKIT\_URL=wss://your-project.livekit.cloud  
LIVEKIT\_API\_KEY=your\_livekit\_api\_key  
LIVEKIT\_API\_SECRET=your\_livekit\_api\_secret  
SIP\_OUTBOUND\_TRUNK\_ID=your\_sip\_trunk\_id

\# Security  
SECRET\_KEY=your-secret-key-change-this-in-production

\# Optional  
PORT=8000  
ENVIRONMENT=development

1. **Database Setup**

    Ensure MongoDB is running and accessible. The application will create the necessary collections automatically.

2. **Pinecone Index Setup**

    Create a Pinecone index with the following specifications:

   * Dimension: 1536 (for OpenAI text-embedding-3-small)  
   * Metric: Cosine similarity  
   * Index name: Should match `PINECONE_INDEX_NAME` in your `.env`

## **🚀 Running the Application**

### **Start the FastAPI Server**

python main.py

The API will be available at `http://localhost:8000`

### **Start the LiveKit Agent Worker**

python agent\_worker.py

## **📚 API Documentation**

Once the server is running, visit:

* **Interactive API Docs**: `http://localhost:8000/docs`  
* **ReDoc Documentation**: `http://localhost:8000/redoc`

### **Key Endpoints**

#### **Authentication**

* `POST /api/register` \- Register a new user  
* `POST /api/login` \- User login  
* `GET /api/me` \- Get current user profile

#### **Agent Management**

* `POST /api/agent` \- Create a new AI agent  
* `GET /api/agents` \- List all agents  
* `GET /api/agent/{agent_id}` \- Get specific agent  
* `PATCH /api/agent/{agent_id}` \- Update agent configuration  
* `DELETE /api/agent/{agent_id}` \- Delete an agent

#### **Calling**

* `POST /api/call` \- Initiate an outbound call  
* `GET /api/calls` \- List all calls  
* `GET /api/call/{call_id}` \- Get call details

#### **Knowledge Base**

* `GET /api/agent/{agent_id}/query` \- Query agent's knowledge base  
* `GET /api/agent/{agent_id}/documents` \- Get agent documents  
* `DELETE /api/agent/{agent_id}/documents/{doc_id}` \- Delete a document

## **🤖 Creating an AI Agent**

### **Basic Agent Creation**

import requests

agent\_data \= {  
    "name": "Customer Support Agent",  
    "language": "English",  
    "model": "gpt-4",  
    "persona": "A helpful and professional customer support representative",  
    "description": "Handles customer inquiries and support requests",  
    "temperature": 0.7,  
    "max\_tokens": 1000,  
    "default\_voice": "alloy",  
    "default\_tone": "professional"  
}

response \= requests.post("http://localhost:8000/api/agent", json=agent\_data)  
agent \= response.json()

### **Agent with Knowledge Base**

agent\_data \= {  
    "name": "Product Expert",  
    "persona": "An expert in our products with deep technical knowledge",  
    "rag\_docs": \[  
        "https://example.com/product-manual.pdf",  
        "https://example.com/faq.txt",  
        "/path/to/local/documentation.docx"  
    \],  
    "instructions": "Always provide accurate product information and help customers understand our offerings"  
}

response \= requests.post("http://localhost:8000/api/agent", json=agent\_data)

## **📞 Making Outbound Calls**

call\_data \= {  
    "agent\_id": "agent\_abc123",  
    "phone\_number": "+1234567890",  
    "voice\_actor": "alloy",  
    "tone": "friendly",  
    "prompt\_vars": {  
        "customer\_name": "John Doe",  
        "appointment\_time": "3 PM today"  
    },  
    "metadata": {  
        "campaign": "appointment\_reminders"  
    }  
}

response \= requests.post("http://localhost:8000/api/call", json=call\_data)  
call \= response.json()

## **🧠 RAG (Retrieval-Augmented Generation)**

The system automatically processes documents and creates vector embeddings for intelligent knowledge retrieval:

1. **Document Processing**: Supports PDF, TXT, DOCX files and web URLs  
2. **Text Chunking**: Intelligently splits documents into manageable chunks  
3. **Vector Embeddings**: Uses OpenAI's text-embedding-3-small model  
4. **Storage**: Vectors stored in Pinecone with metadata  
5. **Retrieval**: Semantic search during conversations for relevant context

### **Supported Document Types**

* **URLs**: Web pages, PDFs hosted online  
* **PDF Files**: Local or remote PDF documents  
* **Text Files**: Plain text documents  
* **Word Documents**: DOCX format files  
* **Direct Text**: Raw text content

## **🎯 Agent Capabilities**

### **Built-in Functions**

Each agent comes with intelligent function calling capabilities:

* **`search_knowledge_base(query)`**: Search the agent's knowledge base  
* **`answer_question(question)`**: Answer questions using knowledge base  
* **`help_with_request(request)`**: General assistance with user requests

### **Voice Configuration**

* **Voice Options**: alloy, echo, fable, onyx, nova, shimmer  
* **Tone Settings**: professional, friendly, casual, enthusiastic  
* **Language Support**: Multiple languages supported  
* **Real-time Processing**: Natural conversation flow

## **📊 Analytics & Monitoring**

### **Dashboard Metrics**

* Total agents created  
* Call statistics and success rates  
* RAG processing status  
* Recent activity tracking

### **Call Analytics**

* Call duration and outcomes  
* Voice quality metrics  
* Agent performance tracking  
* Custom metadata analysis

## **🔧 Configuration Options**

### **Agent Configuration**

{  
    "name": "Agent Name",  
    "language": "English",  
    "model": "gpt-4",  
    "persona": "Agent personality description",  
    "instructions": "Additional behavioral instructions",  
    "temperature": 0.7,  \# 0.0 to 2.0  
    "max\_tokens": 1000,  
    "default\_voice": "alloy",  
    "default\_tone": "professional",  
    "custom\_fields": {  
        "transfer\_to": "+1234567890",  
        "department": "support"  
    },  
    "metadata": {  
        "version": "1.0",  
        "created\_by": "admin"  
    }  
}

### **Call Configuration**

{  
    "agent\_id": "agent\_123",  
    "phone\_number": "+1234567890",  
    "voice\_actor": "nova",  
    "tone": "friendly",  
    "prompt\_vars": {  
        "variable\_name": "value"  
    },  
    "metadata": {  
        "campaign": "outbound\_sales",  
        "priority": "high"  
    }  
}

## **🚨 Error Handling**

The system includes comprehensive error handling:

* **Database Connection**: Graceful fallbacks when MongoDB/Pinecone unavailable  
* **API Rate Limits**: Automatic retry logic for OpenAI API calls  
* **Call Failures**: SIP error tracking and reporting  
* **Document Processing**: Robust error handling for various file formats

## **🔒 Security Features**

* **JWT Authentication**: Secure token-based authentication  
* **Password Hashing**: bcrypt password encryption  
* **CORS Configuration**: Configurable cross-origin request handling  
* **Input Validation**: Pydantic models for request validation  
* **Rate Limiting**: Built-in protection against abuse

## **📈 Scaling Considerations**

### **Performance Optimization**

* **Batch Processing**: Efficient document processing in batches  
* **Vector Search**: Optimized Pinecone queries with relevance thresholds  
* **Caching**: MongoDB indexing for fast agent retrieval  
* **Async Processing**: Background tasks for RAG document processing

## **🛠️ Development**

### **Project Structure**

ai-agent-system/  
├── main.py                 \# FastAPI backend server  
├── agent\_worker.py         \# LiveKit agent worker  
├── requirements.txt        \# Python dependencies  
├── .env                   \# Environment configuration  
├── README.md              \# This file  
└── docs/                  \# Additional documentation

### **Dependencies**

Key dependencies include:

* **FastAPI**: Web framework  
* **LiveKit**: Real-time communication  
* **OpenAI**: LLM and embeddings  
* **Pinecone**: Vector database  
* **PyMongo**: MongoDB client  
* **PyPDF2**: PDF processing  
* **python-docx**: Word document processing

## **🤝 Contributing**

1. Fork the repository  
2. Create a feature branch (`git checkout -b feature/amazing-feature`)  
3. Commit your changes (`git commit -m 'Add amazing feature'`)  
4. Push to the branch (`git push origin feature/amazing-feature`)  
5. Open a Pull Request

## **🔄 Changelog**

### **v1.0.0**

* Initial release  
* AI agent creation and management  
* RAG integration with Pinecone  
* Outbound calling with LiveKit  
* User authentication system  
* Analytics dashboard

---

**Built with ❤️ using FastAPI, LiveKit, OpenAI, and Pinecone,Plivo**

pinecone -- https://app.pinecone.io/organizations/-OWodZyDTiYiOqoBzFGF/projects/029c5e24-40fc-4ed3-b2ae-2f580d147841/indexes/luminous-pine/browser