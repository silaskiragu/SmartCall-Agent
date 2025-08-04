#!/usr/bin/env python3
"""
LiveKit Agent Worker - Outbound Calling with RAG Integration
Handles real-time voice conversations with knowledge base access using OpenAI Realtime API
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from typing import Any, Dict, List

from dotenv import load_dotenv
from livekit import rtc, api
from livekit.agents import (
    Agent,
    AgentSession,
    AutoSubscribe,
    JobContext,
    WorkerOptions,
    cli,
    function_tool,
    RunContext,
)
from livekit.plugins import openai
from pymongo import MongoClient
from pinecone import Pinecone

# Load environment variables
load_dotenv(dotenv_path=".env")

# Logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("outbound-caller")

# Configuration
outbound_trunk_id = os.getenv("SIP_OUTBOUND_TRUNK_ID")

# Database connections
MONGODB_URI = os.getenv("MONGODB_URI", "mongodb://localhost:27017")
MONGO_DB_NAME = os.getenv("MONGO_DB_NAME", "ai_agent_demo")

try:
    mongo_client = MongoClient(MONGODB_URI, serverSelectionTimeoutMS=5000)
    db = mongo_client[MONGO_DB_NAME]
    agents_collection = db["agents"]
    calls_collection = db["calls"]
    logger.info("Connected to MongoDB for agent worker")
except Exception as e:
    logger.error(f"Failed to connect to MongoDB: {e}")
    db = None
    agents_collection = None
    calls_collection = None

# Pinecone setup
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "luminous-pine")

if PINECONE_API_KEY:
    try:
        pc = Pinecone(api_key=PINECONE_API_KEY)
        index = pc.Index(PINECONE_INDEX_NAME)
        logger.info("Connected to Pinecone for agent worker")
    except Exception as e:
        logger.error(f"Failed to connect to Pinecone: {e}")
        index = None
else:
    index = None

def generate_embedding(text: str, model: str = "text-embedding-3-small"):
    """Generate embeddings using OpenAI API"""
    try:
        import openai as openai_client
        
        openai_api_key = os.getenv("OPENAI_API_KEY")
        if not openai_api_key:
            logger.error("OPENAI_API_KEY environment variable is required for embeddings")
            return None
            
        client = openai_client.OpenAI(api_key=openai_api_key)
        
        response = client.embeddings.create(
            input=text,
            model=model
        )
        return response.data[0].embedding
    except Exception as e:
        logger.error(f"Error generating embedding: {str(e)}")
        return None

async def query_agent_knowledge(agent_id: str, query: str, top_k: int = 5) -> List[Dict]:
    """Query the agent's knowledge base from Pinecone"""
    if index is None:
        logger.warning("Pinecone index not available")
        return []
        
    try:
        namespace = f"agent_{agent_id}"
        
        logger.info(f"Querying Pinecone namespace: {namespace} for query: {query}")
        
        # Generate embedding for the query
        query_embedding = generate_embedding(query)
        if not query_embedding:
            logger.error("Failed to generate query embedding")
            return []
        
        # Query Pinecone
        results = index.query(
            vector=query_embedding,
            top_k=top_k,
            namespace=namespace,
            include_metadata=True
        )
        
        logger.info(f"Pinecone query returned {len(results.matches)} matches")
        
        # Extract relevant context
        contexts = []
        if results.matches:
            for match in results.matches:
                if match.score > 0.3:  # Relevance threshold
                    contexts.append({
                        "text": match.metadata.get("text", ""),
                        "source": match.metadata.get("source", ""),
                        "title": match.metadata.get("title", ""),
                        "score": match.score
                    })
        
        logger.info(f"Filtered contexts: {len(contexts)} items above threshold")
        return contexts
    except Exception as e:
        logger.error(f"Error querying agent knowledge: {e}")
        return []

class KnowledgeAgent(Agent):
    """Agent with knowledge base integration"""
    
    def __init__(self, *, agent_id: str, agent_data: dict, dial_info: dict):
        self.agent_id = agent_id
        self.agent_data = agent_data
        self.dial_info = dial_info
        
        # Build instructions with knowledge base integration
        instructions = f"""
You are {agent_data.get('name', 'Assistant')}, an AI assistant making an outbound call. You MUST respond in English only.

Your persona: {agent_data.get('persona', 'A helpful AI assistant')}

Additional instructions: {agent_data.get('instructions', '')}

CRITICAL GUIDELINES:
1. When the call connects, greet the person naturally and introduce yourself
2. Be helpful and do whatever the user asks for (within appropriate limits)
3. Use the search_knowledge_base function when users ask questions about specific topics, services, or products
4. Follow your persona and be authentic to your character
5. Keep responses conversational and appropriate for phone calls
6. Be flexible - help with whatever the user needs
7. If you detect an answering machine, keep your message brief and professional
8. If the user requests to be transferred to a human, let them know you'll help with that
9. RESPOND ONLY IN ENGLISH
10. Always be helpful and provide useful information

Phone conversation guidelines:
- Be natural and conversational
- Listen to what the user actually wants
- Adapt to their needs and requests
- Keep responses clear and helpful
- Ask follow-up questions when appropriate
- Be engaging and personable according to your persona
"""
        
        super().__init__(instructions=instructions)

    @function_tool()
    async def search_knowledge_base(self, ctx: RunContext, query: str) -> str:
        """Search the agent's knowledge base for relevant information about any topic.
        
        Args:
            query: The search query for finding relevant information
        """
        try:
            logger.info(f"Searching knowledge base for agent {self.agent_id} with query: {query}")
            contexts = await query_agent_knowledge(self.agent_id, query)
            
            if not contexts:
                logger.warning(f"No relevant information found in knowledge base for query: {query}")
                return "No specific information found in my knowledge base for this query, but I'm happy to help in other ways."
            
            # Format the context for the agent
            formatted_context = "Here's what I found in my knowledge base:\n\n"
            for i, context in enumerate(contexts[:3], 1):  # Top 3 results
                # Keep full context but make it conversational
                text = context['text']
                if len(text) > 300:
                    text = text[:300] + "..."
                formatted_context += f"{text}\n\n"
                if context['source']:
                    formatted_context += f"(Source: {context['source']})\n\n"
            
            logger.info(f"Found {len(contexts)} relevant contexts for query: {query}")
            return formatted_context
            
        except Exception as e:
            logger.error(f"Error searching knowledge base: {e}")
            return "I encountered an error while searching for that information, but I'll do my best to help you anyway."

    @function_tool()
    async def answer_question(self, ctx: RunContext, question: str) -> str:
        """Answer any question the user has by searching the knowledge base first.
        
        Args:
            question: The user's question
        """
        # First search knowledge base
        knowledge_result = await self.search_knowledge_base(ctx, question)
        
        if "No specific information found" not in knowledge_result:
            return f"Great question! {knowledge_result}"
        else:
            return "That's a good question. While I don't have specific information about that in my knowledge base, I'm here to help however I can. Could you tell me more about what you're looking for?"

    @function_tool()
    async def help_with_request(self, ctx: RunContext, request: str) -> str:
        """Help the user with whatever they're asking for.
        
        Args:
            request: What the user is asking for help with
        """
        # Search knowledge base for relevant information
        contexts = await query_agent_knowledge(self.agent_id, request)
        
        if contexts:
            # Use the found information to help
            relevant_info = contexts[0]['text'][:400]
            return f"I'd be happy to help you with that. Based on what I know: {relevant_info}"
        else:
            return f"I'll do my best to help you with {request}. Let me know what specific information you need."

async def entrypoint(ctx: JobContext):
    """Main entrypoint for the LiveKit agent worker"""
    logger.info(f"Connecting to room {ctx.room.name}")
    
    try:
        await ctx.connect(auto_subscribe=AutoSubscribe.AUDIO_ONLY)
    except Exception as e:
        logger.error(f"Failed to connect to room: {e}")
        ctx.shutdown()
        return

    # Parse job metadata - try database lookup by room name
    logger.info(f"=== COMPREHENSIVE METADATA DEBUG ===")
    logger.info(f"Room name: {ctx.room.name}")
    logger.info(f"Job ID: {ctx.job.id}")
    
    dial_info = None
    metadata_found_in = None
    
    # First try standard metadata
    if ctx.job.metadata and ctx.job.metadata.strip():
        try:
            dial_info = json.loads(ctx.job.metadata)
            metadata_found_in = "ctx.job.metadata"
            logger.info(f"SUCCESS: Found metadata in {metadata_found_in}: {dial_info}")
        except Exception as e:
            logger.warning(f"Failed to parse ctx.job.metadata: {e}")
    
    # If not found, try database lookup - multiple approaches
    if not dial_info and calls_collection is not None:
        room_name = ctx.room.name
        logger.info(f"Attempting database lookup for room: {room_name}")
        
        try:
            call_record = None
            
            # Method 1: Try room name suffix matching
            if room_name.startswith("call-"):
                call_suffix = room_name.replace("call-", "")
                logger.info(f"Looking for call with suffix: {call_suffix}")
                
                call_record = calls_collection.find_one({
                    "call_id": {"$regex": f".*{call_suffix}$"}
                })
                
                if call_record:
                    logger.info(f"Found call by suffix: {call_record['call_id']}")
            
            # Method 2: If not found, get the most recent call with "initiated" status
            if not call_record:
                logger.info("Suffix method failed, trying most recent initiated call")
                call_record = calls_collection.find_one(
                    {"status": "initiated"},
                    sort=[("created_at", -1)]
                )
                
                if call_record:
                    logger.info(f"Found most recent initiated call: {call_record['call_id']}")
            
            # Method 3: If still not found, get any recent call
            if not call_record:
                logger.info("Trying any recent call from last 5 minutes")
                from datetime import timedelta
                five_minutes_ago = datetime.utcnow() - timedelta(minutes=5)
                call_record = calls_collection.find_one(
                    {"created_at": {"$gte": five_minutes_ago}},
                    sort=[("created_at", -1)]
                )
                
                if call_record:
                    logger.info(f"Found recent call: {call_record['call_id']}")
            
            # Extract dial_info from found call record
            if call_record:
                if 'dial_info' in call_record:
                    dial_info = call_record['dial_info']
                    metadata_found_in = "database_lookup"
                    logger.info(f"SUCCESS: Found dial_info in database: {dial_info}")
                else:
                    # Reconstruct dial_info from call record
                    dial_info = {
                        "agent_id": call_record.get("agent_id"),
                        "phone_number": call_record.get("phone_number"),
                        "voice_actor": call_record.get("voice_actor"),
                        "tone": call_record.get("tone"),
                        "prompt_vars": call_record.get("prompt_vars", {}),
                        "metadata": call_record.get("metadata", {})
                    }
                    metadata_found_in = "database_reconstruction"
                    logger.info(f"SUCCESS: Reconstructed dial_info from database: {dial_info}")
                    
                # Update call status to show it's being processed
                calls_collection.update_one(
                    {"_id": call_record["_id"]},
                    {"$set": {"status": "processing", "worker_room": room_name}}
                )
            else:
                logger.warning(f"No call record found using any method")
                
        except Exception as e:
            logger.error(f"Error during database lookup: {e}")
    
    # Fallback to latest agent if still nothing found
    if not dial_info and agents_collection is not None:
        try:
            latest_agent = agents_collection.find().sort("created_at", -1).limit(1)
            latest_agent = list(latest_agent)
            if latest_agent:
                latest_agent = latest_agent[0]
                dial_info = {
                    "agent_id": latest_agent["agent_id"],
                    "phone_number": "+918789020048",  # Default fallback
                    "voice_actor": latest_agent.get("default_voice", "alloy"),
                    "tone": latest_agent.get("default_tone", "professional"),
                    "metadata": {}
                }
                metadata_found_in = "latest_agent_fallback"
                logger.warning(f"Using latest agent as final fallback: {dial_info}")
        except Exception as e:
            logger.error(f"Error getting latest agent: {e}")
    
    if not dial_info:
        logger.error("Could not determine agent information from any source")
        ctx.shutdown()
        return
    else:
        logger.info(f"Final dial_info source: {metadata_found_in}")
        logger.info(f"Final dial_info: {dial_info}")
        
        # Validate required fields
        if not dial_info.get("agent_id") or not dial_info.get("phone_number"):
            logger.error(f"Invalid dial_info - missing required fields: {dial_info}")
            ctx.shutdown()
            return

    if "phone_number" not in dial_info:
        logger.error("Missing required 'phone_number' in metadata")
        ctx.shutdown()
        return

    # Get agent information from database
    agent_id = dial_info.get("agent_id")
    if not agent_id:
        logger.error("Missing agent_id in dial_info")
        ctx.shutdown()
        return

    # Fetch agent from database
    agent_data = None
    if agents_collection is not None:
        try:
            agent_data = agents_collection.find_one({"agent_id": agent_id})
            logger.info(f"Agent data found: {bool(agent_data)}")
        except Exception as e:
            logger.error(f"Error fetching agent data: {e}")
    
    if not agent_data:
        logger.error(f"Agent {agent_id} not found in database")
        ctx.shutdown()
        return

    participant_identity = phone_number = dial_info["phone_number"]

    # Get voice setting with proper fallback
    voice_setting = dial_info.get("voice_actor")
    if not voice_setting:
        voice_setting = agent_data.get("default_voice")
    if not voice_setting:
        voice_setting = "alloy"  # Final fallback
    
    logger.info(f"Using voice setting: {voice_setting}")

    # Create knowledge-enabled agent
    agent = KnowledgeAgent(
        agent_id=agent_id,
        agent_data=agent_data,
        dial_info=dial_info
    )

    # Create agent session with Realtime API
    try:
        session = AgentSession(
            llm=openai.realtime.RealtimeModel(
                voice=voice_setting,
                temperature=agent_data.get("temperature", 0.7),
                # Don't pass instructions to RealtimeModel - it doesn't accept them
            )
        )
        logger.info("Created AgentSession with Realtime model")
    except Exception as e:
        logger.error(f"Failed to create AgentSession: {e}")
        ctx.shutdown()
        return

    # Create SIP participant and start dialing
    try:
        logger.info(f"Creating SIP participant for {phone_number}")
        await ctx.api.sip.create_sip_participant(
            api.CreateSIPParticipantRequest(
                room_name=ctx.room.name,
                sip_trunk_id=outbound_trunk_id,
                participant_identity=participant_identity,
                sip_call_to=phone_number,
            )
        )

        logger.info("Starting agent session...")
        session_task = asyncio.create_task(
            session.start(agent=agent, room=ctx.room)
        )

        logger.info("Waiting for participant to join...")
        participant = await ctx.wait_for_participant(identity=participant_identity)
        logger.info(f"Participant joined: {participant.identity}")

        # Wait for session to be ready
        await session_task
        
        # Wait a moment for audio to be established
        await asyncio.sleep(2)
        
        # Log audio track status
        for track_pub in participant.track_publications.values():
            logger.info(f"Track: {track_pub.name}, Kind: {track_pub.kind}, Subscribed: {track_pub.subscribed}")

        logger.info("Call established - agent will respond when user speaks")

        # Update call status in database
        if calls_collection is not None:
            try:
                import time
                calls_collection.update_one(
                    {"phone_number": phone_number},
                    {"$set": {"status": "connected", "connected_at": time.time()}}
                )
                logger.info("Updated call status to connected")
            except Exception as e:
                logger.error(f"Failed to update call status: {e}")

        logger.info("Call established successfully")

    except api.TwirpError as e:
        logger.error(
            f"Error creating SIP participant: {e.message}, "
            f"SIP status: {e.metadata.get('sip_status_code') if e.metadata else 'N/A'} "
            f"{e.metadata.get('sip_status') if e.metadata else 'N/A'}"
        )
        
        # Update call status in database
        if calls_collection is not None:
            try:
                import time
                calls_collection.update_one(
                    {"phone_number": phone_number},  
                    {"$set": {
                        "status": "failed", 
                        "error": e.message,
                        "sip_status_code": e.metadata.get('sip_status_code') if e.metadata else None,
                        "failed_at": time.time()
                    }}
                )
            except Exception as db_error:
                logger.error(f"Failed to update call status in database: {db_error}")
        
        ctx.shutdown()
    except Exception as e:
        logger.error(f"Unexpected error during call setup: {e}")
        ctx.shutdown()

if __name__ == "__main__":
    cli.run_app(
        WorkerOptions(
            entrypoint_fnc=entrypoint,
        )
    )