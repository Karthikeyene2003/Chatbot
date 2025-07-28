#fastapi
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import asyncio

from langchain.prompts import PromptTemplate
from langchain_ollama.llms import OllamaLLM
from langchain_core.output_parsers import StrOutputParser

#retriever imports
from hybrid_salespro_nollm import compression_retriever_sales
from hybrid_url_nollm import compression_retriever_url
from hybrid_faq_nollm import compression_retriever_books

#Redis for memory import
import uuid
from redis import Redis
from langchain_community.chat_message_histories import RedisChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

# ------------------setup------------------------ #
app = FastAPI()

#cors
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Redis setup
redis_url = "<YOUR_REDIS_URL>"
redis_client = Redis.from_url(redis_url)

def get_chat_history(session_id: str):
    return RedisChatMessageHistory(
        session_id=session_id,
        url=redis_url,
        key_prefix="chat_history"
    )

llm = OllamaLLM(model="<model-name>", temperature=0)
parser = StrOutputParser()

# ----------------- intent detection prompt ---------------- #
detect_intent_prompt = PromptTemplate(
    input_variables=["query"],
    template="""
You are the intent detection engine.

Classify the user query as one of the following intents:
1. Product Recommendation
2. Website Navigation Help
3. Return, Refund, Order
4. Customer Support and FAQ
4. Unclear Intent
5. Greeting
6. Follow-up Question

Only return the intent.

query: {query}
intent:
"""
)
intent_chain = detect_intent_prompt | llm | parser

# -----------------response generation prompt ---------------- #
response_prompt = PromptTemplate(
    input_variables=["chat_history","query", "retrieved_content"],
    template="""
You are a polite and helpful virtual assistant for an online platform.

Your roles are limited to:
- Helping users navigate the platform or website
- Recommending suitable products or services
- Assisting with order, return, refund, and shipping-related queries

Use only the retrieved content below to generate a response.  
Do not make up information, links, or services that are not explicitly mentioned.

If the query is ambiguous, ask a clarifying question.  
If relevant content is not found, apologize and politely direct the user to contact support.

Strict Instructions:
- Only respond using relevant retrieved content.
- Remove any parts of retrieved content that are weakly related or irrelevant.
- Do not fabricate responses or infer extra context beyond what's retrieved.

Step-by-step Filtering Process:
1. Identify helpful sections from the retrieved content.  
2. Discard unrelated or loosely related sections.  
3. Use what's left to answer the query accurately.

Example:  
User Query: How do I contact the support team?  
Retrieved Content:  
- Call: 1800 000 0000  
- Email: help@example.com  
- Refund policy: return within 1 week, shipping not reimbursed  
Step 1: User only asked for contact info => Use phone/email  
Step 2: Discard refund policy (not asked)  
Response: You can contact our support team via:  
- Call: xxxxxxxxxxxxx 
- Email: help@example.com

Ongoing Converstaion:
{chat_history}

User Query:
{query}

Retrieved Info:
{retrieved_content}

Response:
"""
)
response_chain = response_prompt | llm | parser

response_with_memory = RunnableWithMessageHistory(
    response_chain,
    get_chat_history,
    input_messages_key="query",
    history_messages_key="messages",
)

# ------------------ formatting and session_id handling  ------------------ #
def get_session_id(possible_id=None):
    return possible_id if possible_id else str(uuid.uuid4())

def format_docs(docs):
    return "\n\n".join([
        f"""Title: {doc.metadata.get("name", "N/A")}
Class: {doc.metadata.get("class", "N/A")}
Link: {doc.metadata.get("url", "N/A")}
Summary: {doc.page_content}"""
        for doc in docs
    ])

# ------------------ BOSS------------------ #
def run_bot(query, session_id):
    query = query.strip()
    intent_result = intent_chain.invoke({"query": query}).strip().lower()
# Route to appropriate retriever

    # Handle follow-up using Redis memory:
    if "follow-up" in intent_result:
        last_intent = redis_client.get(f"last_intent:{session_id}")
        last_query = redis_client.get(f"last_query:{session_id}")

        if not last_intent or not last_query:
            return "I'm not sure what you're following up on. Could you please clarify if it's about a product or a service?"
        
        # Use previous intent and full query context
        intent_result = last_intent.decode()
        query = f"{last_query.decode()} --> Follow-up: {query}"

    # Save current query and intent for future follow-ups with TTL
    TTL_SECONDS = 7200  # 2 hours

    redis_client.set(f"last_intent:{session_id}", intent_result, ex=TTL_SECONDS)
    redis_client.set(f"last_query:{session_id}", query, ex=TTL_SECONDS)

    # Set TTL for chat history key manually
    redis_client.expire(f"chat_history:{session_id}", TTL_SECONDS)
#------------------------- Speciial Cases----------------------------#    
    # Handle greetings:
    if "greeting" in intent_result:
        return "Hello! How can I assist you today? Are you looking for a product or help navigating the website?"
    
    # Unclear intention
    elif "unclear" in intent_result:
        return "Could you please clarify if you're looking for a product or trying to navigate a service on the website or need info on our comapany policies or want to customer support?"
#  Route to appropriate retriever  
    # customer support and faq
    elif "customer support and faq" in intent_result:
        docs = compression_retriever_books.invoke(query)
    
    # Handles product recommendation
    elif "product recommendation" in intent_result:
        docs = compression_retriever_sales.invoke(query)
    
    # Handles website navigation:
    elif "website navigation help" in intent_result:
        docs = compression_retriever_url.invoke(query)
    
    elif "return, refund, order" in intent_result:
        docs = compression_retriever_books.invoke(query)

    # Fallback---> detected intent printed, for debugging
    else:
        return f"Sorry, I couldn't understand your request. Detected intent: {intent_result}"
    
    if not docs:
        return "Sorry, I could not find any relevant information for your query.\nPlease contact our CRM team."

    retrieved_content = format_docs(docs)

    # chat_buffer and chat_history calling---> gives bot the sliced interaction of bot and human , providing context 
    chat_history = get_chat_history(session_id)
    messages = chat_history.messages[-10:]  # 5 human + 5 AI = 10 messages

    chat_pairs = []
    for msg in messages:
        role = msg.type.capitalize()
        chat_pairs.append(f"{role}: {msg.content}")

    chat_buffer = "\n".join(chat_pairs)

    # Generate response
    final_response = response_with_memory.invoke({
        "chat_history": chat_buffer,
        "query": query,
        "retrieved_content": retrieved_content
    }, config={"configurable": {"session_id": session_id}})

    return final_response

# ------------------ FASTAPI ------------------ #
class QueryPayload(BaseModel):
    query: str
    session_id: str = None

@app.post("/chat")
async def chat_handler(payload: QueryPayload):
    session_id = get_session_id(payload.session_id)

    # Run the CPU-heavy run_bot in a separate thread to avoid blocking
    loop = asyncio.get_event_loop()
    response = await loop.run_in_executor(None, run_bot, payload.query, session_id)

    return {
        "session_id": session_id,
        "response": response
    }
#clear session
@app.post("/clear-session")
def clear_chat(payload: QueryPayload):
    session_id = get_session_id(payload.session_id)
    get_chat_history(session_id).clear()
    redis_client.delete(f"last_intent:{session_id}")
    redis_client.delete(f"last_query:{session_id}")
    return {"message": f"Cleared session history for session_id: {session_id}"}
