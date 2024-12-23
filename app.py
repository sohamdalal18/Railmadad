import streamlit as st
import os
import warnings
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_core.embeddings import Embeddings
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

warnings.filterwarnings("ignore")

# Embedding Wrapper Class
class SentenceTransformerEmbeddings(Embeddings):
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        self.model = SentenceTransformer(model_name)

    def embed_documents(self, texts):
        return self.model.encode(texts).tolist()

    def embed_query(self, query):
        return self.model.encode(query).tolist()


# Load environment variables
load_dotenv()
api_key = os.getenv("GROQ_API_KEY")

if not api_key:
    raise ValueError("GROQ_API_KEY environment variable is not set.")

# Initialize LLM
llm = ChatGroq(groq_api_key=api_key, model_name="Llama3-8b-8192")

# Define Prompt Template
prompt = ChatPromptTemplate.from_template(
    """
Answer the questions based on the provided context only.
Please provide the most accurate response based on the question.
<context>
{context}
</context>
Questions: {input}
"""
)


# Function to Create Vector Embeddings
def vector_embedding():
    try:
        embeddings = SentenceTransformerEmbeddings()
        loader = PyPDFDirectoryLoader("D:/Railmadad/Railmadad/data")  # Correct PDF directory path
        docs = loader.load()

        if not docs or len(docs) == 0:
            st.error("❌ No documents were loaded. Please ensure the directory contains valid PDF files.")
            return

        # Debug the number of documents loaded
        st.write(f"📄 Loaded documents successfully.")

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        final_documents = text_splitter.split_documents(docs[:20])  # First 20 documents

        if not final_documents or len(final_documents) == 0:
            st.error("❌ Failed to split documents. Check the content and formatting of your PDFs.")
            return

        # Create FAISS vector database
        st.session_state.vectors = FAISS.from_documents(final_documents, embeddings)
        st.write("✅ Vector Store DB Is Ready")

    except Exception as e:
        st.error(f"❌ Error creating vector database: {e}")



# Function to Process User Queries
def process_query_with_document(prompt1):
    if not prompt1:
        st.warning("⚠️ Please enter a query before clicking 'Get Answer'.")
        return False  # Return False if no query entered
    
    if "vectors" not in st.session_state or st.session_state.vectors is None:
        st.warning("⚠️ Please create a vector database first!")
        return False  # Return False if vector database is missing

    try:
        # Document Similarity Search (RAG)
        retriever = st.session_state.vectors.as_retriever()
        document_chain = create_stuff_documents_chain(llm, prompt1)
        retrieval_chain = create_retrieval_chain(retriever, document_chain)
        response = retrieval_chain.invoke({'input': prompt1})

        # Check if the response contains useful information
        if response.get('answer') and response['answer'].strip():
            st.subheader("✅ Document-Based Answer:")
            st.write(response['answer'])

            with st.expander("📚 Document Similarity Search"):
                for doc in response["context"]:
                    st.write(doc.page_content)
                    st.write("---")

            return True  # Return True if a valid answer is found
        else:
            st.subheader("❌ No Relevant Document Found")
            st.write("We couldn't find any relevant information in the documents.")
            return False  # Return False if no relevant content found
    
    except AttributeError:
        st.warning("Hold on a second...")
        return False

    except Exception as e:
        st.error(f"❌ Hold on a second: {e}")
        return False  # Return False if an error occurs

    
def process_query_with_custom(prompt1, custom_prompt_template):
    if not prompt1:
        st.warning("⚠️ Please enter a query before clicking 'Get Answer'.")
        return
    
    try:
        output_parser = StrOutputParser()
        chatbot_chain = custom_prompt_template | llm | output_parser
        response = chatbot_chain.invoke({"question": prompt1})
        
        st.subheader("🤖 Chatbot Answer:")
        st.write(response)
    except Exception as e:
        st.error(f"❌ Error processing query with chatbot: {e}")


# Women Safety Page
def women_safety_page():
    st.title("👩 Women Safety Department")
    st.write("This page provides details and support for women safety services.")
    
    if st.button("🛠️ Create Vector Database"):
        vector_embedding()
    
    prompt1 = st.text_input("💬 How may I help you?")
    
    if st.button("📤 Get Answer"):
        if not prompt1:
            st.warning("⚠️ Please enter a query before clicking 'Get Answer'.")
            return
        
        if "vectors" not in st.session_state or st.session_state.vectors is None:
            st.warning("⚠️ Please create a vector database first!")
            return
        
        if not process_query_with_document(prompt1):  # This will now return False if no relevant answer is found
            
            women_safety_prompt = ChatPromptTemplate.from_template("""
            You are an expert in Indian Railways' Women Safety services with in-depth knowledge of safety protocols, helpline services, grievance redressal mechanisms, and initiatives aimed at ensuring the safety and security of women passengers. 
            Your responses should be clear, accurate, and tailored to address women safety-related queries effectively.

            ### Emergency Handling Instructions:
            If the users query indicates an urgent safety concern (e.g., harassment, molestation, stalking, or assault), you must:
            - Provide immediate SOS action steps.
            - Share the helpline number 182 for Indian Railways or 112 for general emergencies.
            - Offer self-protection tips and steps to report the incident.
            - Use a professional, compassionate, and reassuring tone.
            - Redirect the user to the Indian Railways Police website: [Indian Railways Police (RPF)](https://railmadad.indianrailways.gov.in/madad/final/home.jsp). 
            **Emergency Query Examples:**
            - "I am being harassed on the train. What should I do?"
            - "What to do if someone is stalking me at the station?"
            - "How can I report a molestation incident on the train?"

            For non-emergency queries related to womens safety, provide clear and concise information about:
            - Women helpline numbers and emergency contact details.
            - Reserved coaches or seats for women.
            - Onboard safety measures such as the presence of security personnel.
            - Filing complaints via helpline or grievance portals.
            - Security arrangements at stations during late hours.

            ### Strict Query Handling:
            If the user asks a question unrelated to women safety in Indian Railways, respond with:
            "I am here to assist with women safety-related queries in Indian Railways only. Please ask me a question related to Women Safety in Indian Railways."

            **Examples of Non-Emergency Queries You Can Handle:**
            - What helpline numbers can women passengers use in emergencies?  
            - Are there reserved coaches or seats for women on trains?  
            - How can a woman passenger file a safety complaint while onboard?  
            - What are the security measures available for women passengers on Indian trains?  
            - How does Indian Railways ensure women's safety at stations during late hours?

            ### Tone and Style:
            - Be professional, compassionate, and reassuring.
            - Provide detailed, actionable advice for safety and emergencies.
            - Simplify complex procedures for better understanding.

            ### Constraints:
            - Avoid using raw Markdown or asterisks in the text directly.
            Question: {question}
            """
            )
            
            process_query_with_custom(prompt1, women_safety_prompt)


# Emergency Services Page
def emergency_services_page():
    st.title("🚨 Emergency Services")
    st.write("This page provides immediate help and support.")
    
    if st.button("🛠️ Create Vector Database"):
        vector_embedding()
    
    prompt1 = st.text_input("💬 How may I help you?")
    
    if st.button("📤 Get Answer"):
        if not prompt1:
            st.warning("⚠️ Please enter a query before clicking 'Get Answer'.")
            return
        
        if "vectors" not in st.session_state or st.session_state.vectors is None:
            st.warning("⚠️ Please create a vector database first!")
            return
        
        if not process_query_with_document(prompt1):  # This will now return False if no relevant answer is found
            emergency_prompt = ChatPromptTemplate.from_template(
                    """
                    You are an expert in Indian Railways' Emergency Services, equipped with detailed knowledge of emergency protocols, helpline services, accident response measures, passenger safety mechanisms, and real-time assistance systems. Your expertise includes helpline numbers (e.g., 139, 182), onboard security measures, accident relief procedures, fire safety protocols, medical assistance, and emergency evacuation guidelines.

                    ### Emergency Handling Guidelines:
                    - If a user’s query indicates an urgent emergency (e.g., fire, medical distress, train accident, or evacuation), you must:
                    - Provide the **immediate helpline numbers**: 139 (general emergency) or 182 (Railway Protection Force).  
                    - Offer step-by-step action plans (e.g., how to report the emergency, initial first-aid steps).  
                    - Suggest resources available onboard or at the nearest station (e.g., first-aid kits, security personnel).  
                    - Redirect the user to Indian Railways Emergency Assistance: [Railway Helpline Portal](https://railmadad.indianrailways.gov.in/madad/final/home.jsp).  

                    ### Examples of Emergency Queries You Can Handle:
                    - "What is the emergency helpline number for Indian Railways?"  
                    - "What should I do if there is a fire onboard?"  
                    - "How can passengers report accidents while traveling by train?"  
                    - "Where can I find first-aid assistance onboard?"  
                    - "What safety measures are in place for train evacuation during emergencies?"

                    ### Query Filtering:
                    - If the users question is **unrelated** to emergency services in Indian Railways, politely respond with:  
                    "I am here to assist with emergency-related queries in Indian Railways only. Please ask me a question related to Emergency Services in Indian Railways."

                    ### Response Guidelines:
                    - **For Fire Emergencies**:  
                    1. Alert train staff or pull the emergency chain immediately.  
                    2. Use available fire extinguishers onboard (found near train exits).  
                    3. Evacuate to a safe location, following staff instructions.  
                    4. Contact **139** or **182** for further assistance.  
                    5. Administer first aid for burns: Cool the area with clean water, avoid applying ointments, and cover loosely with a sterile cloth.

                    - **For Medical Emergencies**:  
                    1. Call **139** and provide details of the affected passenger.  
                    2. Check if trained personnel or medical kits are available onboard.  
                    3. Administer first-aid (e.g., CPR, wound care) if trained and needed.  
                    4. Request emergency medical assistance at the next station.

                    - **For Train Accidents**:  
                    1. Stay calm and help others remain composed.  
                    2. Follow evacuation instructions from train staff.  
                    3. Contact **139** or **182** to report the incident.  
                    4. Assist in helping injured passengers to safety.

                    - **For Security-Related Emergencies**:  
                    1. Contact **182** immediately for RPF assistance.  
                    2. Alert train staff or security personnel onboard.  
                    3. Move to a secure area until help arrives.

                    ### Tone and Style:
                    - Be professional, calm, and reassuring to instill confidence in the user.
                    - Provide concise and **actionable steps** for emergencies.
                    - Simplify complex procedures for easy understanding.

                    ### Constraints:
                    - Avoid using raw Markdown or asterisks in the text directly.
                    Question: {question}
                    """

                )
            process_query_with_custom(prompt1, emergency_prompt)
        else:
            st.warning("⚠️ Please enter a query before clicking 'Get Answer'.")

# Cleanliness Page
def cleanliness_page():
    st.title("🧹 Cleanliness Department")
    st.write("This page helps maintain clean railway premises.")
    
    if st.button("🛠️ Create Vector Database"):
        vector_embedding()
    
    prompt1 = st.text_input("💬 How may I help you?")
    
    if st.button("📤 Get Answer"):
        if not prompt1:
            st.warning("⚠️ Please enter a query before clicking 'Get Answer'.")
            return
        
        if "vectors" not in st.session_state or st.session_state.vectors is None:
            st.warning("⚠️ Please create a vector database first!")
            return
        
        if not process_query_with_document(prompt1):  # This will now return False if no relevant answer is found
            cleanliness_prompt = ChatPromptTemplate.from_template(
                    """
                    You are an expert in Indian Railways' Cleanliness services with in-depth knowledge of cleanliness initiatives, waste management practices, onboard and station hygiene protocols, and passenger grievance redressal mechanisms related to sanitation. You are well-versed in Swachh Bharat initiatives, bio-toilet systems, cleanliness helpline services, and sanitation best practices.

                    ### Cleanliness Assistance Guidelines:
                    - If the users query indicates an **unclean condition** (e.g., dirty train coaches, littered platforms, malfunctioning bio-toilets), you must:
                    - Provide the **cleanliness helpline number**: **139**.  
                    - Guide users to file complaints via the [Cleanliness Complaint Portal](https://cleanmycoach.com) or SMS `CLEAN <PNR>` to **139**.  
                    - Share actionable steps to report issues onboard or at stations.  
                    - Highlight Indian Railways' key cleanliness initiatives for context.

                    ### Examples of Cleanliness Queries You Can Handle:
                    - "How can I report an unclean coach or platform?"  
                    - "What is the bio-toilet system used in Indian Railways?"  
                    - "How does Indian Railways maintain hygiene on trains and platforms?"  
                    - "What cleanliness initiatives are part of the Swachh Bharat Mission?"  
                    - "What can passengers do to help maintain cleanliness onboard?"

                    ### Query Filtering:
                    - If the users question is **unrelated** to cleanliness in Indian Railways, politely respond with:  
                    "I am here to assist with cleanliness-related queries in Indian Railways only. Please ask me a question related to Cleanliness in Indian Railways."

                    ### Response Guidelines:
                    - **For Reporting Unclean Coaches or Platforms**:  
                    1. File a complaint via SMS: `CLEAN <PNR>` to **139**.  
                    2. Use the [Clean My Coach Portal](https://cleanmycoach.com).  
                    3. Inform onboard staff or station authorities for immediate action.

                    - **For Understanding Bio-Toilet Systems**:  
                    1. Explain that bio-toilets use anaerobic bacteria to decompose waste, reducing environmental impact.  
                    2. Mention that they are designed to eliminate open discharge and ensure better hygiene.

                    - **For General Hygiene Practices**:  
                    1. Inform passengers about the frequency of cleaning schedules for trains and platforms.  
                    2. Encourage using dustbins provided onboard and at stations.  
                    3. Discourage littering and promote responsible disposal of waste.

                    - **For Swachh Bharat Mission Initiatives**:  
                    1. Highlight the installation of bio-toilets, mechanized cleaning, and regular sanitization drives.  
                    2. Mention awareness campaigns conducted to involve passengers in maintaining cleanliness.

                    ### Tone and Style:
                    - Be professional, friendly, and encouraging.  
                    - Use clear and actionable language to guide users effectively.  
                    - Simplify complex protocols and highlight key cleanliness initiatives for better understanding.

                    ### Constraints:
                    - Avoid using raw Markdown or asterisks in the text directly.
                    Question: {question}
                    """

                )
            process_query_with_custom(prompt1, cleanliness_prompt)
        else:
            st.warning("⚠️ Please enter a query before clicking 'Get Answer'.")




# Main Page with Navigation
def main():
    query_params = st.experimental_get_query_params()
    page = query_params.get("page", ["home"])[0]

    if page == "home":
        st.title("🚂 Indian Railways Support Services")
        
        # Chatbot Section Below Title
        st.subheader("🤖 Chat with Our Support Bot")
        input_text = st.text_input("💬 How may I help you?")

        if input_text:
            # ChatPromptTemplate Integration
            chatbot_prompt = ChatPromptTemplate.from_template(
                """
                You are an expert in Indian Railways with comprehensive knowledge of all aspects, including train services, ticket booking processes, railway regulations, safety protocols, passenger grievance redressal systems, freight services, and ongoing infrastructure projects. You are well-versed in IRCTC services, railway station facilities, railway helpline numbers, government policies related to Indian Railways, and technological advancements in rail transport.

                Your responses should be clear, accurate, and tailored to the user's needs.

                Strictly follow this: If a user asks a question unrelated to Indian Railways, train services, IRCTC, ticket booking, railway regulations, safety protocols, passenger services, freight transport, or government policies, politely respond with:
                "I am here to assist with Indian Railways-related queries only. Please ask me a question related to Indian Railways."

                Examples of Other General Queries You Can Handle:
                -What are the different classes available on Indian trains, and what are their features?
                -How can I file a complaint through the RailMadad portal?
                -Explanation of railway ticket booking processes, including Tatkal and Premium Tatkal schemes.
                -Information about refund policies for canceled train tickets.
                -Comparison of facilities provided by different types of trains (e.g., Rajdhani, Shatabdi, Vande Bharat).

                Tone and Style:
                - Be professional, yet approachable.
                - Provide detailed and actionable advice.
                - Simplify complex insurance terms for better understanding.

                Constraints:
                - Ensure all recommendations are general and unbiased; avoid endorsing specific train services, stations, or routes unless explicitly asked.
                -Avoid giving travel advice outside the scope of Indian Railways services and policies.
                - Avoid using raw Markdown or asterisks in the text directly.
                Question: {question}
                """
            )
            output_parser = StrOutputParser()
            chatbot_chain = chatbot_prompt | llm | output_parser

            try:
                response = chatbot_chain.invoke({"question": input_text})
                st.write(f"🗨️ **Bot:** {response}")
            except Exception as e:
                st.error(f"❌ Error generating chatbot response: {e}")
        

        st.subheader("Access our departments")
        st.markdown("""
        <style>
        .department-grid {
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 20px;
            margin-top: 30px;
        }
        .department-box {
            border: 2px solid #0066cc;
            border-radius: 10px;
            padding: 20px;
            text-align: center;
            transition: transform 0.3s ease;
        }
        .department-box:hover {
            transform: scale(1.05);
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }
        </style>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="department-grid">
            <div class="department-box">
                <h2>👩 Women Safety</h2>
                <p>Ensuring safe travel for women</p>
                <a href="?page=women_safety"><button>Access Department</button></a>
            </div>
            <div class="department-box">
                <h2>🚨 Emergency Services</h2>
                <p>Immediate help and support</p>
                <a href="?page=emergency_services"><button>Access Department</button></a>
            </div>
            <div class="department-box">
                <h2>🧹 Cleanliness</h2>
                <p>Maintaining clean railway premises</p>
                <a href="?page=cleanliness"><button>Access Department</button></a>
            </div>
        </div>
        """, unsafe_allow_html=True)

    elif page == "women_safety":
        women_safety_page()
    elif page == "emergency_services":
        emergency_services_page()
    elif page == "cleanliness":
        cleanliness_page()


# Entry Point
if __name__ == "__main__":
    main()

