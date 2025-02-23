import streamlit as st
import requests
import uuid
import openai
from datetime import datetime
import json
from typing import Dict, List, Optional
import pandas as pd

# Configuration
BACKEND_URL = "http://localhost:8080"

# Page configuration
st.set_page_config(
    page_title="AI Chatbot Setup",
    page_icon="🤖",
    layout="wide"
)

# Custom CSS
st.markdown("""
    <style>
    .stButton button {
        width: 100%;
        border-radius: 5px;
        height: 3em;
    }
    .success-box {
        padding: 1em;
        border-radius: 5px;
        border: 1px solid #28a745;
        background-color: #d4edda;
        color: #155724;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if 'step' not in st.session_state:
    st.session_state.step = 'user_registration'
    st.session_state.user_id = str(uuid.uuid4())
    st.session_state.messages = []
    st.session_state.products = []

def make_request(endpoint: str, method: str = "GET", data: Optional[Dict] = None) -> Dict:
    """Helper function to make HTTP requests to backend"""
    try:
        url = f"{BACKEND_URL}/{endpoint}"
        if method == "GET":
            response = requests.get(url)
        else:
            response = requests.post(url, json=data)
        
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Error communicating with backend: {str(e)}")
        if hasattr(e.response, 'text'):
            st.error(f"Backend error details: {e.response.text}")
        return None

def display_products(products: List[Dict]):
    """Display products in a grid layout"""
    if not products:
        st.info("No products found. Products will appear here after synchronization is complete.")
        return

    # Convert to DataFrame for easier handling
    df = pd.DataFrame(products)
    
    # Display products in a grid
    cols = st.columns(3)
    for idx, product in enumerate(products):
        with cols[idx % 3]:
            st.subheader(product['title'])
            if product.get('images') and len(product['images']) > 0:
                st.image(product['images'][0]['src'], use_column_width=True)
            
            # Display variant information
            if product.get('variants'):
                variant = product['variants'][0]  # Display first variant
                st.write(f"💰 ${variant['price']}")
                st.write(f"📦 Stock: {variant['inventory_quantity']}")
            
            # Truncate description
            if product.get('description'):
                description = product['description']
                if len(description) > 100:
                    description = description[:100] + "..."
                st.write(description)
            
            st.divider()
def handle_chat(prompt: str):
    """Handle chat interaction with backend"""
    try:
        # Add user message to state
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Get response from backend
        response = requests.post(
            f"{BACKEND_URL}/chat/{st.session_state.user_id}",
            json={"query": prompt, "user_id": st.session_state.user_id},
            timeout=10  # Add timeout
        )
        
        if response.status_code == 200:
            data = response.json()
            
            # Format the response based on type
            if data.get("type") == "product":
                message = "🛍️ " + data["message"] + "\n\n"
                if data.get("results"):
                    message += "**Available Products:**\n\n"
                    for product in data["results"]:
                        message += f"""**{product['title']}**
                        💰 ${product['price']}
                        {product['description'][:100]}...\n\n"""
            elif data.get("type") == "support":
                message = "🎯 " + data["message"] + "\n\n"
                message += f"📞 Support Link: {data.get('support_link', '')}\n"
                if data.get('meeting_link'):
                    message += f"📅 Schedule Meeting: {data['meeting_link']}"
            else:
                message = data["message"]
            
            # Add assistant message to state
            st.session_state.messages.append({"role": "assistant", "content": message})
        else:
            error_message = f"Error: Server returned status code {response.status_code}"
            st.session_state.messages.append({"role": "assistant", "content": error_message})
            st.error(error_message)
            
    except requests.exceptions.RequestException as e:
        error_message = f"Error connecting to server: {str(e)}"
        st.session_state.messages.append({"role": "assistant", "content": error_message})
        st.error(error_message)
    except Exception as e:
        error_message = f"Unexpected error: {str(e)}"
        st.session_state.messages.append({"role": "assistant", "content": error_message})
        st.error(error_message)

# Main App Logic
st.title("AI Chatbot Setup")

# Step 1: User Registration
if st.session_state.step == 'user_registration':
    st.header("Create Your Account")
    
    with st.form("registration_form"):
        company_name = st.text_input("Company Name*")
        user_name = st.text_input("Your Name*")
        email = st.text_input("Email*")
        
        submit = st.form_submit_button("Create Account")
        
        if submit:
            if company_name and user_name and email:
                user_data = {
                    "user_id": st.session_state.user_id,
                    "company_name": company_name,
                    "user_name": user_name,
                    "email": email
                }
                
                if make_request("setup/user", method="POST", data=user_data):
                    st.session_state.step = 'onboarding'
                    st.rerun()
            else:
                st.error("Please fill in all required fields")

# Step 2: Onboarding
elif st.session_state.step == 'onboarding':
    st.header("Bot Configuration")
    
    with st.form("onboarding_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            industry = st.text_input("Industry*")
            goal = st.text_area("What's the main goal for your chatbot?*")
            general_info = st.text_area("Tell us about your company")

        with col2:
            support_link = st.text_input("Customer Support Link*")
            meeting_link = st.text_input("Meeting/Calendar Link (optional)")
            
        # Shopify Details
        st.subheader("Shopify Integration")
        shopify_store = st.text_input("Shopify Store Name* (e.g., your-store.myshopify.com)")
        shopify_token = st.text_input("Shopify Access Token*", type="password")
        
        submit = st.form_submit_button("Start Bot Setup")
        
        if submit:
            if industry and goal and support_link and shopify_store and shopify_token:
                with st.spinner("Setting up your bot and syncing products..."):
                    onboarding_data = {
                        "user_id": st.session_state.user_id,
                        "industry": industry,
                        "goal": goal,
                        "support_link": support_link,
                        "meeting_link": meeting_link,
                        "general_info": general_info,
                        "shopify_store": shopify_store,
                        "shopify_token": shopify_token
                    }
                    
                    if make_request("setup/onboarding", method="POST", data=onboarding_data):
                        st.session_state.step = 'complete'
                        st.rerun()
            else:
                st.error("Please fill in all required fields")

# Step 3: Complete
elif st.session_state.step == 'complete':
    st.success("🎉 Setup Complete!")
    
    tab1, tab2, tab3 = st.tabs(["Integration", "Products", "Chat Preview"])
    
    with tab1:
        st.header("Integration Options")
        
        # Widget Code
        st.subheader("Add to Your Website")
        widget_code = f"""
        <!-- AI Chatbot Widget -->
        <script>
            window.AIChatbot = {{
                widgetId: "{st.session_state.user_id}",
                theme: "light"
            }};
        </script>
        <script src="https://your-domain.com/chatbot-widget.js"></script>
        """
        st.code(widget_code, language="html", line_numbers=True)
        
        # API Documentation
        st.subheader("API Integration")
        st.markdown("""
        **Base URL**: `https://your-domain.com/api`
        
        **Authentication Header**:
        ```
        Authorization: Bearer YOUR_USER_ID
        ```
        
        **Available Endpoints**:
        - `POST /chat`: Send chat messages
        - `GET /products`: Get product catalog
        """)
    
    with tab2:
        st.header("Products")
        
        # Sync button
        if st.button("🔄 Sync Products"):
            with st.spinner("Syncing products..."):
                try:
                    response = requests.post(
                        f"{BACKEND_URL}/sync/{st.session_state.user_id}"
                    )
                    
                    if response.status_code == 200:
                        result = response.json()
                        st.success(f"Successfully synced {result['sync_results']['processed']} products!")
                        
                        # Refresh products display
                        st.experimental_rerun()
                    else:
                        st.error(f"Error syncing products: {response.json().get('detail', 'Unknown error')}")
                        
                except Exception as e:
                    st.error(f"Error during sync: {str(e)}")
                    if hasattr(e, 'response'):
                        st.error(f"Backend error details: {e.response.text}")
        
        # Display products
        products = make_request(f"products/{st.session_state.user_id}")
        if products is not None:
            display_products(products)
    
    with tab3:
        st.header("Chat Preview")
        
        # Display chat history
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

    # Chat input - moved outside tabs to fix StreamlitAPIException
    if st.session_state.step == 'complete':  # Only show chat input when setup is complete
        if prompt := st.chat_input("Ask something..."):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)
            
            handle_chat(prompt)

if __name__ == "__main__":
    st.sidebar.title("About")
    st.sidebar.info("""
    This is the setup interface for your AI Chatbot.
    
    Need help? Contact support at:
    support@your-domain.com
    """)