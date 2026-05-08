import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import json  # <-- Add
import os  # <-- Add
from dotenv import load_dotenv  # <-- Add

# <-- Add LangChain & Gemini Imports
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory

# <-- Load API Key
load_dotenv() # This loads the .env file

# Function to load the trained model and make a prediction
# (This function is from your original file)
def model_prediction(test_image):
    model = tf.keras.models.load_model("trained_plant_disease_model.keras")
    image = tf.keras.preprocessing.image.load_img(test_image, target_size=(128, 128))
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr])  # Convert single image to a batch
    predictions = model.predict(input_arr)
    return np.argmax(predictions)  # Return index of the class with the highest probability

# <-- New function to load disease information
@st.cache_data
def load_disease_info():
    """Loads the disease information from the JSON file."""
    try:
        with open("disease_info.json", "r") as f:
            return json.load(f)
    except FileNotFoundError:
        st.error("Error: `disease_info.json` file not found.")
        st.stop()
    except json.JSONDecodeError:
        st.error("Error: `disease_info.json` is not a valid JSON file.")
        st.stop()

# Sidebar
# (This section is mostly from your original file)
st.sidebar.title("Plant Disease Detection System")
app_mode = st.sidebar.selectbox("Select Page", ["🏠 HOME", "🔍 DISEASE RECOGNITION", "ℹ️ ABOUT"])

# Main Pages
# (This section is from your original file)
if app_mode == "🏠 HOME":
    st.markdown("<h1 style='text-align: center;'>Plant Disease Detection System for Sustainable Agriculture</h1>", unsafe_allow_html=True)
    
    st.markdown("""
    Welcome to the **Plant Disease Detection System**! 🌱
    This application helps farmers and agricultural enthusiasts identify plant diseases from images. 
    Our system uses a powerful deep learning model to accurately recognize diseases in various crops,
    helping you take timely action to protect your plants and improve crop yield.
    """)
    
    try:
        img = Image.open("Diseases.png")
        st.image(img, caption="Our system can detect diseases in various plants.")
    except FileNotFoundError:
        st.warning("Could not find 'Diseases.png'. Please ensure the image is in the project directory.")
    
    st.info("""
    ### How It Works
    1. **Navigate** to the **'DISEASE RECOGNITION'** page using the sidebar.
    2. **Upload** an image of a plant leaf.
    3. **Get Instant Results**: Our model will analyze the image, identify the disease, provide symptoms, and offer treatment advice.
    4. **Ask an Expert**: Use the integrated chatbot for follow-up questions.
    """)
    
    st.markdown("---")
    
    st.markdown("""
    ### Why is this important?
    Early detection of plant diseases is crucial for sustainable agriculture. By using this tool, you can:
    * **Reduce crop loss** and increase productivity.
    * **Minimize the use of harmful pesticides**, promoting a healthier environment.
    * **Save time and resources** that would otherwise be spent on guesswork.
    """)

# (This page is heavily modified)
elif app_mode == "🔍 DISEASE RECOGNITION":
    st.header("🔍 Detect Plant Disease")
    st.markdown("Upload an image of a plant leaf, and our system will tell you its health status.")
    
    # Image upload
    test_image = st.file_uploader("Choose an Image:", type=["jpg", "jpeg", "png"])
    
    if test_image is not None:
        # Display the uploaded image
        st.image(test_image, caption="Uploaded Image")
        
        # Create a button for prediction
        if st.button("Predict"):
            
            # <-- Clear old chat history on new prediction
            st.session_state.messages = []
            st.session_state.memory = ConversationBufferMemory(return_messages=True)
            if 'chain' in st.session_state:
                del st.session_state.chain
            
            st.write("---")
            with st.spinner("Analyzing image..."):
                try:
                    # Reading Labels
                    class_name = ['Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
                                'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew', 
                                'Cherry_(including_sour)___healthy', 'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 
                                'Corn_(maize)___Common_rust_', 'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy', 
                                'Grape___Black_rot', 'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 
                                'Grape___healthy', 'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot',
                                'Peach___healthy', 'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 
                                'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy', 
                                'Raspberry___healthy', 'Soybean___healthy', 'Squash___Powdery_mildew', 
                                'Strawberry___Leaf_scorch', 'Strawberry___healthy', 'Tomato___Bacterial_spot', 
                                'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold', 
                                'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite', 
                                'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus',
                                'Tomato___healthy']
                    
                    # <-- Load the disease info from JSON
                    disease_info_db = load_disease_info()
                    
                    result_index = model_prediction(test_image) #
                    
                    st.success("Analysis Complete!")
                    
                    # Displaying the result in a more user-friendly way
                    prediction = class_name[result_index]
                    
                    # <-- NEW SECTION: Display Static Info ---
                    if prediction not in disease_info_db:
                        st.error(f"Details for '{prediction}' not found in disease_info.json. Please update the file.")
                        # Still store it so the chatbot can *try*
                        st.session_state.disease_detected = prediction
                        st.session_state.disease_info = {"symptoms": "Not available in database.", "treatment": "Not available in database."}
                    else:
                        info = disease_info_db[prediction]
                        
                        if 'healthy' in prediction.lower():
                            st.balloons()
                            st.markdown(f"**Prediction:** This plant is likely **{prediction.split('___')[1].replace('_', ' ').replace('(including sour)', '').strip()}**! 🌱")
                        else:
                            st.warning(f"**Prediction:** This plant is likely suffering from **{prediction.split('___')[1].replace('_', ' ').replace('(including sour)', '').strip()}**.")
                            st.markdown("### 📋 Disease Information")
                        
                        st.markdown(f"**Symptoms:** {info['symptoms']}")
                        st.markdown(f"**Suggested Treatment:** {info['treatment']}")
                        
                        # <-- Store prediction for the chatbot
                        st.session_state.disease_detected = prediction
                        st.session_state.disease_info = info
                        
                except Exception as e:
                    st.error(f"An error occurred during prediction: {e}")
                    st.error("Please check if 'trained_plant_disease_model.keras' and 'disease_info.json' are in the project directory.")

    # --- NEW: CHATBOT SECTION ---
    st.write("---")
    st.header("💬 Ask an Expert")
    
    # Check if a prediction has been made
    if 'disease_detected' not in st.session_state:
        st.markdown("Upload and predict an image first to activate the expert chatbot.")
    else:
        disease = st.session_state.disease_detected.split('___')[1].replace('_', ' ')
        info = st.session_state.disease_info

        # Initialize session state for memory and messages
        if "memory" not in st.session_state:
            st.session_state.memory = ConversationBufferMemory(return_messages=True)
        if "messages" not in st.session_state:
            st.session_state.messages = []

        # Define the expert prompt template
        template = f"""
        You are a helpful and empathetic botanist and plant disease expert. You are assisting a user who has just identified a plant disease using an AI model.
        
        The detected disease is: {disease}
        Known symptoms are: {info['symptoms']}
        The suggested general treatment is: {info['treatment']}

        Your role is to have a natural, conversational dialogue. Do not just repeat the information above. 
        Start by introducing yourself and ask an open-ended follow-up question to get more context about their specific situation.
        For example: "Hello, I see your plant might have {disease}. To help you better, could you tell me a bit more about what you're seeing? For example, how long have the symptoms been present?"
        
        Based on the user's answers, provide more detailed, actionable advice. Be like a real human expert.

        Current conversation:
        {{history}}
        Human: {{input}}
        AI:
        """
        
        PROMPT = PromptTemplate(input_variables=["history", "input"], template=template)

        # Initialize the ConversationChain in session state
        if "chain" not in st.session_state:
            try:
                llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.7)
                st.session_state.chain = ConversationChain(
                    llm=llm,
                    prompt=PROMPT,
                    memory=st.session_state.memory,
                    verbose=True
                )
                
                # <-- Add an initial greeting from the bot
                initial_bot_message = st.session_state.chain.run(input="Hello")
                st.session_state.messages.append({"role": "assistant", "content": initial_bot_message})

            except Exception as e:
                st.error(f"Error initializing chatbot. Is your GOOGLE_API_KEY correct? Error: {e}")
                st.stop()


        # Display chat history
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        # Chat input
        if user_prompt := st.chat_input(f"Ask me about {disease}..."):
            # Add user message to history and display
            st.session_state.messages.append({"role": "user", "content": user_prompt})
            with st.chat_message("user"):
                st.markdown(user_prompt)
            
            # Get AI response
            with st.spinner("Expert is thinking..."):
                try:
                    response = st.session_state.chain.run(user_prompt)
                    # Add AI response to history and display
                    st.session_state.messages.append({"role": "assistant", "content": response})
                    with st.chat_message("assistant"):
                        st.markdown(response)
                except Exception as e:
                    st.error(f"An error occurred with the chatbot: {e}")

# (This section is from your original file)
elif app_mode == "ℹ️ ABOUT":
    st.header("About This Project")
    st.markdown("""
    This project was developed to provide an accessible and efficient tool for plant disease detection using deep learning. 
    It leverages a convolutional neural network (CNN) trained on a large dataset of plant leaves.
    
    **Our Goal:** To support sustainable agriculture by empowering farmers with a tool for early disease detection, reducing crop loss, and minimizing environmental impact.
    
    **Developed by:** [Your Name/Team Name]
    
    **Technology Stack:**
    - **Frontend:** Streamlit
    - **Backend:** TensorFlow, Keras
    - **Chatbot:** Google Gemini & LangChain
    - **Libraries:** NumPy, PIL
    """)
    st.markdown("---")
    st.header("Contact Us")
    st.markdown("""
    If you have any questions or feedback, please feel free to reach out.
    * **Email:** your.email@example.com
    * **GitHub:** [Link to your GitHub repository]
    """)