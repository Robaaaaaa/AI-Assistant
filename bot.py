import streamlit as st
from transformers import LlamaForCausalLM, LlamaTokenizer

st.title("Blog  AI")

# Set up the API key from Streamlit secrets or environment variables
api_key = None
tokenizer = LlamaTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B")

# Check if the API key is provided via Streamlit secrets
if 'HUGGINGFACE_API_TOKEN' in st.secrets:
    api_key = st.secrets['HUGGINGFACE_API_TOKEN']
else:
    # If not, allow the user to input the API key manually
    with st.sidebar:
        api_key = st.text_input('Enter Hugging Face API token:', type='password')

        # Basic validation for the API key
        if api_key:
            if not (api_key.startswith('hf_')):
                st.error('Please enter a valid API token!')
            else:
                st.success('Proceed to entering your prompt message!', icon='👉🏿')

# Store generated responses
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "What do you wish to know about?"}]

# Display/Clear chat messages
for i, message in enumerate(st.session_state.messages):
    if message["role"] == "user":
        st.text_input("Your message:", value=message["content"], disabled=True, key=f"user_message_{i}")
    elif message["role"] == "assistant":
        st.write(message["content"], key=f"assistant_message_{i}")

def clear_chat_history():
    st.session_state.messages = [{"role": "assistant", "content": "What do you wish to know about ?"}]
    st.experimental_rerun()

st.sidebar.button('Clear Chat History', on_click=clear_chat_history)

# Function to generate responses using Hugging Face
@st.cache
def generate_response(prompt):
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(
        inputs.input_ids,
        max_length=200,
        temperature=0.7,
        top_p=0.85,
        repetition_penalty=1.2
    )
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

# Load the model and tokenizer
@st.cache
def load_model():
    model_name = "meta-llama/Meta-Llama-3-8B"  # Correct model name
    tokenizer = LlamaTokenizer.from_pretrained(model_name)
    model = LlamaForCausalLM.from_pretrained(model_name)
    return model, tokenizer

model, tokenizer = load_model()

# User prompt
if prompt := st.chat_input(disabled=not api_key):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

    # Generate new response if the last message was from the user
    with st.chat_message("assistant"):
        with st.spinner("Preparing your answer..."):
            response = generate_response(prompt)
            st.write(response)
            st.session_state.messages.append({"role": "assistant", "content": response})
