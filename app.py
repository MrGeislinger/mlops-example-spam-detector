import streamlit as st
import time
import logging
from joblib import load

logger = logging.getLogger(__name__)

@st.cache(suppress_st_warning=True)
def load_model(model_fname):
    #TODO: Create model if one doesn't exist
    model = load(model_fname)
    return model

def clean_message(message: str):
    message_lowercase = message.lower()
    return message_lowercase 

###### UI

st.title('Spam Detection')

# Enter potential spam message to submit
with st.form('message_form'):
    st.write('Enter potential spam message')
    message = st.text_input(label='Message')
    submitted = st.form_submit_button('Submit')
    
    placeholder_message = st.empty()

    # Predict from model
    if submitted:
        # Reset the screen's text
        placeholder_message.text('')
        time.sleep(0.1)

        with placeholder_message.container():
             # load model
            st.write('Loading model & preparing to make prediction...')
            model = load_model('model.joblib')
            time.sleep(0.5)
            
            # Prepare data
            message_clean = clean_message(message)
            st.write('Data prepped and making prediction now...')
            
            # Predict
            prediction = model.predict([message_clean])
            time.sleep(0.5)
            # Clean out response
            spam_or_ham = 'Spam' if prediction[0] else 'Ham'
            st.write(f'# That looks like it is a **{spam_or_ham}** message!')
            # Internal logging
            logger.info(
                f'{prediction=} for (cleaned) message: {message_clean!r}'
            )