# app.py
import streamlit as st
import pandas as pd
import os
import random
from datetime import datetime
import tempfile

# Import functions from your logic file
from excuse_generator_logic import (
    initialize_model_and_tokenizer,
    load_dataframes_from_objects,
    generate_response,
    generate_voice_output_st,
    generate_whatsapp_screenshot_st,
    generate_location_context,
    trigger_fake_emergency,
    add_to_history_df,
    save_history_to_csv,
    toggle_favorite_in_df,
    record_feedback_in_df,
    get_ranked_suggestion_list,
    BASE_MODEL_ID, # Constant
    FINE_TUNED_ADAPTER_DIR, # Constant
    RECIPIENT_REPLIES, # Constant
    DEFAULT_FONT_SIZE_MSG, # Constant
    DEFAULT_FONT_SIZE_INFO, # Constant
    HISTORY_DF_COLUMNS
)

# --- Page Configuration ---
st.set_page_config(page_title="Excuse Generator AI", layout="wide", initial_sidebar_state="expanded")

# --- Session State Initialization ---
if 'model' not in st.session_state: st.session_state.model = None
if 'tokenizer' not in st.session_state: st.session_state.tokenizer = None
if 'device' not in st.session_state: st.session_state.device = "cpu"
if 'is_model_fine_tuned' not in st.session_state: st.session_state.is_model_fine_tuned = False
if 'font_path_streamlit' not in st.session_state: st.session_state.font_path_streamlit = None
if 'combined_df_for_context' not in st.session_state: st.session_state.combined_df_for_context = pd.DataFrame()
if 'history_df' not in st.session_state: st.session_state.history_df = pd.DataFrame(columns=HISTORY_DF_COLUMNS)
if 'history_id_counter' not in st.session_state: st.session_state.history_id_counter = 0
if 'last_generation' not in st.session_state: st.session_state.last_generation = {} # Stores details of the most recent generation
if 'ui_messages' not in st.session_state: st.session_state.ui_messages = [] # For status messages

# Temporary directory for proofs
TEMP_PROOF_DIR = tempfile.mkdtemp(prefix="excusegen_proofs_")

# --- Helper Functions for UI ---
def add_ui_message(message, type="info"):
    st.session_state.ui_messages.append({"message": message, "type": type, "time": datetime.now()})

def display_ui_messages():
    # Display only recent messages or clear them periodically if desired
    for msg_info in st.session_state.ui_messages[-5:]: # Show last 5
        if msg_info["type"] == "error": st.error(msg_info["message"])
        elif msg_info["type"] == "warning": st.warning(msg_info["message"])
        elif msg_info["type"] == "success": st.success(msg_info["message"])
        else: st.info(msg_info["message"])
    if len(st.session_state.ui_messages) > 10: # Prune old messages
        st.session_state.ui_messages = st.session_state.ui_messages[-5:]


# --- Sidebar for Setup & Configuration ---
with st.sidebar:
    st.header("⚙️ Setup & Configuration")
    st.markdown("---")

    # Hugging Face Token
    hf_token_input = st.text_input("Hugging Face Token (Optional)", type="password", help="Needed for private models or to avoid rate limits.")
    st.session_state.hf_token_input = hf_token_input # Store for use in button callback

    st.markdown("---")
    st.subheader("📤 Upload Files")
    uploaded_excuse_csv = st.file_uploader("Excuse Dataset (CSV)", type="csv", key="excuse_csv")
    uploaded_apology_csv = st.file_uploader("Apology Dataset (CSV)", type="csv", key="apology_csv")
    uploaded_history_csv = st.file_uploader("Load Previous History (CSV, Optional)", type="csv", key="history_csv_upload")
    uploaded_font_file = st.file_uploader("Custom Font File (.ttf, Optional)", type="ttf", key="font_upload")

    if st.button("🔄 Process Uploaded Files", key="process_files_btn"):
        with st.spinner("Processing files..."):
            if uploaded_font_file:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".ttf", dir=TEMP_PROOF_DIR) as tmp_font:
                    tmp_font.write(uploaded_font_file.getvalue())
                    st.session_state.font_path_streamlit = tmp_font.name
                add_ui_message(f"Font '{uploaded_font_file.name}' ready.", "success")
            else:
                st.session_state.font_path_streamlit = None # Ensure it's reset if no file
                add_ui_message("No custom font uploaded. Default will be used for images.", "warning")

            # Load datasets and history
            st.session_state.combined_df_for_context, st.session_state.history_df, st.session_state.history_id_counter = \
                load_dataframes_from_objects(uploaded_excuse_csv, uploaded_apology_csv, uploaded_history_csv)

            if not st.session_state.combined_df_for_context.empty:
                add_ui_message(f"Excuse/Apology datasets loaded: {len(st.session_state.combined_df_for_context)} entries.", "success")
            else:
                add_ui_message("No excuse/apology data loaded. Fine-tuning context might be limited.", "warning")
            if not st.session_state.history_df.empty:
                add_ui_message(f"History loaded: {len(st.session_state.history_df)} entries. Next ID: {st.session_state.history_id_counter}", "success")


    st.markdown("---")
    if st.button("🚀 Initialize AI Model", key="init_model_btn"):
        if st.session_state.model is None: # Only initialize if not already done
            with st.spinner("Initializing AI model... This may take a few minutes."):
                model, tokenizer, is_ft, device = initialize_model_and_tokenizer(
                    model_id=BASE_MODEL_ID,
                    adapter_dir=FINE_TUNED_ADAPTER_DIR, # Make sure this path is accessible
                    hf_token=st.session_state.get("hf_token_input")
                )
                if model and tokenizer:
                    st.session_state.model = model
                    st.session_state.tokenizer = tokenizer
                    st.session_state.is_model_fine_tuned = is_ft
                    st.session_state.device = device
                    add_ui_message(f"AI Model initialized successfully on {device}. Fine-tuned: {is_ft}", "success")
                else:
                    add_ui_message("AI Model initialization FAILED. Check logs.", "error")
        else:
            add_ui_message("AI Model already initialized.", "info")

    st.markdown("---")
    if st.session_state.model:
        st.success(f"✅ Model Loaded ({st.session_state.device})")
        if st.session_state.is_model_fine_tuned: st.info("Fine-tuned adapter active.")
    else:
        st.warning("Model not yet initialized.")

    if not st.session_state.combined_df_for_context.empty:
        st.success("✅ Context Data Loaded")
    else:
        st.info("Context data not loaded.")


# --- Main Application Area ---
st.title("🧠 Intelligent Excuse & Apology Generator")
display_ui_messages() # Display status messages

if not st.session_state.model or not st.session_state.tokenizer:
    st.warning("👈 Please initialize the AI model using the button in the sidebar first.")
else:
    tab1, tab2, tab3 = st.tabs(["💬 Generate New", "📜 History", "⚡ Actions"])

    with tab1:
        st.header("📝 Generate New Message")
        with st.form("generation_form"):
            c1, c2 = st.columns(2)
            with c1:
                situation = st.text_input("Situation (e.g., 'late for meeting')", key="situation_input")
                priority = st.selectbox("Priority", ["low", "medium", "high"], key="priority_input", index=1)
            with c2:
                plausibility = st.selectbox("Plausibility", ["low", "medium", "high"], key="plausibility_input", index=1)
                msg_type = st.selectbox("Message Type", ["excuse", "apology"], key="msg_type_input")

            user_context = st.text_area("Additional Context/Reason (be specific for better results)", key="user_context_input", height=100)
            submit_button = st.form_submit_button("✨ Generate Message")

        if submit_button:
            if not situation or not user_context:
                add_ui_message("'Situation' and 'Context/Reason' cannot be empty.", "error")
            else:
                with st.spinner("AI is thinking..."):
                    # 1. Get Ranked Suggestions
                    ranked_suggestions = get_ranked_suggestion_list(
                        st.session_state.history_df, situation, priority, plausibility, msg_type
                    )
                    if ranked_suggestions:
                        st.subheader("💡 Previously Rated Suggestions")
                        for sug in ranked_suggestions:
                            st.markdown(f"- \"{sug['text']}\" *(Rated: {sug['rating']})*")
                        st.markdown("---")

                    # 2. Generate New Message
                    prompt = (
                        f"Generate a short, informal text message style {msg_type} for the situation: '{situation}'. "
                        f"The specific reason or context is: '{user_context}'. "
                        f"The message should sound like it has {plausibility} plausibility and {priority} urgency. "
                        f"Keep it concise (1-2 sentences typically). Avoid formal greetings or sign-offs. "
                        f"Output ONLY the message text itself."
                    )
                    generated_text = generate_response(st.session_state.model, st.session_state.tokenizer, prompt, st.session_state.device)

                    if generated_text and not generated_text.startswith("Error:") and generated_text != "[Empty Response]":
                        current_id_for_history = st.session_state.history_id_counter
                        st.session_state.last_generation = {
                            "id": current_id_for_history,
                            "situation": situation, "priority": priority, "plausibility": plausibility,
                            "msg_type": msg_type, "user_context": user_context, "generated_text": generated_text
                        }
                        # Add to history
                        history_entry_data = {
                            'timestamp': datetime.now(), 'situation': situation, 'priority': priority,
                            'plausibility': plausibility, 'message_type': msg_type, 'user_context': user_context,
                            'generated_text': generated_text, 'effectiveness_rating': np.nan, 'is_favorite': False
                        }
                        st.session_state.history_df, st.session_state.history_id_counter = add_to_history_df(
                            st.session_state.history_df, history_entry_data, current_id_for_history
                        )
                        add_ui_message(f"Generated new {msg_type} (ID: {current_id_for_history}).", "success")

                        # 3. Generate Proofs (asynchronously if possible, but Streamlit runs sequentially)
                        st.session_state.last_generation["audio_file_path"] = generate_voice_output_st(generated_text, output_dir=TEMP_PROOF_DIR)
                        st.session_state.last_generation["whatsapp_img_path"] = generate_whatsapp_screenshot_st(
                            generated_text, random.choice(RECIPIENT_REPLIES),
                            st.session_state.font_path_streamlit,
                            DEFAULT_FONT_SIZE_MSG, DEFAULT_FONT_SIZE_INFO, output_dir=TEMP_PROOF_DIR
                        )
                        st.session_state.last_generation["location_context"] = generate_location_context(
                            st.session_state.model, st.session_state.tokenizer, generated_text, user_context, st.session_state.device
                        )
                    else:
                        add_ui_message(f"Message generation failed: {generated_text}", "error")
                        st.session_state.last_generation = {} # Clear previous if failed
        # --- Display Last Generation Results & Feedback ---
        if st.session_state.last_generation.get("generated_text"):
            lg = st.session_state.last_generation
            st.divider()
            st.subheader(f"💬 Generated {lg['msg_type'].capitalize()} (ID: {lg['id']})")
            st.markdown(f"##### **Message:**\n> {lg['generated_text']}")

            proof_cols = st.columns(3)
            with proof_cols[0]:
                if lg.get("audio_file_path") and os.path.exists(lg["audio_file_path"]):
                    st.caption("🎤 Voice Note:")
                    st.audio(lg["audio_file_path"])
                else: st.caption("🎤 Voice Note: (failed or N/A)")
            with proof_cols[1]:
                if lg.get("whatsapp_img_path") and os.path.exists(lg["whatsapp_img_path"]):
                    st.caption("📱 WhatsApp Proof:")
                    st.image(lg["whatsapp_img_path"], width=300) # Adjusted width
                else: st.caption("📱 WhatsApp Proof: (failed or N/A)")
            with proof_cols[2]:
                if lg.get("location_context"):
                    st.caption("📍 Location Context:")
                    st.text_area("", value=lg["location_context"], height=100, disabled=True, key=f"loc_{lg['id']}")
                else: st.caption("📍 Location Context: (failed or N/A)")

            st.markdown("---")
            st.markdown("**Rate this generation (ID: {}):**".format(lg['id']))
            fb_cols = st.columns([1,3,1.5]) # Favorite, Slider, Button
            with fb_cols[0]:
                is_fav = st.session_state.history_df.loc[st.session_state.history_df['id'] == lg['id'], 'is_favorite'].iloc[0] if lg['id'] in st.session_state.history_df['id'].values else False
                if st.button("❤️" if is_fav else "🤍", key=f"fav_btn_{lg['id']}", help="Toggle Favorite"):
                    st.session_state.history_df = toggle_favorite_in_df(st.session_state.history_df, lg['id'])
                    st.experimental_rerun() # Rerun to update button icon and history display
            with fb_cols[1]:
                current_rating_val = st.session_state.history_df.loc[st.session_state.history_df['id'] == lg['id'], 'effectiveness_rating'].iloc[0] if lg['id'] in st.session_state.history_df['id'].values and pd.notna(st.session_state.history_df.loc[st.session_state.history_df['id'] == lg['id'], 'effectiveness_rating'].iloc[0]) else 5
                rating_input = st.slider("Effectiveness (0-10)", 0, 10, value=int(current_rating_val), key=f"rate_slider_{lg['id']}")
            with fb_cols[2]:
                st.write("") # Spacer
                if st.button("Submit Rating", key=f"submit_rating_btn_{lg['id']}"):
                    st.session_state.history_df = record_feedback_in_df(st.session_state.history_df, lg['id'], rating_input)
                    add_ui_message(f"Rating {rating_input}/10 saved for ID {lg['id']}.", "success")
                    st.experimental_rerun()

    with tab2:
        st.header("📜 Generation History")
        if not st.session_state.history_df.empty:
            history_display_df = st.session_state.history_df.copy()
            history_display_df['timestamp'] = history_display_df['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')
            history_display_df['effectiveness_rating'] = history_display_df['effectiveness_rating'].apply(lambda x: f"{int(x)}/10" if pd.notna(x) else "N/A")
            st.dataframe(history_display_df[['id', 'timestamp', 'situation', 'message_type', 'generated_text', 'effectiveness_rating', 'is_favorite']].sort_values(by="timestamp", ascending=False), height=400)

            export_cols = st.columns(3)
            with export_cols[0]:
                if st.button("💾 Save Current History to CSV", key="save_hist_btn"):
                    saved_path = os.path.join(TEMP_PROOF_DIR, f"excuse_generator_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
                    if save_history_to_csv(st.session_state.history_df, saved_path):
                        add_ui_message(f"History saved to {saved_path}", "success")
                        # Provide download link
                        with open(saved_path, "rb") as fp:
                            st.download_button(
                                label="⬇️ Download History CSV",
                                data=fp,
                                file_name=os.path.basename(saved_path),
                                mime="text/csv"
                            )
                    else:
                        add_ui_message("Failed to save history.", "error")
        else:
            st.info("No history yet. Generate some messages!")

    with tab3:
        st.header("⚡ Other Actions")
        if st.button("🚨 Trigger Fake Emergency Simulation", key="emergency_btn"):
            with st.spinner("Simulating emergency..."):
                emergency_text = trigger_fake_emergency(st.session_state.model, st.session_state.tokenizer, st.session_state.device)
                st.info(emergency_text)

        st.markdown("---")
        st.subheader("🗑️ Clear Temporary Files")
        st.caption(f"Proof files are stored temporarily in: {TEMP_PROOF_DIR}")
        if st.button("Clear Proofs Directory", help="Deletes all files in the temporary proofs directory."):
            cleared_count = 0
            try:
                for filename in os.listdir(TEMP_PROOF_DIR):
                    file_path = os.path.join(TEMP_PROOF_DIR, filename)
                    if os.path.isfile(file_path):
                        os.unlink(file_path)
                        cleared_count +=1
                add_ui_message(f"Cleared {cleared_count} files from temporary proofs directory.", "success")
            except Exception as e:
                add_ui_message(f"Error clearing proofs directory: {e}", "error")


# --- Footer ---
st.markdown("---")
st.caption("Excuse Generator AI - Alpha Version")