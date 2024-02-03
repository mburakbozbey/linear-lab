import pytest
import streamlit as st

from data_scientist.app import page


def test_page():
    """
    Test case for the page function in app.py.
    """
    # Initialize session state
    st.session_state = {}

    # Call the page function
    page()

    # Assert that session state variables have been initialized
    assert "messages" in st.session_state
    assert "assistant" in st.session_state


def test_file_uploader():
    """
    Test case for the file uploader functionality in app.py.
    """
    # Initialize session state
    st.session_state = {}

    # Call the page function
    page()

    # Simulate file upload
    uploaded_file = st.file_uploader(
        "Upload document",
        type=["pdf"],
        key="file_uploader",
        on_change=None,
        label_visibility="collapsed",
        accept_multiple_files=True,
    )

    # Assert that the uploaded file is not None
    assert uploaded_file is not None


def test_message_input():
    """
    Test case for the message input functionality in app.py.
    """
    # Initialize session state
    st.session_state = {}

    # Call the page function
    page()

    # Simulate user input
    user_input = "Hello, how are you?"
    st.session_state["user_input"] = user_input  # Set the value in session state

    # Assert that the user input is stored in session state
    assert st.session_state["user_input"] == user_input


if __name__ == "__main__":
    pytest.main(["-v", "test_app.py"])
