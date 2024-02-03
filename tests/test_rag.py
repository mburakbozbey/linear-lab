import pytest

from data_scientist.rag import ChatPDF


@pytest.fixture
def chat_pdf():
    return ChatPDF()


def test_chat_pdf_ingest(chat_pdf, mocker):
    # Mock the PyPDFLoader and FastEmbedEmbeddings
    mocker.patch("data_scientist.rag.PyPDFLoader")
    mocker.patch("data_scientist.rag.FastEmbedEmbeddings")

    # Mock the split_documents method
    mocker.patch.object(
        chat_pdf.text_splitter, "split_documents", return_value=["chunk1", "chunk2"]
    )

    # Mock the Chroma.from_documents method
    mocker.patch(
        "data_scientist.rag.Chroma.from_documents", return_value="vector_store"
    )

    # Call the ingest method
    chat_pdf.ingest("test.pdf")

    # Assert that the PyPDFLoader is called with the correct file path
    data_scientist.rag.PyPDFLoader.assert_called_once_with(file_path="test.pdf")

    # Assert that the split_documents method is called with the loaded documents
    chat_pdf.text_splitter.split_documents.assert_called_once_with(
        data_scientist.rag.PyPDFLoader().load()
    )

    # Assert that the Chroma.from_documents method is called with the chunks and FastEmbedEmbeddings
    data_scientist.rag.Chroma.from_documents.assert_called_once_with(
        documents=["chunk1", "chunk2"],
        embedding=data_scientist.rag.FastEmbedEmbeddings(),
    )

    # Assert that the retriever and chain attributes are set correctly
    assert chat_pdf.retriever == "vector_store".as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={
            "k": 3,
            "score_threshold": 0.5,
        },
    )
    assert chat_pdf.chain is not None


def test_chat_pdf_ask(chat_pdf, mocker):
    # Set up the chain attribute
    chat_pdf.chain = mocker.MagicMock()
    chat_pdf.chain.invoke = mocker.MagicMock(return_value="Answer")

    # Call the ask method
    result = chat_pdf.ask("What is the answer?")

    # Assert that the chain.invoke method is called with the query
    chat_pdf.chain.invoke.assert_called_once_with("What is the answer?")

    # Assert that the result is correct
    assert result == "Answer"


def test_chat_pdf_clear(chat_pdf):
    # Set up the vector_store, retriever, and chain attributes
    chat_pdf.vector_store = mocker.MagicMock()
    chat_pdf.retriever = mocker.MagicMock()
    chat_pdf.chain = mocker.MagicMock()

    # Call the clear method
    chat_pdf.clear()

    # Assert that the vector_store, retriever, and chain attributes are set to None
    assert chat_pdf.vector_store is None
    assert chat_pdf.retriever is None
    assert chat_pdf.chain is None
