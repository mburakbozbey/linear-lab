""" Tests for the RAG model. """

import os
import tempfile

import pytest
from fpdf import FPDF
from langchain_community.document_loaders import PyPDFLoader

from data_scientist.rag import ChatPDF


class MockDocument:
    """
    Mock document class for testing.
    """

    def __init__(self, content, metadata=None):
        self.page_content = content
        self.metadata = metadata if metadata else {}

    def public_method(self):
        """
        Public method for testing.
        """


@pytest.fixture(scope="session")
def pdf_file():
    """
    Fixture for creating a simple PDF file.
    """
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt="Capital of France is Paris.", ln=True, align="C")

    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as fp:
        pdf.output(fp.name)
        yield fp.name

    os.remove(fp.name)


@pytest.fixture
def chat_pdf():
    """
    Fixture for creating a ChatPDF object.
    """
    return ChatPDF()


def test_chat_pdf_ingest_with_valid_pdf(chat_pdf, mocker, pdf_file):
    """
    Test the ingest method of ChatPDF with a valid PDF file.
    """
    # Mock the PyPDFLoader.load method and specify a return value
    mocker.patch.object(
        PyPDFLoader, "load", return_value=[MockDocument("Capital of France is Paris.")]
    )

    chat_pdf.ingest(pdf_file)

    assert chat_pdf.retriever is not None
    assert chat_pdf.chain is not None


def test_chat_pdf_ask_with_ingest(chat_pdf, mocker, pdf_file):
    """
    Test the ask method of ChatPDF after ingesting a PDF file.
    """
    mocker.patch.object(
        PyPDFLoader, "load", return_value=[MockDocument("Capital of France is Paris.")]
    )

    chat_pdf.ingest(pdf_file)

    query = "What is the capital of France?"

    result = chat_pdf.ask(query)

    assert isinstance(result, str)
