"""Upload lifecycle regressions using real, local ephemeral Chroma collections."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import chromadb
import pytest
from chromadb.config import Settings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from module5.document_upload import DocumentUploader, UploadValidationError


class DeterministicEmbeddings(Embeddings):
    """Tiny embeddings with an injectable failure and no model or API access."""

    def __init__(self):
        self.fail = False

    def embed_documents(self, texts):
        if self.fail:
            raise RuntimeError("simulated upload embedding failure")
        return [self.embed_query(text) for text in texts]

    def embed_query(self, text):
        return [float(len(text)), float(sum(text.encode("utf-8")) % 101), 1.0]


def _collection_names(client):
    return {collection.name for collection in client.list_collections()}


@pytest.fixture
def chroma_client(monkeypatch):
    monkeypatch.setenv("ANONYMIZED_TELEMETRY", "False")
    client = chromadb.EphemeralClient(Settings(anonymized_telemetry=False))
    existing = _collection_names(client)
    yield client
    # Clean up only collections created by this test, even against the buggy code.
    for name in _collection_names(client) - existing:
        client.delete_collection(name)


def _sources(uploader):
    return {document.metadata["source"] for document in uploader.search("upload", k=20)}


def test_simultaneous_uploaders_keep_search_results_isolated(chroma_client):
    alice = DocumentUploader(DeterministicEmbeddings())
    bob = DocumentUploader(DeterministicEmbeddings())

    with ThreadPoolExecutor(max_workers=2) as pool:
        uploads = [
            pool.submit(alice.index_documents, b"Alice has a private research note.", "alice.txt"),
            pool.submit(bob.index_documents, b"Bob has a separate private research note.", "bob.txt"),
        ]
        assert [upload.result() for upload in uploads] == [1, 1]

    assert _sources(alice) == {"alice.txt"}
    assert _sources(bob) == {"bob.txt"}
    assert {doc.metadata["source"] for doc, _ in alice.search_with_scores("upload", k=20)} == {"alice.txt"}
    assert {doc.metadata["source"] for doc, _ in bob.search_with_scores("upload", k=20)} == {"bob.txt"}
    assert alice.vectorstore._collection.name != bob.vectorstore._collection.name
    assert not alice.vectorstore._client.get_settings().is_persistent
    assert not bob.vectorstore._client.get_settings().is_persistent


def test_replacement_removes_old_documents_and_owned_collections(chroma_client):
    existing = _collection_names(chroma_client)
    uploader = DocumentUploader(DeterministicEmbeddings())
    uploader.index_documents(b"The first upload contains an obsolete fact.", "first.txt")
    previous_name = uploader.vectorstore._collection.name

    for filename in ("second.txt", "third.md"):
        assert uploader.index_documents(b"The replacement contains only a current fact.", filename) == 1
        current_name = uploader.vectorstore._collection.name
        assert _sources(uploader) == {filename}
        assert uploader.vectorstore._collection.count() == uploader.num_chunks == 1
        assert current_name != previous_name
        assert previous_name not in _collection_names(chroma_client)
        assert _collection_names(chroma_client) == existing | {current_name}
        previous_name = current_name

    reported_chunks = uploader.chunks
    reported_chunks.clear()
    assert uploader.num_chunks == 1


def test_clear_deletes_only_its_collection_and_preserves_other_uploader(chroma_client):
    alice = DocumentUploader(DeterministicEmbeddings())
    bob = DocumentUploader(DeterministicEmbeddings())
    alice.index_documents(b"Alice has a private research note.", "alice.txt")
    bob.index_documents(b"Bob has a separate private research note.", "bob.txt")
    alice_name = alice.vectorstore._collection.name
    bob_name = bob.vectorstore._collection.name

    alice.clear()
    alice.clear()  # Clearing an already cleared upload is harmless.

    assert alice_name not in _collection_names(chroma_client)
    assert bob_name in _collection_names(chroma_client)
    assert not alice.is_ready
    assert alice.filename == ""
    assert alice.num_chunks == 0
    assert alice.chunks == []
    assert alice.vectorstore is None
    assert alice.search("upload") == []
    assert alice.search_with_scores("upload") == []
    assert bob.is_ready
    assert _sources(bob) == {"bob.txt"}


@pytest.mark.parametrize("failure_stage", ["parse", "chunk", "index"])
def test_failed_replacement_preserves_last_successful_upload(chroma_client, failure_stage):
    embeddings = DeterministicEmbeddings()
    uploader = DocumentUploader(embeddings)
    uploader.index_documents(b"The previous upload must remain searchable.", "previous.txt")
    previous_store = uploader.vectorstore
    previous_chunks = uploader.chunks
    previous_names = _collection_names(chroma_client)

    content = b"The failed replacement must never become the active upload."
    options = {}
    if failure_stage == "parse":
        content = b" \n\n "
        error = UploadValidationError
    elif failure_stage == "chunk":
        options = {"chunk_size": 10, "chunk_overlap": 20}
        error = ValueError
    else:
        embeddings.fail = True
        error = RuntimeError

    with pytest.raises(error):
        uploader.index_documents(content, "failed.txt", **options)

    assert uploader.filename == "previous.txt"
    assert uploader.chunks == previous_chunks
    assert uploader.vectorstore is previous_store
    assert uploader.is_ready
    assert _sources(uploader) == {"previous.txt"}
    assert _collection_names(chroma_client) == previous_names


def test_failed_first_index_leaves_no_active_upload_or_collection(chroma_client):
    existing = _collection_names(chroma_client)
    embeddings = DeterministicEmbeddings()
    embeddings.fail = True
    uploader = DocumentUploader(embeddings)

    with pytest.raises(RuntimeError, match="simulated upload embedding failure"):
        uploader.index_documents(b"An upload that cannot be embedded must not be published.", "failed.txt")

    assert uploader.filename == ""
    assert uploader.chunks == []
    assert uploader.vectorstore is None
    assert not uploader.is_ready
    assert _collection_names(chroma_client) == existing


@pytest.mark.parametrize("collection_name", ["langchain", "external-upload-isolation"])
@pytest.mark.parametrize("operation", ["clear", "replace"])
def test_unowned_default_and_external_collections_are_preserved(
    chroma_client, collection_name, operation
):
    embeddings = DeterministicEmbeddings()
    external_document = Document(
        page_content="This collection belongs to another component.",
        metadata={"source": "external.txt"},
    )
    external = Chroma.from_documents(
        [external_document], embedding=embeddings, collection_name=collection_name
    )
    uploader = DocumentUploader(embeddings)
    # A reference alone does not grant ownership of a pre-existing collection.
    uploader._vectorstore = external
    uploader._chunks = [external_document]
    uploader._filename = "external.txt"

    if operation == "clear":
        uploader.clear()
    else:
        uploader.index_documents(b"Only the uploader owns this new replacement.", "replacement.txt")
        assert _sources(uploader) == {"replacement.txt"}

    assert collection_name in _collection_names(chroma_client)
    assert external._collection.count() == 1
    assert external.similarity_search("external", k=20)[0].metadata["source"] == "external.txt"
