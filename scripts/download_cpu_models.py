import argparse
from pathlib import Path

from sentence_transformers import CrossEncoder, SentenceTransformer


DEFAULT_EMBEDDING = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
DEFAULT_RERANKER = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"


def main() -> None:
    parser = argparse.ArgumentParser(description="Download CPU-friendly multilingual RAG models.")
    parser.add_argument("--output", default="storage/models")
    parser.add_argument("--embedding", default=DEFAULT_EMBEDDING)
    parser.add_argument("--reranker", default=DEFAULT_RERANKER)
    args = parser.parse_args()

    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)

    embedding_path = output / args.embedding.rsplit("/", 1)[-1]
    reranker_path = output / args.reranker.rsplit("/", 1)[-1]

    embedding = SentenceTransformer(args.embedding)
    dimension = embedding.get_sentence_embedding_dimension()
    if dimension != 384:
        raise RuntimeError(f"Expected a 384-dimensional embedding model, got {dimension}")
    embedding.save(str(embedding_path))

    reranker = CrossEncoder(args.reranker, max_length=256)
    reranker.save(str(reranker_path))

    print(f"Embedding model: {embedding_path}")
    print(f"Reranker model: {reranker_path}")


if __name__ == "__main__":
    main()
