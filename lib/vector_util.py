import numpy as np

def get_vectors_from_collections(cake_documents, vector_type):
    vectors = [cake_document[vector_type] for cake_document in cake_documents]

    return np.array(vectors).astype('float32')