import faiss

def get_index(dimension, vectors, search='euclidean'):
		if search == 'euclidean':
				index = faiss.IndexFlatL2(dimension)
		elif search == 'cosine':
				index = faiss.IndexFlatIP(dimension)
				faiss.normalize_L2(vectors)
		else:
				raise ValueError('search must be euclidean or cosine')
		index.add(vectors)
		return index