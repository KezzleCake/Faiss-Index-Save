# -*- coding: utf-8 -*-

import pymongo
import os
import json
import faiss
from lib.faiss_util import get_index
from lib.vector_util import get_vectors_from_collections

conn = pymongo.MongoClient(host=os.environ.get('MONGO_HOST'), port=int(os.environ.get('MONGO_PORT')), username=os.environ.get('MONGO_USERNAME'), password=os.environ.get('MONGO_PASSWORD'))
db = conn[os.environ.get('MONGO_DBNAME')]

def get_cake_documents():
    return list(db.cakes.find({}, {'vit': 1, 'koclip': 1}))

index_save_path = os.environ.get('INDEX_SAVE_PATH')

def lambda_handler(event, context):
    cakes_documents = get_cake_documents()

    vit_vectors = get_vectors_from_collections(cakes_documents, 'vit')
    koclip_vectors = get_vectors_from_collections(cakes_documents, 'koclip')

    vit_index = get_index(1000, vit_vectors, 'euclidean')
    koclip_index = get_index(512, koclip_vectors, 'cosine')

    print('vit num of vectors: ', vit_index.ntotal)
    print('koclip num of vectors: ', koclip_index.ntotal)

    faiss.write_index(vit_index, index_save_path + 'vit.index')
    faiss.write_index(koclip_index, index_save_path + 'koclip.index')

    save_indices = os.listdir(index_save_path)
    print(save_indices)

    conn.close()

    return {
        'save_indices_result': json.dumps(save_indices)
    }