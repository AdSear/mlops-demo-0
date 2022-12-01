import pickle
from google.cloud import storage
from config import MODEL_PATH

def hepatitis_c_predict(request):
    req_json = request.get_json()
    if req_json:
        storage_client = storage.Client()

        bucket = storage_client.bucket(MODEL_PATH.replace('gs://', '').split('/')[0])
        blob = bucket.blob("/".join(MODEL_PATH.replace('gs://', '').split('/')[1:]))
        pickle_in = blob.download_as_string()
        ml_model = pickle.loads(pickle_in)

        try:
            Ypredict = ml_model.predict([req_json['value']])
            dic_map = {0: "Blood Donor", 1: "Hepatitis", 2: "Fibrosis", 3: "Cirrhosis"}
            return dic_map[int(Ypredict[0])]
        
        except Exception as e:
            return e
    else:
        return 'hello World'

# def hello_world(request):
#     print(request)
#     print(request.get_json())
#     return 'hello World'
