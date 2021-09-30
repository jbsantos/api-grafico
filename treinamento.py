from sklearn.cluster import KMeans

def objGrafico():
    import numpy as np
    import json

    class NumpyEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return json.JSONEncoder.default(self, obj)

    #url = "https://apto-api-rest-ifpe.herokuapp.com/api/desafio-tecnico/rankearCandidatosSimplificado"
    #url = "https://run.mocky.io/v3/3b892b3c-3ca8-42db-92d9-b2c164064c61"
    url = "https://run.mocky.io/v3/61703339-173a-4f8d-b235-edfe2405242e"
    
    infoNotas = salvaDados()
    a = np.array(kmeansLabel())
    img_json = np.array(jsonImagem())
    json_dump = json.dumps({'legenda': a, 'imagem': img_json, 'infoNotas': infoNotas, 'url': url}, cls=NumpyEncoder)

    return json_dump

def salvaDados():

    import numpy as np
    import requests
    import pandas as pd
    import json
    import re

    
    #r = requests.get("https://apto-api-rest-ifpe.herokuapp.com/api/desafio-tecnico/rankearCandidatosSimplificado").json()
    #r = requests.get(" https://run.mocky.io/v3/3b892b3c-3ca8-42db-92d9-b2c164064c61").json()
    r = requests.get("https://run.mocky.io/v3/61703339-173a-4f8d-b235-edfe2405242e").json()
    notas = r['data']
    print(notas)
    print("notas")
    data = r['data']
    for k, v in data.items():
        print(k, v)
        
    for idx, val in enumerate(r['data']):
        #data = val['candidatoNotasDtoList']
        data.append(val['candidatoNotasDtoList'][idx])
        print(idx)
     
    dataFrame = pd.DataFrame(data)
    print(dataFrame)
    print("chegou")
    df = pd.DataFrame(dataFrame)
    print(df)
    print("df chegou")
    df.to_csv(index=False)
    df.to_csv('aptoClassificacao/Apto_KNN.csv1', index=False, encoding='utf-8')  
    print(df)
    return r
 

def kmeansLabel():

    import numpy as np
    import pickle
    import pandas as pd
    apto = pd.read_csv('aptoClassificacao/Apto_KNN.csv1')
    X = apto.iloc[:, 2:4].values

    kmeans = KMeans(n_clusters=2, init='random')
    distancias = kmeans.fit_transform(X)
    legendas = kmeans.labels_

    with  open("kmeans_n_cluster4_padrao.pkl", "wb") as file: pickle.dump(kmeans, file)

    import matplotlib.pyplot as plt
    plt.scatter(X[:, 0], X[:,1], s= 100 , c = kmeans.labels_)
    plt.scatter(kmeans.cluster_centers_[:,0], kmeans.cluster_centers_[:, 1],  s=100, c= 'blue', label = 'Centroids')
    plt.title('APTO Clusters Questionário vs Programação')
    plt.xlabel('Questionario')
    plt.ylabel('Programacao')
    plt.legend()
    plt.savefig("static/grafico.jpeg")
    plt.close()
    return legendas


def jsonImagem():

    import base64
    data = {}

    with open('static/grafico.jpeg', mode='rb') as file:
        img = file.read()
    data['img'] = base64.encodebytes(img).decode('utf-8').replace('\n', '')

    return data
