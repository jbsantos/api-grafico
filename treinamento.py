from sklearn.cluster import KMeans

def objGrafico():
    import numpy as np
    import json

    class NumpyEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return json.JSONEncoder.default(self, obj)

    url = "https://apto-api-rest-ifpe.herokuapp.com/api/desafio-tecnico/rankearCandidatosSimplificado"
    #url = "https://run.mocky.io/v3/bd659a0b-5b5f-4989-b47d-657076841398"
    #url = "https://run.mocky.io/v3/61703339-173a-4f8d-b235-edfe2405242e"
    
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

    
    r = requests.get("https://apto-api-rest-ifpe.herokuapp.com/api/desafio-tecnico/rankearCandidatosSimplificado").json()
    #r = requests.get("https://run.mocky.io/v3/20963c21-73ca-406c-b12c-141c63aef532").json()
    #r = requests.get("https://run.mocky.io/v3/61703339-173a-4f8d-b235-edfe2405242e").json()
    notas = r['data']
    #print(notas)
    #print("notas")
    data = []
#print(notas[0]['candidatoNotasDtoList'])   
        
    for idx, val in enumerate(notas):
        
        #for count, nota in enumerate(val['candidatoNotasDtoList']):
        
        if (val['idDesafioTecnico'] <= 1000):
            #print(val['idDesafioTecnico'])
                for id, nt in enumerate(val['candidatoNotasDtoList']):
                    if nt['pontuacao'] is None or nt['nota1'] is None or nt['nota2'] is None :
                        print("null")
                    else:
                        #print(nt)
                        data.append(nt)
                
    print(data)
        #print(data)

    dataFrame = pd.DataFrame(data)
    #print(dataFrame)
    #print(data)
    dataFrame = pd.DataFrame(data)
    #print(dataFrame)
    #print("chegou")
    df = pd.DataFrame(dataFrame)
    #print(df)
    #print("df chegou")
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

    kmeans = KMeans(n_clusters=4, init='random')
    distancias = kmeans.fit_transform(X)
    legendas = kmeans.labels_

    with  open("kmeans_n_cluster4_padrao.pkl", "wb") as file: pickle.dump(kmeans, file)

    import matplotlib.pyplot as plt
    plt.scatter(X[:, 0], X[:,1], s= 100 , c = kmeans.labels_)
    plt.scatter(kmeans.cluster_centers_[:,0], kmeans.cluster_centers_[:, 1],  s=100, c= 'red', label = 'Centroids')
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
