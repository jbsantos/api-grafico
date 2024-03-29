import treinamento
from flask import Flask, request, render_template
from flask_cors import CORS

#app = Flask(__name__, static_url_path='/static')
app = Flask(__name__)
CORS(app)


@app.route('/')
def display_gui():

    treinamento.objGrafico()
    return render_template('template.html',classe='1')

@app.route('/verificar', methods=['GET'])
def verificar():
	return treinamento.imagem()

@app.route('/dados', methods=['GET'])
def dados():
	return treinamento.objGrafico()
	
if __name__ == "__main__":
        #app.run(debug=True)
	app.run()
