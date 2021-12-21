from flask import Flask, request, render_template, Response, send_file
from main import *
from flask_cors import CORS

app = Flask(__name__)

@app.route('/')
def home():
    return '<p>Hello, World!</p>'


@app.route('/images')
def get_image():
    
    if request.args.get('type') == '1':
       filename = 'images/categoriesCountryCount2019-21mean.png'
    else:
       filename = 'images/clusterAnalyze4HierarchicalClustering2019-21.png'
    return send_file(filename, mimetype='png')
        
# @app.get('/images')
# def create_transactions_by_month_image():
#     image_name = request.args.get('name')
#     if not image_name:
#         image_name = 'transactions_by_month.png'
#         transaction_img_path = Transactions.create_transactions_by_month_image(image_name)
#         return send_file(transaction_img_path)


# enable CORS
CORS(app)

if __name__ == "__main__":
    os.environ['FLASK_ENV'] = 'development'
    app.run(host='127.0.0.1', port=3000, debug=True)
