import requests
from flask import Flask, jsonify, request, render_template_string, render_template
from flask_caching import Cache  # Import Cache from flask_caching module
import time
from ner_train.test_ner_hl_v2 import PICO_Class

app = Flask(__name__)
app.config.from_object('config.Config')  # Set the configuration variables to the flask application
cache = Cache(app)  # Initialize Cache

pico_fetcher = PICO_Class()

@app.route('/')
@cache.cached(timeout=300)
def home():  # put application's code here
    return render_template('home.html') #'Hello World!'


@app.route('/', methods=['POST'])
def evidence_post_pico():
    claim = request.form['claim']
    claim = str(claim).strip()
    search_type = request.form['search_type']
    if search_type == 'Raw text':
        search_type = 'all'
    elif search_type == 'PICO':
        search_type = 'pico'
    pico_list = pico_fetcher.get_pico(claim)
    for item in pico_list:
        print('Found entity: {}\t=>\t{}'.format(item[0], item[1]))
    return render_template('home.html', title='Evidence Search PICO',
                           content='This is the evidence search page PICO',
                           query=claim,
                           search_type=search_type,
                           pico_list=pico_list)


@app.route("/universities")
@cache.cached(timeout=30, query_string=True)
def get_universities():
    API_URL = "http://universities.hipolabs.com/search?country="
    search = request.args.get('country')
    r = requests.get(f"{API_URL}{search}")
    return jsonify(r.json())


@app.route("/test")
@cache.cached(timeout=60, query_string=True)
def get_test():
    search = request.args.get('country')
    if cache.get(search) is not None:
        bar = cache.get(search)
    else:
        time.sleep(3)
        results = {'k': 'v', 'k2': 33}
        results2 = {'key': 'value', 'k2': 22}
        cache.set('haoliu', results)
        cache.set(search, results2)
        bar = cache.get(search)
    return render_template_string(
        "<html><body>foo cache: {{bar['key']}}</body></html>", bar=bar
    )


@app.route("/class")
@cache.cached(timeout=60, query_string=True)
def get_class():
    search = request.args.get('country')
    time.sleep(3)
    print(search)
    cls_list = [SentenceExample("haoliu", 0.98765), SentenceExample("sjdfljsdl", 0.8765)]
    return render_template_string(
        "<html><body>foo cache: {{cls_list[0].sentence}}  foo cache: {{cls_list[1].sent_score}}</body></html>", cls_list=cls_list
    )


@app.route("/pico")
@cache.cached(timeout=60, query_string=True)
def get_pico_test():
    query_text = request.args.get('pico')
    results = pico_fetcher.get_pico(query_text)
    return render_template_string(
        "<html><body>foo cache: {{results[0]}}  foo cache: {{results[1]}}</body></html>", results=results
    )


class SentenceExample():
    def __init__(self, sentence, sent_score):
        self.sentence = sentence
        self.sent_score = sent_score


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5555)
