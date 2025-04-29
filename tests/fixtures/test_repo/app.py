from flask import Flask, render_template, jsonify

app = Flask(__name__)

@app.route('/')
def hello_world():
    """Renders the index page."""
    return render_template('index.html', message='Hello, World!')

@app.route('/about')
def about_page():
    """Renders the about page."""
    # Example symbol: local variable
    creator = "AI Assistant"
    return f"About this app, created by {creator}"

@app.route('/data')
def get_data():
    """Returns some sample JSON data."""
    # Example symbol: dictionary
    sample_data = {'id': 123, 'value': 'example'}
    return jsonify(sample_data)

if __name__ == '__main__':
    app.run(debug=True)