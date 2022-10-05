from flask import Flask
app = Flask(__name__)

@app.route('/')
def ggg():
    return 'ggg'

if __name__ == '__main__':
    app.run()