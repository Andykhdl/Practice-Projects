# Import Flask module
from flask import Flask

# Create an instance of the Flask class
app = Flask(__name__)

# Define a route for the homepage
@app.route('/')
def home():
    return "Hello, World! Welcome to Flask!"
@app.route('/about')
def about():
    return "This is the About page"
@app.route('/greet/<name>')
def greet(name):
    return f"Hello, {name}!"

# Run the app if this file is executed directly
if __name__ == "__main__":
    app.run(debug=True)
