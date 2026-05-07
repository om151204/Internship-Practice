from flask import Flask, request, jsonify
import jwt, datetime

app = Flask(__name__)
app.config['SECRET_KEY'] = 'mysecretkey'

# Login route to generate token
@app.route('/login', methods=['POST'])
def login():
    data = request.get_json()
    if data['username'] == 'admin' and data['password'] == '1234':
        token = jwt.encode(
            {'user': data['username'], 'exp': datetime.datetime.utcnow() + datetime.timedelta(minutes=30)},
            app.config['SECRET_KEY'],
            algorithm='HS256'
        )
        return jsonify({'token': token})
    return jsonify({'message': 'Invalid credentials'}), 401

# Protected route
@app.route('/protected', methods=['GET'])
def protected():
    token = request.headers.get('Authorization').split()[1]
    try:
        decoded = jwt.decode(token, app.config['SECRET_KEY'], algorithms=['HS256'])
        return jsonify({'message': f'Welcome {decoded["user"]}'})
    except jwt.ExpiredSignatureError:
        return jsonify({'message': 'Token expired'}), 401
    except jwt.InvalidTokenError:
        return jsonify({'message': 'Invalid token'}), 401

if __name__ == '__main__':
    app.run(debug=True)
