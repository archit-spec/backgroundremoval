# flowerconfig.py
# RabbitMQ management api
broker_api = 'http://guest:guest@localhost:15672/api/'

# Enable debug logging
logging = 'DEBUG'

# Web server address
address = '0.0.0.0'
port = 5555

# Refresh dashboards automatically
auto_refresh = True

# Run the Flower process in the background as a daemon
daemon = False

# Real-time charts
max_tasks = 10000

# Custom persistent storage
db = './flower.db'

# Enable basic auth
basic_auth = ['username:password']