import pika
from pymongo import MongoClient
import time

def callback(ch, method, properties, body):
    print("Received")
    post = {str(body): str(body)}
#    posts = db.posts
#    post_id = posts.insert_one(post).inserted_id
#    print (" [x] Received %r" % (body,))
#    print("Post_id:", post_id)
    db.test_collection.save(post)
    print("Saved")

print("START")

time.sleep(30)
#        connection = pika.BlockingConnection(pika.URLParameters("amqp://guest:guest@queue:5672"))
connection = pika.BlockingConnection(pika.ConnectionParameters(host='queue', port=5672))
#amqp://guest:guest@queue:5672"))  
#        channel = connection.channel()
#        client = MongoClient(host='mongo')
print("HERE")
#exit(0)

client = MongoClient(host='database')
channel = connection.channel()

client.drop_database("test_database")
db = client.test_database

db.drop_collection('test_collection')
collection = db.test_collection

channel.queue_declare(queue='hello')

channel.basic_consume(callback,
                      queue='hello',
                                     no_ack=True)
print (' [*] Waiting for messages. To exit press CTRL+C')
channel.start_consuming()
