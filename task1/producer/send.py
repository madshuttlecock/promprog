#!/usr/bin/env python3
import pika

connection = pika.BlockingConnection(pika.ConnectionParameters(
        host='queue', port=5672))
        
        
        
channel = connection.channel()

channel.queue_declare(queue='hello')
print("Enter your string. Press 'enter' to exit")

s = input()
while(s != ""):
    channel.basic_publish(exchange='',
                              routing_key='hello',
                                                    body=s)                                                
    print(" [x] Sent", s)
    s = input()

connection.close()
