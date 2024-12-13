from kafka import KafkaProducer
import json
import time

# Kafka brokers地址列表
brokers = ['10.0.2.15:29092', '10.0.2.15:29093', '10.0.2.15:29094']

# 创建生产者实例
producer = KafkaProducer(
    bootstrap_servers=brokers,  # Kafka brokers的地址列表
    value_serializer=lambda v: json.dumps(v).encode('utf-8')  # 序列化消息为JSON格式
)

# 发送消息
for i in range(10):
    message = {'message_number': i, 'content': f'Message {i}'}
    future = producer.send('test-topic', value=message)
    result = future.get(timeout=10)
    print(f"Message {i} sent to partition {result.partition} at offset {result.offset}")
    time.sleep(1)

# 关闭生产者
producer.close()
