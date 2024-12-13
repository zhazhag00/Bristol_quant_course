from kafka import KafkaConsumer
import json

# Kafka brokers地址列表
brokers = ['10.0.2.15:29092', '10.0.2.15:29093', '10.0.2.15:29094']

# 创建消费者实例
consumer = KafkaConsumer(
    'test-topic',  # 主题名称
    bootstrap_servers=brokers,  # Kafka brokers的地址列表
    auto_offset_reset='earliest',  # 从最早的消息开始读取
    enable_auto_commit=True,  # 自动提交偏移量
    group_id='my-consumer-group',  # 消费者组ID
    value_deserializer=lambda x: json.loads(x.decode('utf-8'))  # 反序列化消息为JSON格式
)

# 读取消息
for message in consumer:
    print(f"Received message: {message.value}")

# 关闭消费者
consumer.close()
