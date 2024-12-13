import requests

# 文件下载URL（需替换为实际链接）
url = "https://your-download-link"  # 例如 sandbox:/mnt/data/converted_algo_strategy.py
response = requests.get(url)

# 保存到本地文件系统
with open("converted_algo_strategy.py", "wb") as file:
    file.write(response.content)
print("文件已下载到当前目录下：converted_algo_strategy.py")
