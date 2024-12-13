import chardet

# 读取文件的部分内容来检测编码
with open('C:\\Users\\MI\\Documents\\MSCI_ESG.dta', 'rb') as f:
    raw_data = f.read(10000)  # 读取前10,000个字节
    result = chardet.detect(raw_data)
    encoding = result['encoding']
    confidence = result['confidence']

print(f"Detected encoding: {encoding} with confidence: {confidence}")
