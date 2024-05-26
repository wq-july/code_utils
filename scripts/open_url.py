# import subprocess
# import time

# # 设置要打开的网站
# url = 'https://baidu.com'

# # 调用外部谷歌浏览器打开指定网站
# chrome_path = r'C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe'
# # 打开一个新窗口并访问指定网站
# chrome_process = subprocess.Popen([chrome_path, '--new-window', url])

# # 等待浏览器加载页面
# time.sleep(2)

# # 循环刷新网站10次
# for i in range(10):
#     print(f'正在刷新网站：{url}，第 {i+1} 次')
#     # 刷新当前已打开的窗口
#     subprocess.run([chrome_path, '--new-window', url, '--refresh'])
#     time.sleep(1)  # 可选：等待一段时间再刷新页面，避免刷新过快

# # 关闭浏览器窗口
# chrome_process.kill()
# print('网页刷新完成！')


import time
from selenium import webdriver

def refresh_page(url, refresh_time, num_refresh):
    driver = webdriver.Chrome()  # 使用Chrome浏览器
    driver.get(url)  # 打开指定网页
    for _ in range(num_refresh):  # 循环执行指定次数
        time.sleep(refresh_time)  # 等待指定的刷新时间
        driver.refresh()  # 刷新页面

# 使用示例
if __name__ == '__main__':
    url = 'https://baidu.com'
    refresh_time = 1  # 刷新时间间隔，单位为秒
    num_refresh = 10  # 循环执行的次数
    refresh_page(url, refresh_time, num_refresh)