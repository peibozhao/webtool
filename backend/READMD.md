
# 接口:
## POST /copy/submit:
* 请求body: json
  * text: 要保存的文本
* 响应body: json
  * code: 短字符码
## GET /copy/retrieve:
* 参数: code
* 响应body: json
  * text: 文本

