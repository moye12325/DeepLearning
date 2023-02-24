from googletrans import Translator
translator = Translator(service_urls=[
      'translate.google.com/?hl=zh-CN',])# 如果可以上外网，还可添加 'translate.google.com' 等
trans=translator.translate('Hello World', src='en', dest='zh-cn')
# 原文
print(trans.origin)
# 译文
print(trans.text)