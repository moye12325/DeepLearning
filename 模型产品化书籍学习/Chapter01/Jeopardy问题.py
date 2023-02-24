import pandas
import pd
import os

cwd = os.getcwd()

# 导入Jeopardy问题
data = pd.read_csv(cwd + '/data/jeopady_questions/jeopardy_questions.csv')
data = pd.DataFrame(data=data)

# 小写字母 删除空白 查看列名字
data.columns = map(lambda x: x.lower().strip(), data.columns)

# 减少数据
data = data[0:1000]

# 标记解析Jeopardy问题
data["question_tokens"] = data["question"].apply(lambda x:nlp(x))