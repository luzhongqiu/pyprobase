# pyprobase
implement probase algorithm


整个代码分为3个部分， 每个部分通过进程队列传输数据
- master
- worker
- calculate

其中calculate必须只有1个，master一般情况只有1个，worker按照cpu调整

---

`master` 负责
- 读取文件
- 断点续读

`woker` 负责
- 抽取名词
- 正则匹配

`calculate`负责
- super concept detection
- sub concep detection

---
- 运行:
```
python main.py ${corpus_path}
```
- 输出数据在data文件夹下
- master每10w存储下当前状态
- calculate每2分钟存储下当前的状态和结果
- spacy最好是1.x版，速度快，需要下载en模型
```
python -m spacy download en
```
- nltk 需要下载模型
```
import nltk
nltk.download('wordnet')
```

---
todo
- [] local merge
- [] Horizontal merge
- [] hierarchy merge