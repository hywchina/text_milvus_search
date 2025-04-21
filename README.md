## 项目目录如下

```shell
.
├── data
│   ├── 红楼梦.txt
│   ├── 三国演义.txt
│   ├── 水浒传.txt
│   └── 西游记.txt
├── README.md
├── requirements.txt
└── src
    ├── embedding_service.py
    ├── ingest.py
    ├── search_api.py
    ├── search_streamlit.py
    └── test_api.py

## 说明
data目录：需要存在milvus数据库的所有文档（后期会对内容进行增删改）
requirements.txt：本项目需要安装的依赖

└── src
    ├── embedding_service.py: 本地部署embedding 服务，embedding 模型使用 “BAAI/bge-large-zh-v1.5”或其他
    ├── ingest.py : 进行数据库插入操作（落表），对数据进行预处理，并进行Chunk切分，并使用embedding服务进行mebedding抽取，然后插入到数据库中（需要设计避免重复插入）， 每个文档基本属性有：名称，日期，路径，chunkid等信息；
    ├── search_api.py ：基于已落表的数据库进行api搭建（可以使用fastapi）
    ├── search_streamlit.py：基于search api 搭建基于streamlit的前端页面
    └── test_api.py：测试 search api 的使用情况
```

## 步骤 
1. 启动embedding 服务
uvicorn embedding_service:app --host 0.0.0.0 --port 8001

2. Milvus Standalone 本地部署 && 数据可视化（attu）
        # Download the installation script
        $ curl -sfL https://raw.githubusercontent.com/milvus-io/milvus/master/scripts/standalone_embed.sh -o standalone_embed.sh

        # Start the Docker container
        $ bash standalone_embed.sh start

        # Stop Milvus
        $ bash standalone_embed.sh stop

        # Delete Milvus data
        $ bash standalone_embed.sh delete

    链接attu：
    docker run -p 8000:3000 -e MILVUS_URL=172.18.20.155:19530 zilliz/attu:v2.5


2. 插入数据
python src/ingest.py

3. 启动search_api 
uvicorn src.search_api:app --host 0.0.0.0 --port 8002

4. 测试api 服务 
python test/test_search_api.py 

5. 启动streamlit 前端
streamlit run src/streamlit_milvus.py



## 待办
- [ ] 各脚本涉及到的超参数统一配置
- [ ] docker 部署
- [ ] 

