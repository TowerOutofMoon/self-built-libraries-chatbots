from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.document_loaders import TextLoader
from langchain.chains import RetrievalQA, ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import TextLoader
from langchain.document_loaders import PyPDFLoader

import panel as pn
import param
import os

os.environ['OPENAI_API_KEY'] = 'sk-YwII0mU6W1M95li1xpHQnUB4rH8aeVsPVZgcLlpYBgMiX5J3'
os.environ['OPENAI_API_BASE'] = 'https://api.chatanywhere.com.cn/v1'
script_dir = os.path.dirname(os.path.realpath(__file__))

llm_name = "gpt-3.5-turbo"


def load_db(file, chain_type, k):
    """
    该函数用于加载 PDF 文件，切分文档，生成文档的嵌入向量，创建向量数据库，定义检索器，并创建聊天机器人实例。

    参数:
    file (str): 要加载的 PDF 文件路径。
    chain_type (str): 链类型，用于指定聊天机器人的类型。
    k (int): 在检索过程中，返回最相似的 k 个结果。

    返回:
    qa (ConversationalRetrievalChain): 创建的聊天机器人实例。
    """
    # 载入文档
    loader = PyPDFLoader(file)
    documents = loader.load()
    # 切分文档
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    docs = text_splitter.split_documents(documents)
    # 定义 Embeddings
    embeddings = OpenAIEmbeddings()
    # 根据数据创建向量数据库
    db = Chroma.from_documents(documents=docs, embedding=embeddings)
    # 定义检索器
    retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": k})
    # 创建 chatbot 链，Memory 由外部管理
    qa = ConversationalRetrievalChain.from_llm(
        llm=ChatOpenAI(model_name=llm_name, temperature=0),
        chain_type=chain_type,
        retriever=retriever,
        return_source_documents=True,
        return_generated_question=True,
    )
    return qa


# 用于存储聊天记录、回答、数据库查询和回复
class cbfs(param.Parameterized):
    chat_history = param.List([])
    answer = param.String("")
    db_query = param.String("")
    db_response = param.List([])

    def __init__(self, **params):
        super(cbfs, self).__init__(**params)
        self.panels = []
        self.loaded_file = os.path.join(script_dir, "docs/matplotlib/第一回：Matplotlib初相识.pdf")
        self.qa = load_db(self.loaded_file, "stuff", 4)

    # 将文档加载到聊天机器人中
    def call_load_db(self, count):
        """
        count: 数量
        """
        if count == 0 or file_input.value is None:  # 初始化或未指定文件 :
            return pn.pane.Markdown(f"载入文件: {self.loaded_file}")
        else:
            file_input.save("temp.pdf")  # 本地副本
            self.loaded_file = file_input.filename
            button_load.button_style = "outline"
            self.qa = load_db("temp.pdf", "stuff", 4)
            button_load.button_style = "solid"
        self.clr_history()
        return pn.pane.Markdown(f"载入文件: {self.loaded_file}")

    # 处理对话链
    def convchain(self, query):
        """
        query: 用户的查询
        """
        if not query:
            return pn.WidgetBox(pn.Row('你:', pn.pane.Markdown("", width=600)), scroll=True)
        result = self.qa({"question": query, "chat_history": self.chat_history})
        self.chat_history.extend([(query, result["answer"])])
        self.db_query = result["generated_question"]
        self.db_response = result["source_documents"]
        self.answer = result['answer']
        self.panels.extend([
            pn.Row('你:', pn.pane.Markdown(query, width=600)),
            pn.Row('机器人:', pn.pane.Markdown(self.answer, width=600, styles={'background-color': '#F6F6F6'}))
        ])
        inp.value = ''  # 清除时清除装载指示器
        return pn.WidgetBox(*self.panels, scroll=True)

    # 获取最后发送到数据库的问题
    @param.depends('db_query ', )
    def get_lquest(self):
        if not self.db_query:
            return pn.Column(
                pn.Row(pn.pane.Markdown(f"最后发送的问题:", styles={'background-color': '#F6F6F6'})),
                pn.Row(pn.pane.Str("无可用的数据库"))
            )
        return pn.Column(
            pn.Row(pn.pane.Markdown(f"查询:", styles={'background-color': '#F6F6F6'})),
            pn.pane.Str(self.db_query)
        )

    # 获取数据库返回的源文件
    @param.depends('db_response', )
    def get_sources(self):
        if not self.db_response:
            return
        rlist = [pn.Row(pn.pane.Markdown(f"查询结果:", styles={'background-color': '#F6F6F6'}))]
        for doc in self.db_response:
            rlist.append(pn.Row(pn.pane.Str(doc)))
        return pn.WidgetBox(*rlist, width=600, scroll=True)

    # 获取当前聊天记录
    @param.depends('convchain', 'clr_history')
    def get_chats(self):
        if not self.chat_history:
            return pn.WidgetBox(pn.Row(pn.pane.Str("无历史对话")), width=600, scroll=True)
        rlist = [pn.Row(pn.pane.Markdown(f"对话历史", styles={'background-color': '#F6F6F6'}))]
        for exchange in self.chat_history:
            rlist.append(pn.Row(pn.pane.Str(exchange)))
        return pn.WidgetBox(*rlist, width=600, scroll=True)

    # 清除聊天记录
    def clr_history(self, count=0):
        self.chat_history = []
        return

    # 初始化聊天机器人


cb = cbfs()

# 定义界面的小部件
file_input = pn.widgets.FileInput(accept='.pdf')  # PDF 文件的文件输入小部件
button_load = pn.widgets.Button(name="数据加载", button_type='primary')  # 加载数据库的按钮
button_clear_history = pn.widgets.Button(name="清除记录", button_type='warning')  # 清除聊天记录的按钮
button_clear_history.on_click(cb.clr_history)  # 将清除历史记录功能绑定到按钮上
inp = pn.widgets.TextInput(placeholder='开始对话')  # 用于用户查询的文本输入小部件

# 将加载数据库和对话的函数绑定到相应的部件上
bound_button_load = pn.bind(cb.call_load_db, button_load.param.clicks)
conversation = pn.bind(cb.convchain, inp)

jpg_pane = pn.pane.Image(os.path.join(script_dir, "img\\bg1.png"))

# 使用 Panel 定义界面布局
tab1 = pn.Column(
    pn.Row(inp),
    pn.layout.Divider(),
    pn.panel(conversation, loading_indicator=True, height=300),
    pn.layout.Divider(),
)
tab2 = pn.Column(
    pn.panel(cb.get_lquest),
    pn.layout.Divider(),
    pn.panel(cb.get_sources),
)
tab3 = pn.Column(
    pn.panel(cb.get_chats),
    pn.layout.Divider(),
)
tab4 = pn.Column(
    pn.Row(file_input, button_load, bound_button_load),
    pn.Row(button_clear_history, pn.pane.Markdown("清除历史对话信息")),
    pn.layout.Divider(),
    pn.Row(jpg_pane.clone(width=400))
)
# 将所有选项卡合并为一个仪表盘
dashboard = pn.Column(
    pn.Row(pn.pane.Markdown('# 使用你自己的PDF定制机器人')),
    pn.Tabs(('对话', tab1), ('详细查询信息', tab2), ('历史对话', tab3), ('配置', tab4))
)
