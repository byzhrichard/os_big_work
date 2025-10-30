from fastapi import FastAPI, Request, Form
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from transformers import BertTokenizer, BertForSequenceClassification
import torch
from pydantic import BaseModel
from pathlib import Path
from contextlib import asynccontextmanager

# 设置path
current_dir = Path(__file__).parent
model_path = current_dir.parent / "my_sentiment_model"
frontend_path = current_dir.parent / "frontend"
static_path = frontend_path / "static"


# 全局变量，用于存储模型和分词器
model = None
tokenizer = None
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备：{device}")

# 生命周期管理器
@asynccontextmanager # 应用的启动/关闭钩子
async def lifespan(app: FastAPI):
    # 启动时加载模型
    global model, tokenizer # 为了去修改全局变量
    print("正在加载分词器和模型...")
    tokenizer = BertTokenizer.from_pretrained(model_path)
    model = BertForSequenceClassification.from_pretrained(model_path)
    model.to(device)
    model.eval()  # 设置为评估模式
    print(f"模型加载完成！使用设备：{device}")

    # yield 之前的代码在“启动”阶段执行
    yield
    # yield 之后的代码在“关闭”阶段执行

    print("正在关闭应用...")


# 初始化FastAPI应用
app = FastAPI(title="中文情感分析系统", lifespan=lifespan)

# 设置模板和静态文件目录
templates = Jinja2Templates(directory=frontend_path) # Jinja2来管理frontend
app.mount("/static", StaticFiles(directory=static_path), name="static") # fastapi来管理静态文件


# 情感分析预测函数
def predict_sentiment(text: str) -> dict:
    inputs = tokenizer(
        text,
        padding=True, # 填充到batch中最长的那一条的长度
        truncation=True, # 超过 max_length 的文本会被截断
        max_length=128,
        return_tensors="pt"
    ).to(device)
    # print(inputs)
    # 'input_ids'中 101是[CLS], 102是[SEP]
    # 'token_type_ids'用来区分 句子A or 句子B
    # 'attention_mask'中 1 代表有效 token
    # {
    #   'input_ids': tensor([[ 101, 1922, 1962,  749,  106,  102]], device='cuda:0'),
    #   'token_type_ids': tensor([[0, 0, 0, 0, 0, 0]], device='cuda:0'),
    #   'attention_mask': tensor([[1, 1, 1, 1, 1, 1]], device='cuda:0')
    # }

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probabilities = torch.softmax(logits, dim=-1)
        predictions = torch.argmax(logits, dim=-1)

    label = int(predictions[0].item())
    confidence = float(probabilities[0][label].item())
    sentiment_label = "正面" if label == 1 else "负面"

    return {
        "text": text,
        "sentiment": sentiment_label, # 情感倾向
        "confidence": round(confidence, 4), # 置信度
        "label": label
    }


# 定义请求数据模型（用于API接口）
class TextRequest(BaseModel):
    text: str


# 主页面路由（返回HTML页面）
# 注册 GET 根路由 /, 声明返回 HTML
@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    # 用 Jinja2Templates 渲染模板文件 index.html
    return templates.TemplateResponse("index.html", {
        "request": request
    })


# 当用户在网页上输入文本并点击“提交”按钮（POST 表单）时，执行这个函数
# user_text: 自动从 HTML <form> 的输入框中读取字段名为 user_text 的内容
@app.post("/", response_class=HTMLResponse)
async def predict_from_web(request: Request, user_text: str = Form(...)):
    result = predict_sentiment(user_text)
    # 将预测结果和原文本重新传回模板
    return templates.TemplateResponse("index.html", {
        "request": request,
        "result": result,
        "user_text": user_text
    })

# 纯 API 接口
# 1.客户端发送一个 POST /api/predict 请求
# 2.FastAPI 会自动将请求体解析为 TextRequest
@app.post("/api/predict")
async def predict_from_api(request: TextRequest):
    try:
        result = predict_sentiment(request.text)
        return result
    except Exception as e:
        return {"error": str(e)}


