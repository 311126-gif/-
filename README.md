"""Scam-Trapper 主程式：LINE Bot + REST API"""
import os
import uuid
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv

from scam_detector import ScamDetector
from trapper_agent import TrapperAgent
from intel_extractor import IntelExtractor

load_dotenv()

# 全域狀態（單機版用 dict，生產環境改用 Redis）
sessions: dict[str, dict] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """啟動 / 關閉時的初始化"""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("❌ OPENAI_API_KEY 未設定，請檢查 .env 檔案")

    app.state.detector = ScamDetector(api_key)
    app.state.extractor = IntelExtractor(api_key)
    print("✅ Scam-Trapper 啟動完成")
    yield
    print("👋 Scam-Trapper 關閉")


app = FastAPI(title="Scam-Trapper API", lifespan=lifespan)


# === API Schema ===
class ChatRequest(BaseModel):
    session_id: str | None = None
    message: str  # 模擬詐騙者訊息


class ChatResponse(BaseModel):
    session_id: str
    is_scam: bool
    scam_type: str
    confidence: float
    agent_reply: str
    should_takeover: bool


class IntelResponse(BaseModel):
    session_id: str
    intel: dict
    full_conversation: str


def get_or_create_session(session_id: str | None, api_key: str) -> tuple[str, dict]:
    """取得或建立新的 session"""
    if session_id and session_id in sessions:
        return session_id, sessions[session_id]
    new_id = session_id or str(uuid.uuid4())[:8]
    sessions[new_id] = {
        "agent": TrapperAgent(api_key=api_key),
        "is_takeover": False,
    }
    return new_id, sessions[new_id]


# === API Endpoints ===
@app.get("/")
async def root():
    """根路徑健康檢查"""
    return {
        "service": "Scam-Trapper",
        "status": "running",
        "version": "0.1.0",
        "endpoints": [
            "POST /chat — 與 AI 對話（模擬詐騙者）",
            "POST /intel — 抽取累積的犯罪情報",
            "GET /session/{id} — 取得 session 對話紀錄",
        ]
    }


@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    """模擬一段對話：使用者傳入詐騙者訊息，系統回傳裝傻回覆"""
    api_key = os.getenv("OPENAI_API_KEY")
    session_id, session = get_or_create_session(req.session_id, api_key)
    agent: TrapperAgent = session["agent"]

    # 1. 偵測是否為詐騙
    messages_so_far = [t.content for t in agent.dialogue] + [req.message]
    detection = await app.state.detector.detect(messages_so_far)

    # 2. 判斷是否觸發 AI 接管
    should_takeover = detection.is_scam and detection.confidence > 0.7
    session["is_takeover"] = should_takeover

    # 3. 如果是詐騙，讓 Agent 回覆裝傻
    if should_takeover:
        reply = await agent.respond(req.message)
    else:
        reply = "[未啟動 AI 接管]"

    return ChatResponse(
        session_id=session_id,
        is_scam=detection.is_scam,
        scam_type=detection.scam_type.value,
        confidence=detection.confidence,
        agent_reply=reply,
        should_takeover=should_takeover,
    )


@app.post("/intel", response_model=IntelResponse)
async def get_intel(session_id: str):
    """抽取累積的犯罪情報"""
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")

    session = sessions[session_id]
    agent: TrapperAgent = session["agent"]
    conversation = agent.get_full_conversation()

    if not conversation.strip():
        raise HTTPException(status_code=400, detail="Session 沒有任何對話紀錄")

    intel = await app.state.extractor.extract(conversation)
    return IntelResponse(
        session_id=session_id,
        intel=intel.model_dump(),
        full_conversation=conversation,
    )


@app.get("/session/{session_id}")
async def get_session(session_id: str):
    """取得 session 的完整對話紀錄"""
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")

    session = sessions[session_id]
    agent: TrapperAgent = session["agent"]

    return {
        "session_id": session_id,
        "is_takeover": session["is_takeover"],
        "stage": agent.stage.value,
        "dialogue": [turn.__dict__ for turn in agent.dialogue],
        "turn_count": len(agent.dialogue),
    }


@app.get("/sessions")
async def list_sessions():
    """列出所有活躍的 session（Debug 用）"""
    return {
        "active_sessions": [
            {
                "session_id": sid,
                "is_takeover": s["is_takeover"],
                "turn_count": len(s["agent"].dialogue),
            }
            for sid, s in sessions.items()
        ]
    }


if __name__ == "__main__":
    import uvicorn
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run(app, host=host, port=port)
