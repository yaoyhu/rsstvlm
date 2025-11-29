import asyncio
import base64
import io

import streamlit as st
from openai import OpenAI
from PIL import Image
from rsstvlm.agent.workflow import AgentWorkflow, StreamEvent
from rsstvlm.utils import (
    LLM_MODEL,
    QWEN3_VL_30B_API_BASE,
    qwen3_vl_30b_function,
)

client = OpenAI(api_key="EMPTY", base_url=QWEN3_VL_30B_API_BASE)

st.set_page_config(
    page_title="天空地一体化超光谱遥感应用工程实验室",
    page_icon="🛰️",
    layout="wide",
)


# ======================
# 🤖 Agent 初始化 (cached)
# ======================
@st.cache_resource
def get_agent():
    """Initialize agent once and cache it."""
    return asyncio.run(
        AgentWorkflow.create(qwen3_vl_30b_function, timeout=120, verbose=True)
    )


# ======================
# 🧩 侧边栏功能
# ======================
with st.sidebar:
    st.header("🔧 控制面板")

    if st.button("🗑️ 清空当前对话"):
        st.session_state.messages = []
        st.rerun()

    st.divider()

    # Agent mode toggle
    use_agent = st.toggle("🤖 启用 Agent 模式", value=False)

    if use_agent:
        with st.spinner("正在加载 Agent..."):
            try:
                agent = get_agent()
                st.success("Agent 已就绪 ✅")
                # 显示可用工具
                st.markdown("#### 🛠️ 可用工具")
                for tool in agent.tools:
                    with st.expander(f"📦 {tool.metadata.name}"):
                        st.markdown(f"**描述:** {tool.metadata.description}")
                        if tool.metadata.fn_schema:
                            st.markdown("**参数:**")
                            schema = (
                                tool.metadata.fn_schema.model_json_schema()
                            )
                            if "properties" in schema:
                                for param, info in schema[
                                    "properties"
                                ].items():
                                    param_type = info.get("type", "any")
                                    param_desc = info.get("description", "")
                                    st.markdown(
                                        f"- `{param}` ({param_type}): {param_desc}"
                                    )
            except Exception as e:
                st.error(f"Agent 加载失败: {e}")
                use_agent = False

    st.divider()
    st.markdown("### 📌 使用说明")
    st.markdown("""
    - 启用 Agent 可以查看并使用工具
    - 目前工具较少，后续会完善
    """)  # noqa: RUF001

# ======================
# 💬 聊天历史初始化
# ======================
if "messages" not in st.session_state:
    st.session_state.messages = []

# 显示历史消息
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        if isinstance(message["content"], list):
            # 多模态消息
            for part in message["content"]:
                if part["type"] == "text":
                    st.markdown(part["text"])
                elif part["type"] == "image_url":
                    st.image(part["image_url"]["url"], width=300)
        else:
            st.markdown(message["content"])

# ======================
# 🖼️ 图像上传 + 文本输入
# ======================
uploaded_file = st.file_uploader("📎 上传图像/视频")

# 检查是否已有太多消息(防过载)
if len(st.session_state.messages) >= 10:
    st.warning("对话较长，建议点击侧边栏「清空当前对话」以获得最佳体验。")  # noqa: RUF001

prompt = st.chat_input("请输入你的问题")


# ======================
# 🔄 Agent 调用函数
# ======================
async def run_agent_stream(agent: AgentWorkflow, query: str):
    """Run agent and yield streaming events."""
    handler = agent.run(input=query)
    async for event in handler.stream_events():
        if isinstance(event, StreamEvent):
            yield event.delta
    # Ensure handler completes
    await handler


def run_agent(query: str) -> str:
    """Run agent synchronously with streaming output."""
    agent = get_agent()

    async def collect_response():
        full_response = ""
        async for delta in run_agent_stream(agent, query):
            full_response += delta
        return full_response

    return asyncio.run(collect_response())


def run_agent_with_placeholder(query: str, placeholder) -> str:
    """Run agent with live streaming to a placeholder."""
    agent = get_agent()

    async def stream_to_placeholder():
        full_response = ""
        handler = agent.run(input=query)
        async for event in handler.stream_events():
            if isinstance(event, StreamEvent):
                full_response += event.delta
                placeholder.markdown(full_response + "▌")
        await handler
        return full_response

    return asyncio.run(stream_to_placeholder())


# ======================
# 🤖 处理用户输入
# ======================
if prompt:
    # 构建用户内容
    user_content = [{"type": "text", "text": prompt}]

    # 如果上传了图片
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        buffered = io.BytesIO()
        image.save(buffered, format="JPEG")
        img_str = base64.b64encode(buffered.getvalue()).decode()
        image_url = f"data:image/jpeg;base64,{img_str}"
        user_content.insert(
            0, {"type": "image_url", "image_url": {"url": image_url}}
        )

    # 显示用户消息
    with st.chat_message("user"):
        if uploaded_file is not None:
            st.image(image, width=300)
        st.markdown(prompt)

    # 添加到历史
    st.session_state.messages.append({"role": "user", "content": user_content})

    # ======================
    # 🧠 调用大模型或Agent
    # ======================
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""

        try:
            if use_agent:
                # Use Agent with streaming
                full_response = run_agent_with_placeholder(
                    prompt, message_placeholder
                )
                message_placeholder.markdown(full_response)
            else:
                # Use direct LLM call
                # TODO: will be deprecated while agent 1st demo is released
                response = client.chat.completions.create(
                    model=LLM_MODEL,
                    messages=[
                        {"role": m["role"], "content": m["content"]}
                        for m in st.session_state.messages
                    ],
                    stream=True,
                )

                for chunk in response:
                    delta = chunk.choices[0].delta.content or ""
                    full_response += delta
                    message_placeholder.markdown(full_response + "▌")

                message_placeholder.markdown(full_response)

        except Exception as e:
            error_msg = f"❌ 模型调用失败: {e!s}"
            message_placeholder.error(error_msg)
            full_response = error_msg

    # 添加助手回复
    st.session_state.messages.append(
        {"role": "assistant", "content": full_response}
    )
